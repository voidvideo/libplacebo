// Compile the production lifecycle with allocator substitution in this test
// target only. No runtime/environment fault-injection control is introduced.
#undef calloc
#undef free
#include <stdlib.h>
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <libplacebo/frame_mixer.h>

void *mixer_test_calloc(size_t n, size_t size);
void mixer_test_free(void *ptr);
static unsigned allocations, fail_next;
void *mixer_test_calloc(size_t n, size_t size)
{
    if (fail_next) { fail_next = 0; return NULL; }
    void *p = calloc(n, size);
    if (p) allocations++;
    return p;
}
void mixer_test_free(void *p)
{
    if (p) { assert(allocations); allocations--; }
    free(p);
}

enum { SLOTS = 8, CYCLES = 1000 };
struct module;
struct instance {
    struct module *module;
    bool live;
    uint64_t revision;
    int setting;
};
struct ticket {
    const struct pl_mixer_job_params *params;
    struct instance *instance;
    enum pl_mixer_status result;
    unsigned polls;
    uint64_t epoch, content;
};
struct owner { int refs, retains, releases, ticket_releases; };
struct module {
    unsigned refs, creates, destroys, submits, cleanups;
    enum pl_mixer_status negotiation_result, creation_result, submission_result;
    struct pl_mixer_requirements requirements;
    struct instance instances[SLOTS];
    struct ticket tickets[SLOTS];
};
static struct pl_fmt_t fmt = {.name = "float", .type = PL_FMT_FLOAT};
static struct pl_fmt_t unorm = {.name = "unorm", .type = PL_FMT_UNORM};
static pl_fmt formats[] = {&fmt, &unorm};
static const struct pl_gpu_t gpu = {.formats = formats, .num_formats = 2};
static struct pl_mixer_requirements requirements(void)
{
    return (struct pl_mixer_requirements) {
        .backend = PL_MIXER_BACKEND_VULKAN, .domain = PL_MIXER_COLOR_LINEAR,
        .radius = 1, .max_frames = 3,
        .max_in_flight = 3, .format = &fmt,
    };
}
static void retain(void *p) { ((struct module *)p)->refs++; }
static void release(void *p)
{
    struct module *m = p;
    assert(m->refs);
    if (m->refs == 1) assert(m->creates == m->destroys);
    m->refs--;
}
static void retain_owner(void *p)
{
    struct owner *o = p; o->refs++; o->retains++;
}
static void release_owner(void *p)
{
    struct owner *o = p;
    assert(o->refs > 0);
    o->refs--; o->releases++;
}
static enum pl_mixer_status negotiate(void *p, pl_gpu g,
    const struct pl_mixer_config *c, struct pl_mixer_requirements *out)
{
    struct module *m = p;
    *out = m->requirements;
    return m->negotiation_result;
}
static enum pl_mixer_status create(void *p, pl_gpu g,
    const struct pl_mixer_config *c, const struct pl_mixer_requirements *r, void **out)
{
    struct module *m = p;
    if (m->creation_result != PL_MIXER_OK) return m->creation_result;
    for (int i = 0; i < SLOTS; i++) {
        if (m->instances[i].live) continue;
        m->instances[i] = (struct instance) {
            .module = m, .live = true, .revision = c->revision,
            .setting = c->size ? *(const int *)c->data : 0,
        };
        m->creates++;
        *out = &m->instances[i];
        return PL_MIXER_OK;
    }
    assert(false);
    return PL_MIXER_FAILED;
}
static void destroy(void *p)
{
    struct instance *i = p;
    assert(i->live);
    for (int t = 0; t < SLOTS; t++)
        assert(i->module->tickets[t].instance != i);
    i->module->destroys++;
    i->live = false;
}
static enum pl_mixer_status submit(void *p, const struct pl_mixer_job_params *j, void **out)
{
    struct instance *i = p;
    struct module *m = i->module;
    assert(i->live && ((struct owner *)j->owner)->refs > 0);
    m->submits++;
    if (m->submission_result != PL_MIXER_OK && m->submission_result != PL_MIXER_PENDING)
        return m->submission_result;
    for (int t = 0; t < SLOTS; t++) {
        if (m->tickets[t].instance) continue;
        m->tickets[t] = (struct ticket) {
            .instance = i, .params = j, .result = m->submission_result,
            .epoch = j->epoch, .content = j->frames[0].content_id,
        };
        *out = &m->tickets[t];
        return m->submission_result;
    }
    assert(false);
    return PL_MIXER_FAILED;
}
static enum pl_mixer_status poll(void *p, void *t)
{
    struct ticket *ticket = t;
    assert(ticket->instance == p && ticket->instance->live);
    assert(ticket->params->epoch == ticket->epoch);
    assert(ticket->params->frames[0].content_id == ticket->content);
    assert(((struct owner *)ticket->params->owner)->refs > 0);
    ticket->polls++;
    return ticket->result;
}
static void cleanup(void *p, void *t)
{
    struct ticket *ticket = t;
    assert(ticket->instance == p && ticket->result != PL_MIXER_PENDING);
    struct owner *o = ticket->params->owner;
    assert(o->refs > 0); // ticket cleanup MUST precede release of image ownership
    o->ticket_releases++;
    ticket->instance->module->cleanups++;
    *ticket = (struct ticket) {0};
}
static struct pl_mixer_descriptor descriptor(struct module *m, const char *id)
{
    return (struct pl_mixer_descriptor) {
        .abi_version = PL_MIXER_ABI_VERSION, .struct_size = sizeof(struct pl_mixer_descriptor),
        .id = id, .name = id, .priv = m, .retain = retain, .release = release,
        .negotiate = negotiate, .create = create, .destroy = destroy,
        .submit = submit, .poll = poll, .release_job = cleanup,
    };
}
struct input {
    struct owner owner;
    struct pl_tex_t textures[3];
    struct pl_mixer_frame frames[2];
    struct pl_mixer_job_params job;
};
static void input_init(struct input *in, uint64_t epoch)
{
    *in = (struct input) {0};
    for (int i = 0; i < 3; i++)
        in->textures[i].params = (struct pl_tex_params) {.w = 16, .h = 16, .format = &fmt};
    for (int i = 0; i < 2; i++)
        in->frames[i] = (struct pl_mixer_frame) {
            .texture = &in->textures[i], .pts = i, .content_id = epoch * 2 + i,
            .allocation_id = epoch * 3 + i,
        };
    in->job = (struct pl_mixer_job_params) {
        .epoch = epoch, .pts = .5, .frames = in->frames, .num_frames = 2,
        .output = &in->textures[2], .output_allocation_id = epoch * 3 + 2,
        .color = {.transfer = PL_COLOR_TRC_LINEAR}, .alpha = PL_ALPHA_PREMULTIPLIED,
        .owner = &in->owner, .retain = retain_owner, .release = release_owner,
    };
}
static pl_mixer make_instance(pl_mixer_registry r, const char *id, uint64_t revision)
{
    int setting = (int)revision;
    struct pl_mixer_config c = {.revision = revision, .data = &setting, .size = sizeof(setting)};
    pl_mixer m = NULL;
    assert(pl_mixer_create(r, id, &gpu, &c, &m) == PL_MIXER_OK);
    setting = -1; // implementation must have copied borrowed settings
    return m;
}
static void cycles(void)
{
    // Deterministic permutations are reproducible and cover all completion orders.
    static const int orders[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
    for (unsigned cycle = 0; cycle < CYCLES; cycle++) {
        struct module module = {.requirements = requirements(), .submission_result = PL_MIXER_PENDING};
        struct pl_mixer_descriptor d = descriptor(&module, "stress");
        pl_mixer_registry r = pl_mixer_registry_create();
        assert(pl_mixer_register(r, &d) == PL_MIXER_OK);
        d.submit = NULL; // registration must own its table copy
        pl_mixer old = make_instance(r, "stress", cycle * 2);
        struct input in[4]; pl_mixer_job jobs[4] = {0};
        for (int n = 0; n < 3; n++) {
            input_init(&in[n], cycle * 4 + n);
            assert(pl_mixer_submit(old, &in[n].job, &jobs[n]) == PL_MIXER_PENDING);
            in[n].job.epoch = UINT64_MAX;
            in[n].frames[0].content_id = UINT64_MAX; // retained metadata is independent
        }
        input_init(&in[3], cycle * 4 + 3);
        assert(pl_mixer_submit(old, &in[3].job, &jobs[3]) == PL_MIXER_BUSY);
        assert(!jobs[3] && !in[3].owner.retains);
        // New configuration can exist while an older generation still has work.
        pl_mixer next = make_instance(r, "stress", cycle * 2 + 1);
        assert(module.instances[0].revision == cycle * 2);
        assert(module.instances[0].setting == (int)(cycle * 2));
        assert(module.instances[1].setting == (int)(cycle * 2 + 1));
        assert(pl_mixer_submit(next, &in[3].job, &jobs[3]) == PL_MIXER_PENDING);
        assert(pl_mixer_unregister(r, "stress") == PL_MIXER_BUSY);
        assert(pl_mixer_registry_destroy(&r) == PL_MIXER_BUSY);
        for (int n = 0; n < 4; n++) {
            assert(pl_mixer_poll(jobs[n]) == PL_MIXER_PENDING);
            assert(pl_mixer_job_destroy(&jobs[n]) == PL_MIXER_BUSY);
        }
        assert(pl_mixer_destroy(&old) == PL_MIXER_BUSY);
        assert(pl_mixer_destroy(&next) == PL_MIXER_BUSY);
        // Invalid callback status cannot establish completion or release ownership.
        module.tickets[0].result = PL_MIXER_UNSUPPORTED;
        assert(pl_mixer_poll(jobs[0]) == PL_MIXER_INVALID);
        assert(pl_mixer_job_destroy(&jobs[0]) == PL_MIXER_BUSY);
        assert(in[0].owner.refs == 1);
        module.tickets[0].result = PL_MIXER_PENDING;
        // Finish the new generation first. Old jobs must remain untouched.
        module.tickets[3].result = PL_MIXER_OK;
        assert(pl_mixer_poll(jobs[3]) == PL_MIXER_OK);
        assert(pl_mixer_job_destroy(&jobs[3]) == PL_MIXER_OK);
        assert(pl_mixer_destroy(&next) == PL_MIXER_OK);
        for (int n = 0; n < 3; n++) {
            int index = orders[cycle % 6][n];
            enum pl_mixer_status result = (cycle + index) % 2 ? PL_MIXER_FAILED : PL_MIXER_OK;
            module.tickets[index].result = result;
            assert(pl_mixer_poll(jobs[index]) == result);
            unsigned polls = module.tickets[index].polls;
            assert(pl_mixer_poll(jobs[index]) == result);
            assert(module.tickets[index].polls == polls); // terminal results are stable
            assert(pl_mixer_job_destroy(&jobs[index]) == PL_MIXER_OK);
        }
        for (int n = 0; n < 4; n++) {
            assert(in[n].owner.retains == 1 && in[n].owner.releases == 1);
            assert(in[n].owner.refs == 0 && in[n].owner.ticket_releases == 1);
        }
        assert(pl_mixer_destroy(&old) == PL_MIXER_OK);
        assert(pl_mixer_unregister(r, "stress") == PL_MIXER_OK);
        assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK);
        assert(module.refs == 0 && module.creates == module.destroys && module.cleanups == 4);
        assert(allocations == 0);
    }
}
static void failures(void)
{
    struct module module = {.requirements = requirements(), .submission_result = PL_MIXER_PENDING};
    struct pl_mixer_descriptor d = descriptor(&module, "failures");
    fail_next = 1;
    assert(!pl_mixer_registry_create() && !allocations);
    pl_mixer_registry r = pl_mixer_registry_create();
    fail_next = 1;
    assert(pl_mixer_register(r, &d) == PL_MIXER_FAILED && !module.refs);
    assert(pl_mixer_register(r, &d) == PL_MIXER_OK);
    struct pl_mixer_config config = {0}; pl_mixer m = NULL;
    fail_next = 1;
    assert(pl_mixer_create(r, d.id, &gpu, &config, &m) == PL_MIXER_FAILED);
    assert(!m && !module.creates && allocations == 2);
    static const enum pl_mixer_status errors[] = {PL_MIXER_UNSUPPORTED, PL_MIXER_INVALID, PL_MIXER_FAILED};
    for (unsigned i = 0; i < sizeof(errors)/sizeof(errors[0]); i++) {
        module.negotiation_result = errors[i];
        assert(pl_mixer_create(r, d.id, &gpu, &config, &m) == errors[i]);
        assert(!m && allocations == 2 && !module.creates);
        module.negotiation_result = PL_MIXER_OK;
        module.creation_result = errors[i];
        assert(pl_mixer_create(r, d.id, &gpu, &config, &m) == errors[i]);
        assert(!m && allocations == 2 && !module.creates);
        module.creation_result = PL_MIXER_OK;
    }
    m = make_instance(r, d.id, 1);
    struct input in; input_init(&in, 1); pl_mixer_job j = NULL;
    fail_next = 1;
    assert(pl_mixer_submit(m, &in.job, &j) == PL_MIXER_FAILED);
    assert(!j && !in.owner.retains && !module.submits && allocations == 3);
    for (unsigned i = 0; i < sizeof(errors)/sizeof(errors[0]); i++) {
        module.submission_result = errors[i];
        assert(pl_mixer_submit(m, &in.job, &j) == errors[i]);
        assert(!j && !in.owner.refs && !module.cleanups && allocations == 3);
        assert(in.owner.retains == in.owner.releases);
    }
    // Immediate completion is valid; poll must not invoke the plugin again.
    module.submission_result = PL_MIXER_OK;
    assert(pl_mixer_submit(m, &in.job, &j) == PL_MIXER_OK);
    assert(pl_mixer_poll(j) == PL_MIXER_OK && !module.tickets[0].polls);
    assert(pl_mixer_destroy(&m) == PL_MIXER_BUSY); // even complete tickets need release
    assert(pl_mixer_job_destroy(&j) == PL_MIXER_OK);
    assert(!in.owner.refs && in.owner.ticket_releases == 1);
    assert(pl_mixer_destroy(&m) == PL_MIXER_OK);
    assert(pl_mixer_unregister(r, d.id) == PL_MIXER_OK);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK && !allocations);
}
static void invalid_jobs(void)
{
    struct module module = {.requirements = requirements(), .submission_result = PL_MIXER_PENDING};
    struct pl_mixer_descriptor d = descriptor(&module, "validation");
    pl_mixer_registry r = pl_mixer_registry_create();
    assert(pl_mixer_register(r, &d) == PL_MIXER_OK);
    pl_mixer m = make_instance(r, d.id, 1);
    for (unsigned n = 0; n < 16; n++) {
        struct input in; input_init(&in, 1);
        switch (n) {
        case 0: in.job.pts = NAN; break;
        case 1: in.job.pts = INFINITY; break;
        case 2: in.frames[0].pts = NAN; break;
        case 3: in.frames[1].pts = -1; break;
        case 4: in.frames[1].pts = in.frames[0].pts; break;
        case 5: in.job.num_frames = 0; break;
        case 6: in.job.num_frames = 4; break;
        case 7: in.job.frames = NULL; break;
        case 8: in.job.retain = NULL; break;
        case 9: in.job.release = NULL; break;
        case 10: in.job.output = in.frames[1].texture; break;
        case 11: in.textures[0].params.w = 0; break;
        case 12: in.textures[2].params.d = 1; break;
        case 13: in.textures[1].params.format = &unorm; break;
        case 14: in.textures[2].params.export_handle = PL_HANDLE_FD; break;
        case 15: in.job.color.transfer = PL_COLOR_TRC_SRGB; break;
        }
        pl_mixer_job j = NULL;
        assert(pl_mixer_submit(m, &in.job, &j) == PL_MIXER_INVALID);
        assert(!j && !module.submits && !in.owner.retains && allocations == 3);
    }
    assert(pl_mixer_destroy(&m) == PL_MIXER_OK);
    assert(pl_mixer_unregister(r, d.id) == PL_MIXER_OK);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK && !allocations);
}

static void negotiation_and_registry(void)
{
    struct module module = {.requirements = requirements()};
    struct pl_mixer_descriptor d[3];
    const char *ids[] = {"first", "middle", "last"};
    pl_mixer_registry r = pl_mixer_registry_create();
    for (unsigned n = 0; n < 13; n++) {
        struct pl_mixer_descriptor bad = descriptor(&module, "bad");
        enum pl_mixer_status expected = PL_MIXER_INVALID;
        switch (n) {
        case 0: bad.id = ""; break;
        case 1: bad.name = NULL; break;
        case 2: bad.retain = NULL; break;
        case 3: bad.release = NULL; break;
        case 4: bad.negotiate = NULL; break;
        case 5: bad.create = NULL; break;
        case 6: bad.submit = NULL; break;
        case 7: bad.poll = NULL; break;
        case 8: bad.release_job = NULL; break;
        case 9: bad.destroy = NULL; break;
        case 10: bad.struct_size = 0; expected = PL_MIXER_UNSUPPORTED; break;
        case 11: bad.abi_version++; expected = PL_MIXER_UNSUPPORTED; break;
        case 12: bad.id = NULL; break;
        }
        assert(pl_mixer_register(r, &bad) == expected);
        assert(!module.refs && pl_mixer_registry_count(r) == 0 && allocations == 1);
    }
    for (unsigned i = 0; i < 3; i++) {
        d[i] = descriptor(&module, ids[i]);
        assert(pl_mixer_register(r, &d[i]) == PL_MIXER_OK);
    }
    assert(pl_mixer_registry_count(r) == 3 && module.refs == 3);
    for (int i = 0; i < 3; i++)
        assert(!strcmp(pl_mixer_registry_get(r, i)->id, ids[i]));
    assert(!pl_mixer_registry_get(r, -1) && !pl_mixer_registry_get(r, 3));
    assert(pl_mixer_unregister(r, "missing") == PL_MIXER_INVALID);
    assert(pl_mixer_unregister(r, "middle") == PL_MIXER_OK);
    assert(!strcmp(pl_mixer_registry_get(r, 1)->id, "last"));
    struct pl_mixer_config config = {0};
    for (unsigned n = 0; n < 11; n++) {
        module.requirements = requirements();
        struct pl_mixer_requirements *q = &module.requirements;
        switch (n) {
        case 0: q->format = NULL; break;
        case 1: q->format = &unorm; break; // linear needs float storage
        case 2: q->radius = -1; break;
        case 3: q->radius = NAN; break;
        case 4: q->max_frames = 0; break;
        case 5: q->max_in_flight = 0; break;
        case 6: q->domain = 99; break;
        case 7: q->backend = 99; break;
        case 8: q->export_handle = PL_HANDLE_FD | PL_HANDLE_WIN32; break;
        case 9: q->export_handle = PL_HANDLE_FD; break; // device lacks export support
        case 10: q->backend = PL_MIXER_BACKEND_CUDA; break; // no export request
        }
        pl_mixer m = NULL;
        assert(pl_mixer_create(r, "first", &gpu, &config, &m) == PL_MIXER_UNSUPPORTED);
        assert(!m && !module.creates && allocations == 3);
    }
    module.requirements = requirements();
    // Negotiation/create are synchronous. They must not publish pending instances.
    pl_mixer m = NULL;
    module.negotiation_result = PL_MIXER_PENDING;
    assert(pl_mixer_create(r, "first", &gpu, &config, &m) == PL_MIXER_INVALID && !m);
    module.negotiation_result = PL_MIXER_OK;
    module.creation_result = PL_MIXER_PENDING;
    assert(pl_mixer_create(r, "first", &gpu, &config, &m) == PL_MIXER_INVALID && !m);
    module.creation_result = PL_MIXER_OK;
    m = make_instance(r, "last", 5);
    // Removing another registration must not move or invalidate the live entry.
    assert(pl_mixer_unregister(r, "first") == PL_MIXER_OK);
    assert(pl_mixer_registry_count(r) == 1);
    assert(pl_mixer_get_requirements(m)->format == &fmt);
    assert(pl_mixer_destroy(&m) == PL_MIXER_OK);
    assert(pl_mixer_unregister(r, "last") == PL_MIXER_OK);
    assert(!module.refs && allocations == 1);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK && !allocations);
}


static void exact_sample_and_completed_color(void)
{
    struct module module = {.requirements = requirements(), .submission_result = PL_MIXER_OK};
    module.requirements.domain = PL_MIXER_COLOR_COMPLETED;
    module.requirements.format = &unorm;
    struct pl_mixer_descriptor d = descriptor(&module, "completed");
    pl_mixer_registry r = pl_mixer_registry_create();
    assert(pl_mixer_register(r, &d) == PL_MIXER_OK);
    pl_mixer m = make_instance(r, d.id, 1);
    // Changing a later negotiation result does not mutate this instance's contract.
    module.requirements = requirements();
    assert(pl_mixer_get_requirements(m)->domain == PL_MIXER_COLOR_COMPLETED);
    struct input in; input_init(&in, 1);
    for (int i = 0; i < 3; i++) in.textures[i].params.format = &unorm;
    in.job.color.transfer = PL_COLOR_TRC_SRGB;
    struct pl_mixer_frame frames[3] = {in.frames[0], in.frames[0], in.frames[1]};
    frames[1].pts = .5;
    in.job.frames = frames;
    in.job.num_frames = 3;
    // Exact sample is additional to the required one past and one future frame.
    pl_mixer_job j = NULL;
    for (unsigned i = 0; i < 16; i++) {
        assert(pl_mixer_submit(m, &in.job, &j) == PL_MIXER_OK);
        assert(pl_mixer_job_destroy(&j) == PL_MIXER_OK && !j);
    }
    assert(in.owner.retains == 16 && in.owner.releases == 16 && !in.owner.refs);
    assert(module.cleanups == 16);
    assert(pl_mixer_destroy(&m) == PL_MIXER_OK);
    assert(pl_mixer_unregister(r, d.id) == PL_MIXER_OK);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK && !allocations);
}


static void shared_frame_lease(void)
{
    struct module module = {.requirements = requirements(), .submission_result = PL_MIXER_PENDING};
    struct pl_mixer_descriptor d = descriptor(&module, "shared");
    pl_mixer_registry r = pl_mixer_registry_create();
    assert(pl_mixer_register(r, &d) == PL_MIXER_OK);
    pl_mixer m = make_instance(r, d.id, 1);
    struct input in[3]; pl_mixer_job jobs[3] = {0};
    for (unsigned n = 0; n < 3; n++) {
        input_init(&in[n], 1);
        in[n].job.frames = in[0].frames; // three outputs reuse one source pair
        in[n].job.owner = &in[0].owner; // one owner retains all fixture textures
        in[n].job.pts = .25 * (n + 1);
        assert(pl_mixer_submit(m, &in[n].job, &jobs[n]) == PL_MIXER_PENDING);
    }
    assert(in[0].owner.refs == 3);
    const unsigned order[] = {2, 0, 1};
    for (unsigned n = 0; n < 3; n++) {
        unsigned index = order[n];
        module.tickets[index].result = PL_MIXER_OK;
        assert(pl_mixer_poll(jobs[index]) == PL_MIXER_OK);
        assert(in[0].owner.refs == 3 - n); // completion alone must not free ownership
        assert(pl_mixer_job_destroy(&jobs[index]) == PL_MIXER_OK);
        assert(in[0].owner.refs == 2 - (int)n);
    }
    assert(in[0].owner.retains == 3 && in[0].owner.releases == 3);
    assert(pl_mixer_destroy(&m) == PL_MIXER_OK);
    assert(pl_mixer_unregister(r, d.id) == PL_MIXER_OK);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK && !allocations);
}

int main(void)
{
    failures(); invalid_jobs(); negotiation_and_registry(); exact_sample_and_completed_color(); shared_frame_lease(); cycles();
    printf("%d lifecycle cycles, %d admitted jobs, six completion orders; fault and validation cases passed\n",
           CYCLES, CYCLES * 4);
    return 0;
}
