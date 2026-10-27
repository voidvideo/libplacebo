#include "utils.h"
#include <assert.h>
#include <libplacebo/frame_mixer.h>

struct state { int modules, instances, leases, releases, polls; bool fail, cuda; };
static struct pl_fmt_t format = {.name = "test", .type = PL_FMT_FLOAT};
static pl_fmt formats[] = {&format};
static void retain(void *p) { ((struct state *) p)->modules++; }
static void release(void *p) { ((struct state *) p)->modules--; }
static void lease(void *p) { ((struct state *) p)->leases++; }
static void unlease(void *p) { ((struct state *) p)->leases--; }
static enum pl_mixer_status negotiate(void *p, pl_gpu gpu,
    const struct pl_mixer_config *c, struct pl_mixer_requirements *r)
{
    *r = (struct pl_mixer_requirements) {
        .backend = PL_MIXER_BACKEND_VULKAN, .domain = PL_MIXER_COLOR_LINEAR,
        .past_frames = 1, .future_frames = 1, .min_frames = 2, .max_frames = 2,
        .max_in_flight = 1, .format = &format,
    };
    if (((struct state *)p)->cuda) {
        r->backend = PL_MIXER_BACKEND_CUDA;
        r->export_handle = PL_HANDLE_FD;
    }
    return PL_MIXER_OK;
}
static enum pl_mixer_status create(void *p, pl_gpu gpu,
    const struct pl_mixer_config *c, const struct pl_mixer_requirements *r, void **out)
{
    ((struct state *)p)->instances++; *out = p; return PL_MIXER_OK;
}
static void destroy(void *p) { ((struct state *)p)->instances--; }
static enum pl_mixer_status submit(void *p, const struct pl_mixer_job_params *j, void **out)
{
    // Retain the core-owned metadata itself to verify lifetime after submission.
    *out = (void *) j;
    return PL_MIXER_PENDING;
}
static enum pl_mixer_status poll(void *p, void *t)
{
    struct state *s = p;
    const struct pl_mixer_job_params *j = t;
    assert(j->frames[0].content_id == 100);
    if (!s->polls++) return PL_MIXER_PENDING;
    return s->fail ? PL_MIXER_FAILED : PL_MIXER_OK;
}
static void release_job(void *p, void *t) { ((struct state *)p)->releases++; }
int main(void)
{
    struct state state = {0};
    struct pl_mixer_descriptor d = {
        .abi_version = PL_MIXER_ABI_VERSION, .struct_size = sizeof(d),
        .id = "test.external", .name = "Test", .priv = &state,
        .retain = retain, .release = release, .negotiate = negotiate,
        .create = create, .destroy = destroy, .submit = submit,
        .poll = poll, .release_job = release_job,
    };
    pl_mixer_registry r = pl_mixer_registry_create();
    assert(r);
    d.abi_version++;
    assert(pl_mixer_register(r, &d) == PL_MIXER_UNSUPPORTED);
    d.abi_version--;
    assert(pl_mixer_register(r, &d) == PL_MIXER_OK);
    assert(pl_mixer_register(r, &d) == PL_MIXER_INVALID);
    assert(state.modules == 1 && pl_mixer_registry_count(r) == 1);
    assert(pl_mixer_registry_get(r, 0)->create == create);
    assert(!pl_mixer_registry_get(r, 1));
    const struct pl_gpu_t gpu = {.formats = formats, .num_formats = 1};
    const struct pl_mixer_config config = {.revision = 1};
    pl_mixer m = NULL;
    state.cuda = true;
    assert(pl_mixer_create(r, d.id, &gpu, &config, &m) == PL_MIXER_UNSUPPORTED);
    assert(!m && !state.instances); // reject before creating a CUDA instance
    state.cuda = false;
    assert(pl_mixer_create(r, d.id, &gpu, &config, &m) == PL_MIXER_OK);
    assert(pl_mixer_unregister(r, d.id) == PL_MIXER_BUSY);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_BUSY);
    assert(pl_mixer_get_requirements(m)->max_in_flight == 1);
    struct pl_tex_t tex[3] = {0};
    for (int i = 0; i < 3; i++)
        tex[i].params = (struct pl_tex_params) {.w = 16, .h = 16, .format = &format};
    struct pl_mixer_frame frames[2] = {
        {.texture = &tex[0], .content_id = 100, .pts = 0},
        {.texture = &tex[1], .content_id = 101, .pts = 1},
    };
    struct pl_mixer_job_params job = {
        .pts = .5, .frames = frames, .num_frames = 2, .output = &tex[2],
        .color = {.transfer = PL_COLOR_TRC_LINEAR}, .owner = &state,
        .retain = lease, .release = unlease,
    };
    pl_mixer_job j = NULL, second = NULL;
    job.pts = 2; // insufficient future support must not be accepted
    assert(pl_mixer_submit(m, &job, &j) == PL_MIXER_INVALID);
    job.pts = .5;
    job.color.transfer = PL_COLOR_TRC_SRGB;
    assert(pl_mixer_submit(m, &job, &j) == PL_MIXER_INVALID);
    job.color.transfer = PL_COLOR_TRC_LINEAR;
    job.output = frames[0].texture;
    assert(pl_mixer_submit(m, &job, &j) == PL_MIXER_INVALID);
    assert(!state.leases);
    job.output = &tex[2];
    for (int fail = 0; fail < 2; fail++) {
        state.fail = fail; state.polls = 0;
        frames[0].content_id = 100;
        assert(pl_mixer_submit(m, &job, &j) == PL_MIXER_PENDING);
        frames[0].content_id = 999; // core must have copied input descriptors
        assert(state.leases == 1);
        assert(pl_mixer_submit(m, &job, &second) == PL_MIXER_BUSY && !second);
        assert(pl_mixer_destroy(&m) == PL_MIXER_BUSY);
        assert(pl_mixer_job_destroy(&j) == PL_MIXER_BUSY && j);
        assert(pl_mixer_poll(j) == PL_MIXER_PENDING);
        enum pl_mixer_status expected = fail ? PL_MIXER_FAILED : PL_MIXER_OK;
        assert(pl_mixer_poll(j) == expected);
        assert(pl_mixer_poll(j) == expected && state.polls == 2);
        assert(pl_mixer_job_destroy(&j) == PL_MIXER_OK && !j);
        assert(!state.leases);
    }
    assert(state.releases == 2);
    assert(pl_mixer_destroy(&m) == PL_MIXER_OK && !m);
    assert(!state.instances);
    assert(pl_mixer_unregister(r, d.id) == PL_MIXER_OK);
    assert(!state.modules);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK && !r);
    return 0;
}
