#include "gpu.h"
#include <libplacebo/dummy.h>
#include "pass_prepare_helpers.h"

static _Thread_local bool caller_thread;
static _Thread_local bool forbid_gpu_destroy;
static struct prepare_latch blocked;
static atomic_int created, destroyed, executed;
static bool fail_create, fail_run;
static enum pl_pass_prepare_phase failure_phase;
static bool inspect_snapshot;

static void check_snapshot(const struct pl_pass_params *p)
{
    REQUIRE(!strcmp(p->glsl_shader, "fragment"));
    REQUIRE(!strcmp(p->vertex_shader, "vertex"));
    REQUIRE(!strcmp(p->variables[0].name, "variable"));
    REQUIRE(!strcmp(p->descriptors[0].name, "descriptor"));
    REQUIRE(!strcmp(p->vertex_attribs[0].name, "position"));
    REQUIRE(p->constants[0].id == 9);
    REQUIRE(p->constants[0].offset == 4);
    REQUIRE(((uint32_t *) p->constant_data)[1] == 23);
    REQUIRE(p->blend_params->src_rgb == PL_BLEND_SRC_ALPHA);
}

static pl_pass mock_create(pl_gpu gpu, const struct pl_pass_params *params,
                           enum pl_pass_prepare_phase *phase)
{
    REQUIRE(!caller_thread);
    latch_block(&blocked);
    atomic_fetch_add(&created, 1);
    if (inspect_snapshot)
        check_snapshot(params);
    if (fail_create) {
        *phase = failure_phase;
        return NULL;
    }
    struct pl_pass_t *pass = pl_zalloc_ptr(NULL, pass);
    pass->params = pl_pass_params_copy(pass, params);
    return pass;
}

static void mock_destroy(pl_gpu gpu, pl_pass pass)
{
    REQUIRE(!forbid_gpu_destroy);
    atomic_fetch_add(&destroyed, 1);
    pl_free((void *) pass);
}

static bool mock_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(caller_thread);
    REQUIRE(params->pass);
    atomic_fetch_add(&executed, 1);
    return !fail_run;
}

static pl_pass_preparation submit(pl_gpu gpu, const struct pl_pass_params *params)
{
    pl_pass_preparation request = NULL;
    REQUIRE(pl_pass_prepare_submit(gpu, params, &request) == PL_PASS_PREPARE_ACCEPTED);
    REQUIRE(request);
    return request;
}

static void test_snapshot(pl_gpu gpu)
{
    void *tmp = pl_tmp(NULL);
    struct pl_pass_params *p = pl_zalloc_ptr(tmp, p);
    *p = (struct pl_pass_params) {
        .type = PL_PASS_RASTER,
        .glsl_shader = pl_str0dup0(tmp, "fragment"),
        .vertex_shader = pl_str0dup0(tmp, "vertex"),
        .num_variables = 1,
        .num_descriptors = 1,
        .num_vertex_attribs = 1,
        .num_constants = 1,
        .vertex_stride = 4,
        .target_format = pl_find_named_fmt(gpu, "rgba8"),
        .load_target = true,
    };
    p->variables = pl_alloc_ptr(tmp, p->variables);
    p->variables[0] = pl_var_float(pl_str0dup0(tmp, "variable"));
    p->descriptors = pl_alloc_ptr(tmp, p->descriptors);
    p->descriptors[0] = (struct pl_desc) {
        .name = pl_str0dup0(tmp, "descriptor"), .type = PL_DESC_BUF_STORAGE,
    };
    p->vertex_attribs = pl_alloc_ptr(tmp, p->vertex_attribs);
    p->vertex_attribs[0] = (struct pl_vertex_attrib) {
        .name = pl_str0dup0(tmp, "position"), .fmt = pl_find_named_fmt(gpu, "r32f"),
    };
    p->constants = pl_alloc_ptr(tmp, p->constants);
    p->constants[0] = (struct pl_constant) { .type = PL_VAR_UINT, .id = 9, .offset = 4 };
    p->constant_data = pl_memdup(tmp, (uint32_t[]) {0, 23}, 8);
    p->blend_params = pl_memdup_ptr(tmp, &pl_alpha_overlay);
    inspect_snapshot = true;
    latch_arm(&blocked);
    pl_pass_preparation request = submit(gpu, p);
    latch_entered(&blocked);
    memset((char *) p->glsl_shader, 'x', 8);
    memset(p->constant_data, 0xff, 8);
    pl_free(tmp);
    REQUIRE(pl_pass_prepare_poll(request) == PL_PASS_PREPARE_PENDING);
    pl_prepared_pass result = NULL;
    REQUIRE(pl_pass_prepare_take(&request, &result) == PL_PASS_PREPARE_PENDING);
    REQUIRE(request && !result);
    latch_release(&blocked);
    result = take_prepared(&request);
    check_snapshot(pl_prepared_pass_params(result));
    inspect_snapshot = false;
    pl_prepared_pass_destroy(&result);
    REQUIRE(!result);
}

static void wait_destroyed(int n)
{
    pl_clock_t start = pl_clock_now();
    while (atomic_load(&destroyed) != n) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
}

static void test_progress_and_capacity(pl_gpu gpu, const struct pl_pass_params *p)
{
    pl_pass_preparation request = submit(gpu, p);
    pl_prepared_pass active = take_prepared(&request);
    latch_arm(&blocked);
    request = submit(gpu, p);
    latch_entered(&blocked);
    const int before = atomic_load(&executed);
    REQUIRE(pl_prepared_pass_run(active, pl_pass_run_params(
        .compute_groups = {1, 1, 1})) == PL_PREPARED_PASS_RUN_OK);
    REQUIRE(atomic_load(&executed) == before + 1);
    REQUIRE(pl_pass_prepare_poll(request) == PL_PASS_PREPARE_PENDING);

    // Fill admission while the worker cannot drain it; rejected submission
    // must return promptly with no handle and must never compile inline.
    pl_pass_preparation queued[256] = {0};
    int count = 0;
    for (; count < PL_ARRAY_SIZE(queued); count++) {
        enum pl_pass_prepare_result r = pl_pass_prepare_submit(gpu, p, &queued[count]);
        if (r == PL_PASS_PREPARE_CAPACITY) {
            REQUIRE(!queued[count]);
            break;
        }
        REQUIRE(r == PL_PASS_PREPARE_ACCEPTED);
    }
    REQUIRE(count > 0 && count < PL_ARRAY_SIZE(queued));
    for (int i = 0; i < count; i++) {
        pl_pass_prepare_cancel(queued[i]);
        REQUIRE(pl_pass_prepare_poll(queued[i]) == PL_PASS_PREPARE_CANCELLED);
        pl_pass_prepare_release(&queued[i]);
        REQUIRE(!queued[i]);
    }
    const int expect_destroyed = atomic_load(&destroyed) + 1;
    pl_pass_prepare_cancel(request);
    REQUIRE(pl_pass_prepare_poll(request) == PL_PASS_PREPARE_CANCELLED);
    pl_pass_prepare_release(&request);
    REQUIRE(!request);
    latch_release(&blocked);
    wait_destroyed(expect_destroyed);
    pl_prepared_pass_destroy(&active);
}

static void test_failure_and_variant(pl_gpu gpu, const struct pl_pass_params *p)
{
    for (failure_phase = PL_PASS_PREPARE_PHASE_RESOURCES;
         failure_phase <= PL_PASS_PREPARE_PHASE_PIPELINE; failure_phase++) {
        fail_create = true;
        pl_pass_preparation request = submit(gpu, p);
        REQUIRE(await_prepared(request) == PL_PASS_PREPARE_FAILED);
        REQUIRE(pl_pass_prepare_failure_phase(request) == failure_phase);
        REQUIRE(pl_pass_prepare_error(request) && *pl_pass_prepare_error(request));
        pl_prepared_pass pass = NULL;
        REQUIRE(pl_pass_prepare_take(&request, &pass) == PL_PASS_PREPARE_FAILED);
        REQUIRE(request && !pass);
        pl_pass_prepare_release(&request);
    }
    fail_create = false;
    uint32_t value = 23;
    struct pl_constant spec = { .type = PL_VAR_UINT, .id = 0 };
    struct pl_pass_params specialized = *p;
    specialized.num_constants = 1;
    specialized.constants = &spec;
    specialized.constant_data = &value;
    pl_pass_preparation request = submit(gpu, &specialized);
    pl_prepared_pass pass = take_prepared(&request);
    int before = atomic_load(&executed);
    value = 24;
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .constant_data = &value, .compute_groups = {1, 1, 1})) ==
        PL_PREPARED_PASS_RUN_VARIANT_MISMATCH);
    REQUIRE(atomic_load(&executed) == before);
    value = 23;
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .constant_data = &value, .compute_groups = {1, 1, 1})) == PL_PREPARED_PASS_RUN_OK);

    // Backend execution failure must be reported, never converted to success.
    before = atomic_load(&executed);
    fail_run = true;
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .compute_groups = {1, 1, 1})) == PL_PREPARED_PASS_RUN_FAILED);
    REQUIRE(atomic_load(&executed) == before + 1);
    fail_run = false;

    // Both public argument checks and checked execution validation reject
    // invalid inputs before the backend receives any work.
    before = atomic_load(&executed);
    REQUIRE(pl_prepared_pass_run(pass, NULL) == PL_PREPARED_PASS_RUN_INVALID);
    struct pl_pass_t raw = {0};
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .pass = &raw, .compute_groups = {1, 1, 1})) == PL_PREPARED_PASS_RUN_INVALID);
    REQUIRE(pl_prepared_pass_run(pass, pl_pass_run_params(
        .compute_groups = {-1, 1, 1})) == PL_PREPARED_PASS_RUN_INVALID);
    REQUIRE(atomic_load(&executed) == before);
    pl_prepared_pass_destroy(&pass);

    // Cancellation also wins after readiness, before take.
    request = submit(gpu, p);
    REQUIRE(await_prepared(request) == PL_PASS_PREPARE_READY);
    before = atomic_load(&destroyed);
    forbid_gpu_destroy = true;
    pl_pass_prepare_cancel(request);
    REQUIRE(pl_pass_prepare_take(&request, &pass) == PL_PASS_PREPARE_CANCELLED);
    REQUIRE(request && !pass);
    pl_pass_prepare_release(&request);
    forbid_gpu_destroy = false;
    wait_destroyed(before + 1);

    // Repeatedly race completion against cancellation/release. Every accepted
    // create is retired exactly once and no result can be taken after cancel.
    for (int i = 0; i < 16; i++) {
        latch_arm(&blocked);
        request = submit(gpu, p);
        latch_entered(&blocked);
        before = atomic_load(&destroyed);
        latch_release(&blocked);
        forbid_gpu_destroy = true;
        pl_pass_prepare_cancel(request);
        REQUIRE(pl_pass_prepare_poll(request) == PL_PASS_PREPARE_CANCELLED);
        pl_pass_prepare_release(&request);
        forbid_gpu_destroy = false;
        wait_destroyed(before + 1);
    }
}

static atomic_bool shutdown_started, shutdown_finished;
static PL_THREAD_VOID shutdown_worker(void *arg)
{
    atomic_store(&shutdown_started, true);
    pl_pass_prepare_uninit(arg);
    atomic_store(&shutdown_finished, true);
    PL_THREAD_RETURN();
}

static void test_shutdown(pl_gpu gpu, const struct pl_pass_params *p)
{
    latch_arm(&blocked);
    pl_pass_preparation request = submit(gpu, p);
    latch_entered(&blocked);
    pl_pass_prepare_release(&request);
    pl_thread shutdown;
    REQUIRE(!pl_thread_create(&shutdown, shutdown_worker, (void *) gpu));
    while (!atomic_load(&shutdown_started))
        pl_thread_sleep(0.001);
    REQUIRE(!atomic_load(&shutdown_finished));
    latch_release(&blocked);
    REQUIRE(!pl_thread_join(shutdown));
    REQUIRE(atomic_load(&shutdown_finished));
}

int main(void)
{
    caller_thread = true;
    latch_init(&blocked);
    pl_log log = pl_test_logger();
    pl_gpu gpu = pl_gpu_dummy_create(log, NULL);
    struct pl_pass_params p = { .type = PL_PASS_COMPUTE, .glsl_shader = "compute" };
    pl_pass_preparation request = NULL;
    REQUIRE(!pl_pass_prepare_supported(gpu));
    REQUIRE(pl_pass_prepare_submit(gpu, &p, &request) == PL_PASS_PREPARE_UNSUPPORTED);
    REQUIRE(!request);
    struct pl_gpu_fns *fns = PL_PRIV(gpu);
    fns->pass_create_prepared = mock_create;
    fns->pass_run_prepared = mock_run;
    fns->pass_destroy = mock_destroy;
    pl_pass_prepare_init(gpu);
    REQUIRE(pl_pass_prepare_supported(gpu));
    struct pl_pass_params invalid = p;
    invalid.glsl_shader = NULL;
    REQUIRE(pl_pass_prepare_submit(gpu, &invalid, &request) == PL_PASS_PREPARE_INVALID);
    REQUIRE(!request);
    test_snapshot(gpu);
    test_progress_and_capacity(gpu, &p);
    test_failure_and_variant(gpu, &p);
    test_shutdown(gpu, &p);
    int before = atomic_load(&created);
    REQUIRE(pl_pass_prepare_submit(gpu, &p, &request) == PL_PASS_PREPARE_UNAVAILABLE);
    REQUIRE(!request && atomic_load(&created) == before);
    REQUIRE(atomic_load(&destroyed) == atomic_load(&created) - 3);
    pl_gpu_dummy_destroy(&gpu);
    latch_destroy(&blocked);
    pl_log_destroy(&log);
}
