#include "vulkan/gpu.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/custom.h>
#include "utils.h"
#include "pl_thread.h"

static _Thread_local bool describing, forbid_compile;
static struct pl_gpu_fns original;
static int gpu_runs, writes;

static pl_tex observe_texture(pl_gpu gpu, const struct pl_tex_params *params)
{
    REQUIRE(!describing);
    return original.tex_create(gpu, params);
}

static pl_buf observe_buffer(pl_gpu gpu, const struct pl_buf_params *params)
{
    REQUIRE(!describing);
    return original.buf_create(gpu, params);
}

static pl_pass observe_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    REQUIRE(!describing && !forbid_compile);
    return original.pass_create(gpu, params);
}

static void observe_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(!describing);
    gpu_runs++;
    original.pass_run(gpu, params);
}

static bool observe_prepared_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(!describing);
    gpu_runs++;
    return original.pass_run_prepared(gpu, params);
}

static void observe_write(pl_gpu gpu, pl_buf buffer, size_t offset,
                          const void *data, size_t size)
{
    REQUIRE(!describing);
    writes++;
    original.buf_write(gpu, buffer, offset, data, size);
}

struct snapshot_owner {
    int refs;
    struct pl_render_params params;
    struct pl_hook hook;
    const struct pl_hook *hooks[1];
    pl_dispatch auxiliary;
    pl_buf output;
    int ordinary, resets, descriptions, executions;
    int analysis_executions, output_executions;
};

static void retain(void *priv)
{
    ((struct snapshot_owner *) priv)->refs++;
}

static void release(void *priv)
{
    struct snapshot_owner *owner = priv;
    REQUIRE(owner->refs > 1);
    owner->refs--;
}

static void ordinary_reset(void *priv)
{
    ((struct snapshot_owner *) priv)->resets++;
}

static struct pl_hook_res ordinary_hook(void *priv, const struct pl_hook_params *params)
{
    ((struct snapshot_owner *) priv)->ordinary++;
    return (struct pl_hook_res) {0};
}

static struct pl_hook_prepare_result auxiliary_hook(struct snapshot_owner *owner,
    const struct pl_hook_prepare_params *params, bool execute)
{
    if (execute) {
        owner->executions++;
        if (params->stage == PL_HOOK_RGB)
            owner->analysis_executions++;
        if (params->stage == PL_HOOK_OUTPUT)
            owner->output_executions++;
    } else
        owner->descriptions++;
    pl_shader shader = params->begin(params->context, owner->auxiliary, false);
    REQUIRE(shader);
    struct pl_buffer_var value = { .var = pl_var_uint("value") };
    value.layout = pl_std430_layout(0, &value.var);
    struct pl_shader_desc descriptor = {
        .desc = { .name = "analysis", .type = PL_DESC_BUF_STORAGE,
                  .access = PL_DESC_ACCESS_WRITEONLY },
        .binding.object = execute ? owner->output : NULL,
        .buffer = &owner->output->params,
        .buffer_vars = &value, .num_buffer_vars = 1,
    };
    REQUIRE(pl_shader_custom(shader, &(struct pl_custom_shader) {
        .body = "value = 42u;", .compute = true, .compute_group_size = {1, 1},
        .descriptors = &descriptor, .num_descriptors = 1,
    }));
    enum pl_dispatch_result result = params->compute(params->context, owner->auxiliary,
        pl_dispatch_compute_params(.shader = &shader, .dispatch_size = {1, 1, 1}));
    REQUIRE(!shader);
    return (struct pl_hook_prepare_result) { .status = result };
}

static struct pl_hook_prepare_result describe_hook(void *priv,
    const struct pl_hook_prepare_params *params)
{
    return auxiliary_hook(priv, params, false);
}

static struct pl_hook_prepare_result execute_hook(void *priv,
    const struct pl_hook_prepare_params *params)
{
    return auxiliary_hook(priv, params, true);
}

static struct snapshot_owner *new_snapshot(pl_gpu gpu, bool hook)
{
    struct snapshot_owner *owner = calloc(1, sizeof(*owner));
    REQUIRE(owner);
    owner->refs = 1;
    owner->params = pl_render_default_params;
    owner->params.peak_detect_params = NULL;
    owner->params.frame_mixer = NULL;
    if (hook) {
        owner->auxiliary = pl_dispatch_create(gpu->log, gpu);
        owner->output = pl_buf_create(gpu, pl_buf_params(
            .size = sizeof(uint32_t), .storable = true,
            .host_writable = true, .host_readable = true,
        ));
        REQUIRE(owner->auxiliary && owner->output);
        uint32_t sentinel = 7;
        pl_buf_write(gpu, owner->output, 0, &sentinel, sizeof(sentinel));
        owner->hook = (struct pl_hook) {
            .stages = PL_HOOK_RGB, .input = PL_HOOK_SIG_NONE,
            .priv = owner, .reset = ordinary_reset, .hook = ordinary_hook,
            .signature = 1, .describe = describe_hook,
            .execute_prepared = execute_hook,
        };
        owner->hooks[0] = &owner->hook;
        owner->params.hooks = owner->hooks;
        owner->params.num_hooks = 1;
    }
    return owner;
}

static struct pl_frame frame(pl_tex texture)
{
    struct pl_color_space color = pl_color_space_srgb;
    pl_color_space_infer(&color);
    return (struct pl_frame) {
        .num_planes = 1,
        .planes = {{ .texture = texture, .components = 4,
                     .component_mapping = {0, 1, 2, 3} }},
        .repr = { .sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_FULL,
                  .alpha = PL_ALPHA_INDEPENDENT },
        .color = color,
    };
}

static void test_renderer(pl_gpu gpu, struct pl_frame image, int width, bool hook, bool ewa)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer reference_renderer = pl_renderer_create(gpu->log, gpu);
    REQUIRE(renderer && reference_renderer);
    struct pl_tex_params output_params = {
        .w = width, .h = width, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex output = pl_tex_create(gpu, &output_params);
    pl_tex reference = pl_tex_create(gpu, &output_params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output);
    struct pl_frame reference_target = frame(reference);
    struct snapshot_owner *owner = new_snapshot(gpu, hook);
    if (ewa)
        owner->params.upscaler = &pl_filter_ewa_lanczos;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    pl_renderer_preparation preparation = NULL;
    // Both renderers must generate the same blue-noise LUT for exact pixels.
    srand(1234);
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot,
                                       &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(preparation && owner->refs == 2);
    REQUIRE(!owner->ordinary && !owner->resets && !owner->executions);
    if (hook)
        REQUIRE(owner->descriptions > 0);
    forbid_compile = true;
    REQUIRE(pl_renderer_prepare_submit(preparation) == PL_RENDERER_PREPARE_OK);
    pl_clock_t start = pl_clock_now();
    enum pl_pass_prepare_state state;
    while ((state = pl_renderer_prepare_poll(preparation)) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    if (state != PL_PASS_PREPARE_READY)
        fprintf(stderr, "renderer prepare: %s\n", pl_renderer_prepare_error(preparation));
    REQUIRE(state == PL_PASS_PREPARE_READY);
    describing = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    if (image.num_planes > 1) {
        struct pl_frame changed_plane = image;
        changed_plane.planes[image.num_planes - 1].shift_x += 0.25;
        int descriptions = owner->descriptions;
        REQUIRE(pl_renderer_preflight_image(preparation, &changed_plane, &target) ==
                PL_RENDERER_PREPARE_NOT_READY);
        REQUIRE(owner->descriptions == descriptions);
    }
    describing = false;
    REQUIRE(!owner->executions && !owner->ordinary && !owner->resets);
    REQUIRE(pl_render_image_prepared(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    forbid_compile = false;
    REQUIRE(!owner->ordinary && !owner->resets);
    if (hook) {
        uint32_t value = 0;
        REQUIRE(owner->executions > 0);
        REQUIRE(pl_buf_read(gpu, owner->output, 0, &value, sizeof(value)));
        REQUIRE(value == 42);
    }
    struct pl_render_params reference_params = owner->params;
    reference_params.hooks = NULL;
    reference_params.num_hooks = 0;
    srand(1234);
    REQUIRE(pl_render_image(reference_renderer, &image, &reference_target, &reference_params));
    uint8_t *expected = malloc(width * width * 4), *actual = malloc(width * width * 4);
    REQUIRE(expected && actual);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    REQUIRE_MEMEQ(expected, actual, width * width * 4);

    // An output shader mismatch occurs after the analysis hook is described.
    // Whole-frame preflight must catch it before that hook executes any work.
    if (hook) {
        struct pl_frame changed_repr = target;
        changed_repr.repr.sys = PL_COLOR_SYSTEM_BT_2020_C;
        changed_repr.repr.levels = PL_COLOR_LEVELS_LIMITED;
        const int runs = gpu_runs, buffer_writes = writes;
        const int descriptions = owner->descriptions, executions = owner->executions;
        forbid_compile = true;
        enum pl_renderer_prepare_result changed_result =
            pl_render_image_prepared(preparation, &image, &changed_repr);
        if (changed_result != PL_RENDERER_PREPARE_NOT_READY)
            fprintf(stderr, "changed representation result %d: %s\n", changed_result,
                    pl_renderer_prepare_error(preparation));
        REQUIRE(changed_result == PL_RENDERER_PREPARE_NOT_READY);
        forbid_compile = false;
        REQUIRE(owner->descriptions > descriptions);
        REQUIRE(owner->executions == executions);
        REQUIRE(gpu_runs == runs && writes == buffer_writes);
        REQUIRE(pl_renderer_prepare_poll(preparation) == PL_PASS_PREPARE_READY);
    }

    // Changed target metadata must be rejected before an earlier auxiliary
    // analysis pass or any renderer GPU work can run.
    output_params.w++;
    pl_tex changed_output = pl_tex_create(gpu, &output_params);
    REQUIRE(changed_output);
    struct pl_frame changed_target = frame(changed_output);
    const int before_runs = gpu_runs, before_writes = writes;
    const int before_execute = owner->executions;
    describing = forbid_compile = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &changed_target) ==
            PL_RENDERER_PREPARE_NOT_READY);
    describing = false;
    REQUIRE(pl_render_image_prepared(preparation, &image, &changed_target) ==
            PL_RENDERER_PREPARE_NOT_READY);
    forbid_compile = false;
    REQUIRE(gpu_runs == before_runs && writes == before_writes);
    REQUIRE(owner->executions == before_execute && !owner->ordinary && !owner->resets);
    struct pl_render_errors errors = pl_renderer_get_errors(renderer);
    REQUIRE(!errors.errors && !errors.num_disabled_hooks);
    REQUIRE(pl_renderer_prepare_poll(preparation) == PL_PASS_PREPARE_READY);
    describing = forbid_compile = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(pl_render_image_prepared(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    forbid_compile = false;
    REQUIRE(!owner->ordinary && !owner->resets);

    pl_renderer_prepare_destroy(&preparation);
    REQUIRE(!preparation && owner->refs == 1);
    pl_dispatch_destroy(&owner->auxiliary);
    pl_buf_destroy(gpu, &owner->output);
    free(owner);
    free(expected);
    free(actual);
    pl_tex_destroy(gpu, &changed_output);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
    pl_renderer_destroy(&reference_renderer);
    pl_renderer_destroy(&renderer);
}

static void test_mix(pl_gpu gpu, struct pl_frame image, bool direct, bool hook)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer reference_renderer = pl_renderer_create(gpu->log, gpu);
    struct pl_tex_params output_params = {
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex output = pl_tex_create(gpu, &output_params);
    pl_tex reference = pl_tex_create(gpu, &output_params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output), reference_target = frame(reference);
    struct snapshot_owner *owner = new_snapshot(gpu, hook);
    owner->params.skip_caching_single_frame = direct;
    if (hook)
        owner->hook.stages |= PL_HOOK_OUTPUT;
    struct pl_dither_params dither = pl_dither_default_params;
    dither.temporal = true;
    owner->params.dither_params = &dither;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    // A disabled mixer must select the nearest source, even with multiple inputs.
    struct pl_frame other = image;
    const struct pl_frame *frames[] = { &other, &image };
    uint64_t signatures[] = { 123, 7 };
    float timestamps[] = {-1.0, 0.0};
    struct pl_frame_mix mix = {
        .num_frames = 2, .frames = frames, .signatures = signatures,
        .timestamps = timestamps, .vsync_duration = 1.0,
    };
    pl_renderer_preparation preparation = NULL;
    srand(1234);
    describing = true;
    REQUIRE(pl_renderer_describe_image_mix(renderer, &mix, &target, &snapshot,
                                           &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    forbid_compile = true;
    REQUIRE(pl_renderer_prepare_submit(preparation) == PL_RENDERER_PREPARE_OK);
    pl_clock_t start = pl_clock_now();
    enum pl_pass_prepare_state state;
    while ((state = pl_renderer_prepare_poll(preparation)) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    REQUIRE(state == PL_PASS_PREPARE_READY);
    forbid_compile = false;
    uint8_t expected[8 * 8 * 4], actual[8 * 8 * 4];
    for (int i = 0; i < 4; i++) {
        signatures[1] = 7 + i / 2;
        describing = forbid_compile = true;
        int executions = owner->executions, descriptions = owner->descriptions;
        REQUIRE(pl_renderer_preflight_image_mix(preparation, &mix, &target) ==
                PL_RENDERER_PREPARE_OK);
        REQUIRE(owner->executions == executions);
        if (hook && !direct && i % 2)
            REQUIRE(owner->descriptions == descriptions + 1);
        describing = false;
        enum pl_renderer_prepare_result result = pl_render_image_mix_prepared(preparation, &mix, &target);
        if (result != PL_RENDERER_PREPARE_OK)
            fprintf(stderr, "mix result %d: %s\n", result, pl_renderer_prepare_error(preparation));
        REQUIRE(result == PL_RENDERER_PREPARE_OK);
        forbid_compile = false;
        int ordinary = owner->ordinary, resets = owner->resets;
        // The strict callback must never invoke the ordinary callback/reset.
        if (hook) {
            REQUIRE(owner->analysis_executions == (direct ? i + 1 : i / 2 + 1));
            REQUIRE(owner->output_executions == i + 1);
        }
        if (!i)
            srand(1234);
        REQUIRE(pl_render_image_mix(reference_renderer, &mix, &reference_target, &owner->params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
        REQUIRE_MEMEQ(expected, actual, sizeof(expected));
        if (hook) {
            REQUIRE(owner->ordinary == owner->executions);
            REQUIRE(owner->ordinary - ordinary == 1 + (direct || !(i & 1)));
            REQUIRE(owner->resets > resets);
        }
        // Even after a rejected cached-frame preflight, the valid cache and
        // ready generation survive and can present the original frame again.
        struct pl_frame changed = target;
        changed.repr.sys = PL_COLOR_SYSTEM_BT_2020_C;
        changed.repr.levels = PL_COLOR_LEVELS_LIMITED;
        int runs = gpu_runs, buffer_writes = writes;
        executions = owner->executions;
        signatures[1] += 100;
        describing = forbid_compile = true;
        REQUIRE(pl_renderer_preflight_image_mix(preparation, &mix, &changed) ==
                PL_RENDERER_PREPARE_NOT_READY);
        describing = forbid_compile = false;
        signatures[1] -= 100;
        REQUIRE(gpu_runs == runs && writes == buffer_writes);
        REQUIRE(owner->executions == executions);
        REQUIRE(pl_renderer_prepare_poll(preparation) == PL_PASS_PREPARE_READY);
    }
    pl_renderer_prepare_destroy(&preparation);
    REQUIRE(owner->refs == 1);
    pl_dispatch_destroy(&owner->auxiliary);
    pl_buf_destroy(gpu, &owner->output);
    free(owner);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
    pl_renderer_destroy(&reference_renderer);
    pl_renderer_destroy(&renderer);
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_vulkan vulkan = pl_vulkan_create(log, pl_vulkan_params(.allow_software = true));
    if (!vulkan)
        return SKIP;
    pl_gpu gpu = vulkan->gpu;
    struct pl_vk *priv = PL_PRIV(gpu);
    original = priv->impl;
    priv->impl.tex_create = observe_texture;
    priv->impl.buf_create = observe_buffer;
    priv->impl.pass_create = observe_create;
    priv->impl.pass_run = observe_run;
    priv->impl.pass_run_prepared = observe_prepared_run;
    priv->impl.buf_write = observe_write;
    uint8_t pixels[4 * 4 * 4];
    for (int i = 0; i < sizeof(pixels); i++)
        pixels[i] = (i * 17) % 256;
    pl_tex source = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .initial_data = pixels,
    ));
    REQUIRE(source);
    test_renderer(gpu, frame(source), 4, false, false);
    test_renderer(gpu, frame(source), 8, false, false);
    test_renderer(gpu, frame(source), 8, true, false);
    test_renderer(gpu, frame(source), 8, true, true);
    test_mix(gpu, frame(source), false, true);
    test_mix(gpu, frame(source), true, true);
    test_mix(gpu, frame(source), false, false);
    // Subsampled NV12 and planar 4:2:0, including shifted chroma and EWA
    // plane merging/materialization before chroma reconstruction.
    uint8_t y[16], uv[8], u[4], v[4];
    for (int i = 0; i < 16; i++) y[i] = 32 + i * 12;
    for (int i = 0; i < 4; i++) {
        uv[2*i] = u[i] = 90 + i * 9;
        uv[2*i+1] = v[i] = 160 - i * 8;
    }
    pl_tex planes[4];
    const void *data[] = {y, uv, u, v};
    for (int i = 0; i < 4; i++) {
        planes[i] = pl_tex_create(gpu, pl_tex_params(
            .w = i ? 2 : 4, .h = i ? 2 : 4,
            .format = pl_find_named_fmt(gpu, i == 1 ? "rg8" : "r8"),
            .sampleable = true, .initial_data = data[i]));
        REQUIRE(planes[i]);
    }
    for (int planar = 0; planar < 2; planar++) {
        struct pl_frame yuv = frame(source);
        yuv.num_planes = planar ? 3 : 2;
        yuv.repr = (struct pl_color_repr) {
            .sys = PL_COLOR_SYSTEM_BT_709, .levels = PL_COLOR_LEVELS_LIMITED,
        };
        yuv.planes[0] = (struct pl_plane) {
            .texture = planes[0], .components = 1, .component_mapping = {0},
        };
        yuv.planes[1] = (struct pl_plane) {
            .texture = planes[planar ? 2 : 1], .components = planar ? 1 : 2,
            .component_mapping = {1, 2}, .shift_x = 0.5,
        };
        if (planar) yuv.planes[2] = (struct pl_plane) {
            .texture = planes[3], .components = 1, .component_mapping = {2},
            .shift_x = 0.5,
        };
        test_renderer(gpu, yuv, 4, true, false);
        test_renderer(gpu, yuv, 8, true, true);
        test_mix(gpu, yuv, false, true);
    }
    // CapturePlaneUploader exports three 16-bit UNORM planes. Cover both
    // subsampled chroma and same-size planes without reconstruction scaling.
    for (int full_chroma = 0; full_chroma < 2; full_chroma++) {
        struct pl_frame capture = frame(source);
        capture.num_planes = 3;
        capture.repr = (struct pl_color_repr) {
            .sys = PL_COLOR_SYSTEM_BT_709, .levels = PL_COLOR_LEVELS_LIMITED,
            .bits = {.sample_depth = 16, .color_depth = 10, .bit_shift = 6},
        };
        for (int p = 0; p < 3; p++) {
            int dim = !p || full_chroma ? 4 : 2;
            uint16_t capture_data[16];
            for (int j = 0; j < 16; j++)
                capture_data[j] = (p ? 400 + 4 * j : 64 + 50 * j) << 6;
            capture.planes[p] = (struct pl_plane) {
                .texture = pl_tex_create(gpu, pl_tex_params(
                    .w = dim, .h = dim, .format = pl_find_named_fmt(gpu, "r16"),
                    .sampleable = true, .initial_data = capture_data)),
                .components = 1, .component_mapping = {p, -1, -1, -1},
            };
            REQUIRE(capture.planes[p].texture);
        }
        test_renderer(gpu, capture, 4, true, false);
        test_renderer(gpu, capture, 8, true, true);
        test_mix(gpu, capture, false, true);
        for (int p = 0; p < 3; p++) pl_tex_destroy(gpu, &capture.planes[p].texture);
    }
    for (int i = 0; i < 4; i++) pl_tex_destroy(gpu, &planes[i]);
    pl_tex_destroy(gpu, &source);
    pl_vulkan_destroy(&vulkan);
    pl_log_destroy(&log);
}
