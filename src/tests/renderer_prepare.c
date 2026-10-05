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
    if (execute)
        owner->executions++;
    else
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

static void test_renderer(pl_gpu gpu, pl_tex source, int width, bool hook, bool ewa)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer reference_renderer = pl_renderer_create(gpu->log, gpu);
    REQUIRE(renderer && reference_renderer);
    struct pl_tex_params output_params = {
        .w = width, .h = width, .format = source->params.format,
        .renderable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex output = pl_tex_create(gpu, &output_params);
    pl_tex reference = pl_tex_create(gpu, &output_params);
    REQUIRE(output && reference);
    struct pl_frame image = frame(source), target = frame(output);
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
        pixels[i] = i % 4 == 3 ? 255 : (i * 17) % 256;
    pl_tex source = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .initial_data = pixels,
    ));
    REQUIRE(source);
    test_renderer(gpu, source, 4, false, false);
    test_renderer(gpu, source, 8, false, false);
    test_renderer(gpu, source, 8, true, false);
    test_renderer(gpu, source, 8, true, true);
    pl_tex_destroy(gpu, &source);
    pl_vulkan_destroy(&vulkan);
    pl_log_destroy(&log);
}
