#include "vulkan/gpu.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/shaders/icc.h>
#include "utils.h"
#include "pl_thread.h"
#include "mpv_shader_fixtures.h"
#include "pass_prepare_helpers.h"

static _Thread_local bool describing, forbid_compile;
static struct pl_gpu_fns original;
static int gpu_runs, writes;
static struct prepare_latch replacement_compile;
static atomic_bool fail_replacement;

static pl_pass observe_prepare(pl_gpu gpu, const struct pl_pass_params *params,
                               enum pl_pass_prepare_phase *phase)
{
    latch_block(&replacement_compile);
    if (atomic_exchange(&fail_replacement, false))
        return NULL;
    return original.pass_create_prepared(gpu, params, phase);
}


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

static bool observe_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    REQUIRE(!describing);
    return original.tex_upload(gpu, params);
}

static void observe_clear(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    REQUIRE(!describing);
    original.tex_clear_ex(gpu, tex, color);
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
    pl_shader auxiliary_builder;
    pl_buf output;
    int ordinary, resets, descriptions, executions;
    int analysis_executions, output_executions;
    struct pl_sampler_override sampler;
    int sampler_mode, chroma_samples, image_samples;
    int metadata_samples, bound_samples;
    int sampler_passes;
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
    pl_shader shader = params->context->begin(params->context->priv, owner->auxiliary, false);
    REQUIRE(shader);
    // This dispatch has only one builder in flight. Repeated preflight and
    // execution must recycle it instead of filling an unconsumed shader pool.
    if (owner->auxiliary_builder)
        REQUIRE(shader == owner->auxiliary_builder);
    owner->auxiliary_builder = shader;
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
    enum pl_dispatch_result result = params->context->compute(params->context->priv, owner->auxiliary,
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
    owner->params.profile = getenv("PL_TEST_SHADER_PROFILE") != NULL;
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

// A deliberately different filter proves replacement actually occurs at the
// sampling boundary. Force ordinary compute while preserving src normalization,
// crop/siting and the renderer's current color representation.
static enum pl_dispatch_result sample_override(void *priv, pl_shader sh,
    enum pl_sampler_target target, const struct pl_sample_src *src,
    const struct pl_sample_filter_params *params, const struct pl_hook_context *context)
{
    struct snapshot_owner *owner = priv;
    REQUIRE(target == PL_SAMPLER_CHROMA || target == PL_SAMPLER_IMAGE);
    REQUIRE(src->tex || src->texture);
    REQUIRE(owner->sampler_passes || params->filter.polar);
    REQUIRE(owner->sampler_mode != 4); // no-op/downscale must never enter
    if (target == PL_SAMPLER_CHROMA) owner->chroma_samples++;
    if (target == PL_SAMPLER_IMAGE) owner->image_samples++;
    if (src->tex) owner->bound_samples++; else owner->metadata_samples++;
    if (owner->sampler_mode == 2)
        return PL_DISPATCH_UNSUPPORTED;
    if (owner->sampler_mode == 3)
        return PL_DISPATCH_FAILED;
    if (owner->sampler_mode == 5)
        return pl_shader_sample_polar_cached(sh, src, params)
            ? PL_DISPATCH_OK : PL_DISPATCH_FAILED;
    REQUIRE(!params->no_compute);
    return pl_shader_sample_custom(sh, src, PL_TEX_SAMPLE_LINEAR, &(struct pl_custom_shader) {
        .description = "test scaler CS", .body = "color = vec4(textureLod(SRC, src_pos, 0.0));",
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .compute = true, .compute_group_size = {8, 8},
    }) ? PL_DISPATCH_OK : PL_DISPATCH_FAILED;
}

static enum pl_dispatch_result sample_extended(void *priv, pl_shader sh,
    enum pl_sampler_target target, const struct pl_sample_src *src,
    const struct pl_sample_filter_params *params,
    const struct pl_hook_context *ctx)
{
    struct snapshot_owner *owner = priv;
    if (owner->sampler_passes == 1)
        return sample_override(priv, sh, target, src, params, ctx);

    const struct pl_hook_texture *tmp = ctx->get_tex(ctx->priv, src->new_w, src->new_h, NULL);
    REQUIRE(tmp);
    REQUIRE(!!tmp->texture == !!src->tex);
    pl_shader stage = ctx->begin(ctx->priv, ctx->dispatch, true);
    REQUIRE(stage);
    enum pl_dispatch_result result = sample_override(priv, stage, target, src, params, ctx);
    if (result != PL_DISPATCH_OK) {
        ctx->abort(ctx->priv, ctx->dispatch, &stage);
        return result;
    }
    result = ctx->finish(ctx->priv, ctx->dispatch, pl_dispatch_params(
        .shader = &stage, .target = tmp->texture), &tmp->params);
    if (result != PL_DISPATCH_OK)
        return result;
    struct pl_sample_src intermediate = {
        .tex = tmp->texture, .texture = &tmp->params,
        .sampler_type = tmp->sampler_type,
        .components = src->components, .component_mask = src->component_mask,
        .new_w = src->new_w, .new_h = src->new_h,
    };
    return pl_shader_sample_direct(sh, &intermediate) ? PL_DISPATCH_OK : PL_DISPATCH_FAILED;
}

static void test_renderer_sampler(pl_gpu gpu, struct pl_frame image, int width, bool hook, int ewa, int sampler_mode)
{
    const int test_mode = sampler_mode;
    const int sampler_passes = sampler_mode == 36 || sampler_mode == 38 ? 1 : sampler_mode == 37 ? 2 : 0;
    if (sampler_passes)
        sampler_mode = sampler_mode == 38 ? 5 : 1;
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
    if (sampler_mode >= 7 && sampler_mode <= 9) {
        target.rotation = reference_target.rotation = sampler_mode - 6;
        sampler_mode = 0;
    }
    struct snapshot_owner *owner = new_snapshot(gpu, hook);
    struct pl_deband_params deband = pl_deband_default_params;
    if (sampler_mode == 6) {
        owner->params.deband_params = &deband;
        sampler_mode = 0;
    }
    if (sampler_mode == 10) {
        owner->params.error_diffusion = &pl_error_diffusion_sierra_lite;
        target.repr.bits.color_depth = reference_target.repr.bits.color_depth = 6;
        sampler_mode = 0;
    }
    if (sampler_mode >= 11 && sampler_mode <= 14) {
        target.crop = reference_target.crop = (pl_rect2df) {1, 2, width-1, width-2};
        owner->params.border = sampler_mode - 11;
        owner->params.background_color[0] = 0.25f;
        owner->params.background_color[1] = 0.5f;
        owner->params.background_color[2] = 0.75f;
        owner->params.tile_size = 4;
        owner->params.blur_radius = 1.0f;
        pl_tex_clear(gpu, output, (float[4]) {0.75f, 0.25f, 0.5f, 1.0f});
        pl_tex_clear(gpu, reference, (float[4]) {0.75f, 0.25f, 0.5f, 1.0f});
        sampler_mode = 0;
    }
    struct pl_deinterlace_params deint = pl_deinterlace_default_params;
    if (sampler_mode >= 15 && sampler_mode <= 22) {
        deint.algo = (sampler_mode - 15) / 2;
        image.field = (sampler_mode - 15) % 2 ? PL_FIELD_ODD : PL_FIELD_EVEN;
        image.first_field = PL_FIELD_EVEN;
        owner->params.deinterlace_params = &deint;
        sampler_mode = 0;
    }
    struct pl_overlay_part part = {
        .src = {0, 0, 4, 4}, .dst = {1, 1, 3, 3},
        .color = {0.8f, 0.2f, 0.4f, 0.5f},
    };
    struct pl_overlay overlay = {
        .tex = image.planes[0].texture, .parts = &part, .num_parts = 1,
        .color = pl_color_space_srgb, .repr = pl_color_repr_rgb,
    };
    if (sampler_mode >= 23 && sampler_mode <= 26) {
        overlay.mode = (sampler_mode - 23) % 2;
        if (sampler_mode < 25) {
            image.overlays = &overlay;
            image.num_overlays = 1;
        } else {
            target.overlays = reference_target.overlays = &overlay;
            target.num_overlays = reference_target.num_overlays = 1;
        }
        sampler_mode = 0;
    }
    pl_icc_object icc = NULL;
    struct pl_icc_profile profile = TEST_PROFILE(sRGB_v2_nano_icc);
    if (sampler_mode >= 27 && sampler_mode <= 30) {
        icc = pl_icc_open(gpu->log, &profile, pl_icc_params(
            .size_r = 8, .size_g = 8, .size_b = 8));
        REQUIRE(icc);
        if (sampler_mode == 27) image.icc = icc;
        if (sampler_mode == 28) target.icc = reference_target.icc = icc;
        if (sampler_mode == 29) image.profile = profile;
        if (sampler_mode == 30) target.profile = reference_target.profile = profile;
        sampler_mode = 0;
    }
    if (sampler_mode == 31 || sampler_mode == 32) {
        image.film_grain.type = sampler_mode == 31 ? PL_FILM_GRAIN_AV1 : PL_FILM_GRAIN_H274;
        if (sampler_mode == 31) image.film_grain.params.av1 = av1_grain_data;
        else image.film_grain.params.h274 = h274_grain_data;
        image.film_grain.seed = 1234;
        owner->params.deband_params = &deband;
        sampler_mode = 0;
    }
    if (sampler_mode == 33) {
        owner->params.disable_fbos = true;
        sampler_mode = 0;
    }
    struct pl_distort_params distort = pl_distort_default_params;
    if (sampler_mode == 34 || sampler_mode == 35) {
        distort.transform.mat.m[0][1] = 0.2f;
        distort.transform.c[0] = 0.1f;
        distort.constrain = true;
        distort.bicubic = sampler_mode == 35;
        owner->params.distort_params = &distort;
        // Constrained distortion leaves pixels outside its canvas untouched.
        pl_tex_clear(gpu, output, (float[4]) {0, 0, 0, 1});
        pl_tex_clear(gpu, reference, (float[4]) {0, 0, 0, 1});
        sampler_mode = 0;
    }
    const struct pl_filter_config *filters[] = {
        NULL, &pl_filter_ewa_lanczos, &pl_filter_nearest,
        &pl_filter_bicubic, &pl_filter_hermite, &pl_filter_gaussian,
    };
    if (ewa)
        owner->params.upscaler = filters[ewa];
    owner->sampler_mode = sampler_mode;
    owner->sampler_passes = sampler_passes;
    if (sampler_mode) {
        owner->sampler = (struct pl_sampler_override) {
            .priv = owner, .signature = test_mode,
            .sample = sampler_passes ? sample_extended : sample_override,
        };
        owner->params.sampler_override = &owner->sampler;
    }
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    pl_renderer_preparation preparation = NULL;
    // Both renderers must generate the same blue-noise LUT for exact pixels.
    srand(1234);
    describing = true;
    enum pl_renderer_prepare_result described = pl_renderer_describe_image(
        renderer, &image, &target, &snapshot, &preparation);
    if (sampler_mode == 3) {
        describing = false;
        REQUIRE(described == PL_RENDERER_PREPARE_FAILED);
        pl_renderer_prepare_destroy(&preparation);
        REQUIRE(owner->refs == 1);
        REQUIRE(!pl_render_image(renderer, &image, &target, &owner->params));
        free(owner);
        pl_tex_destroy(gpu, &reference);
        pl_tex_destroy(gpu, &output);
        pl_renderer_destroy(&reference_renderer);
        pl_renderer_destroy(&renderer);
        return;
    }
    REQUIRE(described == PL_RENDERER_PREPARE_OK);
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
    for (int i = 0; i < 64; i++)
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
    reference_params.sampler_override = NULL;
    if (sampler_mode == 1) {
        reference_params.upscaler = &pl_filter_bilinear;
        // Keep the same linear/sigmoid stage as the replaced complex scaler.
        reference_params.disable_builtin_scalers = true;
    }
    if (sampler_mode == 4) {
        REQUIRE(!owner->metadata_samples && !owner->bound_samples);
    } else if (sampler_mode) {
        REQUIRE(owner->metadata_samples > 0 && owner->bound_samples > 0);
        if (width > 4) REQUIRE(owner->image_samples > 0);
        if (image.num_planes > 1) REQUIRE(owner->chroma_samples > 0);
    }
    srand(1234);
    REQUIRE(pl_render_image(reference_renderer, &image, &reference_target, &reference_params));
    uint8_t *expected = malloc(width * width * 4), *actual = malloc(width * width * 4);
    REQUIRE(expected && actual);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    if (sampler_mode == 1) {
        // Separable LUT bilinear and direct compute bilinear may round one
        // 8-bit code differently; larger changes indicate geometry/domain bugs.
        for (int i = 0; i < width * width * 4; i++)
            REQUIRE(abs((int) expected[i] - actual[i]) <= 1);
        // The same callback through ordinary and prepared rendering must be exact.
        reference_params = owner->params;
        reference_params.hooks = NULL;
        reference_params.num_hooks = 0;
        REQUIRE(pl_render_image(reference_renderer, &image, &reference_target, &reference_params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    }
    if (memcmp(expected, actual, width * width * 4))
        fprintf(stderr, "renderer mismatch mode=%d planes=%d\n", test_mode, image.num_planes);
    REQUIRE_MEMEQ(expected, actual, width * width * 4);
    if (image.enhancement_layer) {
        struct pl_frame base_only = image;
        base_only.enhancement_layer = NULL;
        REQUIRE(pl_render_image(reference_renderer, &base_only, &reference_target, &reference_params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
        REQUIRE(memcmp(expected, actual, width * width * 4));
    }
    if (owner->params.error_diffusion) {
        reference_params.error_diffusion = NULL;
        reference_params.dither_params = NULL;
        REQUIRE(pl_render_image(reference_renderer, &image, &reference_target, &reference_params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
        REQUIRE(memcmp(expected, actual, width * width * 4) != 0);
    }

    if (image.film_grain.type) {
        for (int seed = 1; seed <= 3; seed++) {
            image.film_grain.seed = 5678 + seed;
            describing = forbid_compile = true;
            REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
            describing = false;
            REQUIRE(pl_render_image_prepared(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
            forbid_compile = false;
            REQUIRE(pl_render_image(reference_renderer, &image, &reference_target, &reference_params));
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
            REQUIRE_MEMEQ(expected, actual, width * width * 4);
        }
    }

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
    pl_icc_close(&icc);
}

static void test_renderer(pl_gpu gpu, struct pl_frame image, int width, bool hook, bool ewa)
{
    test_renderer_sampler(gpu, image, width, hook, ewa, 0);
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
    struct pl_overlay_part overlay_part = {
        .src = {0, 0, 4, 4}, .dst = {1, 1, 3, 3},
        .color = {0.8f, 0.2f, 0.1f, 0.75f},
    };
    struct pl_overlay overlay = {
        .tex = image.planes[0].texture, .mode = PL_OVERLAY_MONOCHROME,
        .color = pl_color_space_srgb, .repr = pl_color_repr_rgb,
        .parts = &overlay_part, .num_parts = 1,
    };
    image.overlays = &overlay;
    image.num_overlays = 1;
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

static void wait_prepared(pl_renderer_preparation preparation)
{
    REQUIRE(pl_renderer_prepare_submit(preparation) == PL_RENDERER_PREPARE_OK);
    pl_clock_t start = pl_clock_now();
    enum pl_pass_prepare_state state;
    while ((state = pl_renderer_prepare_poll(preparation)) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    if (state != PL_PASS_PREPARE_READY)
        fprintf(stderr, "MPV prepare: %s\n", pl_renderer_prepare_error(preparation));
    REQUIRE(state == PL_PASS_PREPARE_READY);
}

// Generation replacement must preserve the active renderer through pending,
// cancelled and failed candidates. The compiler barrier makes overlap certain.
static void test_replacement_lifecycle(pl_gpu gpu, struct pl_frame image)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    struct snapshot_owner *active_owner = new_snapshot(gpu, true);
    active_owner->params.dither_params = NULL;
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true));
    REQUIRE(output);
    struct pl_frame target = frame(output);
    struct pl_renderer_snapshot snapshot = {
        .params = &active_owner->params, .owner = active_owner,
        .retain = retain, .release = release,
    };
    pl_renderer_preparation active = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot, &active) == PL_RENDERER_PREPARE_OK);
    describing = false;
    wait_prepared(active);
    uint8_t baseline[8 * 8 * 4], actual[sizeof(baseline)];
    forbid_compile = true;
    REQUIRE(pl_render_image_prepared(active, &image, &target) == PL_RENDERER_PREPARE_OK);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = baseline)));
    forbid_compile = false;
    // Source identity changes of the same layout are runtime bindings. Loss
    // and incompatible geometry must leave both output and analysis untouched.
    uint8_t replacement_pixels[4 * 4 * 4];
    memset(replacement_pixels, 255, sizeof(replacement_pixels));
    struct pl_tex_params source_params = image.planes[0].texture->params;
    source_params.initial_data = replacement_pixels;
    pl_tex replacement = pl_tex_create(gpu, &source_params);
    REQUIRE(replacement);
    struct pl_frame rebound = image;
    rebound.planes[0].texture = replacement;
    forbid_compile = true;
    REQUIRE(pl_render_image_prepared(active, &rebound, &target) == PL_RENDERER_PREPARE_OK);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    REQUIRE(memcmp(actual, baseline, sizeof(actual)));
    forbid_compile = false;
    struct pl_tex_params changed_params = output->params;
    changed_params.w++;
    pl_tex resized = pl_tex_create(gpu, &changed_params);
    REQUIRE(resized);
    struct pl_frame resized_target = frame(resized);
    for (int event = 0; event < 3; event++) {
        int runs = gpu_runs, before_writes = writes;
        int executions = active_owner->executions;
        describing = event < 2;
        forbid_compile = true;
        enum pl_renderer_prepare_result result = pl_render_image_prepared(active,
            event == 0 ? NULL : &image, event == 1 ? &resized_target : &target);
        describing = forbid_compile = false;
        if (event < 2) {
            REQUIRE(result == PL_RENDERER_PREPARE_NOT_READY);
            REQUIRE(gpu_runs == runs && writes == before_writes);
            REQUIRE(active_owner->executions == executions);
        } else {
            REQUIRE(result == PL_RENDERER_PREPARE_OK);
        }
    }
    pl_tex_destroy(gpu, &resized);
    pl_tex_destroy(gpu, &replacement);
    for (int outcome = 0; outcome < 3; outcome++) {
        struct snapshot_owner *owner = new_snapshot(gpu, true);
        owner->params.dither_params = NULL;
        struct pl_color_adjustment adjustment = pl_color_adjustment_neutral;
        adjustment.brightness = 0.2f;
        owner->params.color_adjustment = &adjustment;
        snapshot.params = &owner->params;
        snapshot.owner = owner;
        pl_renderer_preparation candidate = NULL;
        describing = true;
        REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot, &candidate) == PL_RENDERER_PREPARE_OK);
        describing = false;
        latch_arm(&replacement_compile);
        atomic_store(&fail_replacement, outcome == 1);
        REQUIRE(pl_renderer_prepare_submit(candidate) == PL_RENDERER_PREPARE_OK);
        latch_entered(&replacement_compile);
        REQUIRE(pl_renderer_prepare_poll(candidate) == PL_PASS_PREPARE_PENDING);
        int runs = gpu_runs, before_writes = writes;
        REQUIRE(pl_render_image_prepared(candidate, &image, &target) == PL_RENDERER_PREPARE_NOT_READY);
        REQUIRE(gpu_runs == runs && writes == before_writes);
        forbid_compile = true;
        REQUIRE(pl_render_image_prepared(active, &image, &target) == PL_RENDERER_PREPARE_OK);
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
        REQUIRE(!memcmp(actual, baseline, sizeof(actual)));
        forbid_compile = false;
        if (outcome == 0) {
            pl_renderer_prepare_destroy(&candidate);
            REQUIRE(owner->refs == 1); // cancellation cannot wait on the compiler
        }
        latch_release(&replacement_compile);
        // Do not rearm until the worker has consumed this release.
        pl_clock_t released = pl_clock_now();
        for (;;) {
            pl_mutex_lock(&replacement_compile.lock);
            bool armed = replacement_compile.armed;
            pl_mutex_unlock(&replacement_compile.lock);
            if (!armed) break;
            REQUIRE(pl_clock_diff(pl_clock_now(), released) < 10.0);
            pl_thread_sleep(0.001);
        }
        if (candidate) {
            pl_clock_t start = pl_clock_now();
            enum pl_pass_prepare_state state;
            while ((state = pl_renderer_prepare_poll(candidate)) == PL_PASS_PREPARE_PENDING) {
                REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
                pl_thread_sleep(0.001);
            }
            REQUIRE(state == (outcome == 1 ? PL_PASS_PREPARE_FAILED : PL_PASS_PREPARE_READY));
            forbid_compile = true;
            if (outcome == 2) {
                REQUIRE(pl_render_image_prepared(candidate, &image, &target) == PL_RENDERER_PREPARE_OK);
                REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
                REQUIRE(memcmp(actual, baseline, sizeof(actual)));
            }
            // Even executing a replacement does not mutate the old handle.
            REQUIRE(pl_render_image_prepared(active, &image, &target) == PL_RENDERER_PREPARE_OK);
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
            REQUIRE(!memcmp(actual, baseline, sizeof(actual)));
            forbid_compile = false;
            pl_renderer_prepare_destroy(&candidate);
            REQUIRE(owner->refs == 1);
        }
        pl_buf_destroy(gpu, &owner->output);
        pl_dispatch_destroy(&owner->auxiliary);
        free(owner);
    }
    pl_renderer_prepare_destroy(&active);
    REQUIRE(active_owner->refs == 1);
    pl_buf_destroy(gpu, &active_owner->output);
    pl_dispatch_destroy(&active_owner->auxiliary);
    free(active_owner);
    pl_tex_destroy(gpu, &output);
    pl_renderer_destroy(&renderer);
}

static void test_empty_render(pl_gpu gpu, struct pl_frame image)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer native = pl_renderer_create(gpu->log, gpu);
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true));
    pl_tex reference = pl_tex_create(gpu, &output->params);
    REQUIRE(output && reference);
    for (int absent = 0; absent < 2; absent++) {
        for (int border = PL_CLEAR_COLOR; border <= PL_CLEAR_BLUR; border++) {
            for (int overlays = 0; overlays < 2; overlays++) {
                struct snapshot_owner *owner = new_snapshot(gpu, false);
                owner->params.border = border;
                owner->params.background_color[0] = 0.25f;
                owner->params.tile_size = 2;
                struct pl_frame target = frame(output), expected = frame(reference);
                if (!absent)
                    target.crop = expected.crop = (pl_rect2df) {20, 20, 24, 24};
                struct pl_overlay_part part = {
                    .src = {0, 0, 4, 4}, .dst = {1, 1, 3, 3}, .color = {1, 1, 1, 1},
                };
                struct pl_overlay overlay = {
                    .tex = image.planes[0].texture, .parts = &part, .num_parts = 1,
                    .color = pl_color_space_srgb, .repr = pl_color_repr_rgb,
                };
                target.overlays = expected.overlays = &overlay;
                target.num_overlays = expected.num_overlays = overlays;
                const struct pl_frame *input = absent ? NULL : &image;
                struct pl_renderer_snapshot snapshot = {
                    .params = &owner->params, .owner = owner, .retain = retain, .release = release,
                };
                pl_renderer_preparation p = NULL;
                describing = true;
                REQUIRE(pl_renderer_describe_image(renderer, input, &target, &snapshot, &p) == PL_RENDERER_PREPARE_OK);
                describing = false;
                wait_prepared(p);
                const float initial[4] = {0.2f, 0.3f, 0.4f, 1.0f};
                pl_tex_clear(gpu, output, initial);
                pl_tex_clear(gpu, reference, initial);
                describing = forbid_compile = true;
                REQUIRE(pl_renderer_preflight_image(p, input, &target) == PL_RENDERER_PREPARE_OK);
                describing = false;
                REQUIRE(pl_render_image_prepared(p, input, &target) == PL_RENDERER_PREPARE_OK);
                forbid_compile = false;
                REQUIRE(pl_render_image(native, input, &expected, &owner->params));
                uint8_t actual[256], wanted[256];
                REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
                REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = wanted)));
                if (memcmp(actual, wanted, sizeof(actual)))
                    fprintf(stderr, "empty rendering mismatch absent=%d border=%d overlays=%d\n", absent, border, overlays);
                REQUIRE_MEMEQ(actual, wanted, sizeof(actual));
                pl_renderer_prepare_destroy(&p);
                REQUIRE(owner->refs == 1);
                pl_buf_destroy(gpu, &owner->output);
                pl_dispatch_destroy(&owner->auxiliary);
                free(owner);
            }
        }
    }
    pl_tex_destroy(gpu, &output);
    pl_tex_destroy(gpu, &reference);
    pl_renderer_destroy(&renderer);
    pl_renderer_destroy(&native);
}

struct frame_access {
    int acquired, released;
    bool fail;
    pl_tex replacement;
    int *mapped_sources;
};

static bool test_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    REQUIRE(!describing);
    struct frame_access *access = frame->user_data;
    REQUIRE(access->acquired == access->released);
    access->acquired++;
    if (access->mapped_sources) REQUIRE(++*access->mapped_sources == 1);
    if (access->replacement)
        frame->planes[0].texture = access->replacement;
    return !access->fail;
}

static void test_release(pl_gpu gpu, struct pl_frame *frame)
{
    REQUIRE(!describing);
    struct frame_access *access = frame->user_data;
    REQUIRE(access->acquired == access->released + 1);
    access->released++;
    if (access->mapped_sources) REQUIRE(--*access->mapped_sources == 0);
}

static void test_frame_access(pl_gpu gpu, struct pl_frame image, bool neighbors)
{
    struct snapshot_owner *owner = new_snapshot(gpu, true);
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"), .renderable = true));
    REQUIRE(output);
    struct pl_frame target = frame(output);
    struct frame_access input_access = {0}, target_access = {0};
    struct frame_access prev_access = {0}, next_access = {0};
    struct pl_frame prev = image, next = image;
    struct pl_deinterlace_params deint = pl_deinterlace_default_params;
    if (neighbors) {
        deint.algo = PL_DEINTERLACE_YADIF;
        owner->params.deinterlace_params = &deint;
        image.field = PL_FIELD_EVEN;
        image.first_field = PL_FIELD_EVEN;
        image.prev = &prev;
        image.next = &next;
        prev.acquire = next.acquire = test_acquire;
        prev.release = next.release = test_release;
        prev.user_data = &prev_access;
        next.user_data = &next_access;
    }
    image.acquire = target.acquire = test_acquire;
    image.release = target.release = test_release;
    image.user_data = &input_access;
    target.user_data = &target_access;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    pl_renderer_preparation p = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot, &p) == PL_RENDERER_PREPARE_OK);
    describing = false;
    wait_prepared(p);
    struct pl_tex_params resized_params = image.planes[0].texture->params;
    resized_params.w++;
    resized_params.initial_data = NULL;
    pl_tex resized = pl_tex_create(gpu, &resized_params);
    REQUIRE(resized);
    for (int event = 0; event < (neighbors ? 6 : 4); event++) {
        input_access.fail = event == 1;
        prev_access.fail = event == 4;
        next_access.fail = event == 5;
        target_access.fail = event == 2;
        input_access.replacement = event == 3 ? resized : NULL;
        int inputs = input_access.acquired, targets = target_access.acquired;
        describing = forbid_compile = true;
        REQUIRE(pl_renderer_preflight_image(p, &image, &target) == PL_RENDERER_PREPARE_OK);
        describing = false;
        REQUIRE(input_access.acquired == inputs && target_access.acquired == targets);
        int runs = gpu_runs, before_writes = writes, executions = owner->executions;
        enum pl_renderer_prepare_result result = pl_render_image_prepared(p, &image, &target);
        forbid_compile = false;
        REQUIRE(result == (event == 0 ? PL_RENDERER_PREPARE_OK : event == 3
                ? (neighbors ? PL_RENDERER_PREPARE_INVALID : PL_RENDERER_PREPARE_NOT_READY)
                : PL_RENDERER_PREPARE_FAILED));
        REQUIRE(target_access.acquired == targets + 1);
        REQUIRE(input_access.acquired == inputs + (event != 2));
        REQUIRE(input_access.acquired == input_access.released);
        REQUIRE(target_access.acquired == target_access.released);
        REQUIRE(prev_access.acquired == prev_access.released);
        REQUIRE(next_access.acquired == next_access.released);
        if (event) {
            REQUIRE(gpu_runs == runs && writes == before_writes);
            REQUIRE(owner->executions == executions);
        }
    }
    input_access = (struct frame_access) {0};
    target_access = (struct frame_access) {0};
    prev_access = (struct frame_access) {0};
    next_access = (struct frame_access) {0};
    forbid_compile = true;
    REQUIRE(pl_render_image_prepared(p, &image, &target) == PL_RENDERER_PREPARE_OK);
    forbid_compile = false;
    REQUIRE(input_access.acquired == 1 && input_access.released == 1);
    REQUIRE(target_access.acquired == 1 && target_access.released == 1);
    pl_renderer_prepare_destroy(&p);
    REQUIRE(owner->refs == 1);
    pl_buf_destroy(gpu, &owner->output);
    pl_dispatch_destroy(&owner->auxiliary);
    free(owner);
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &output);
    pl_tex_destroy(gpu, &resized);
}

static struct pl_hook_prepare_result failed_description(void *priv,
    const struct pl_hook_prepare_params *params)
{
    return (struct pl_hook_prepare_result) { .status = PL_DISPATCH_FAILED };
}

static void test_preparation_policy(pl_gpu gpu, struct pl_frame image)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true));
    REQUIRE(output);
    struct pl_frame target = frame(output);
    for (int mode = PL_RENDERER_PREPARE_ASYNC; mode <= PL_RENDERER_PREPARE_SYNC; mode++) {
        for (int legacy = 0; legacy < 2; legacy++) {
            for (int mixing = 0; mixing < 2; mixing++) {
                struct snapshot_owner *owner = new_snapshot(gpu, true);
                if (legacy) {
                    owner->hook.describe = NULL;
                    owner->hook.execute_prepared = NULL;
                }
                struct pl_renderer_snapshot snapshot = {
                    .params = &owner->params, .owner = owner,
                    .retain = retain, .release = release, .mode = mode,
                };
                const struct pl_frame *frames[] = {&image};
                uint64_t signature = 7;
                float timestamp = 0;
                struct pl_frame_mix mix = {
                    .frames = frames, .signatures = &signature,
                    .timestamps = &timestamp, .num_frames = 1, .vsync_duration = 1,
                };
                pl_renderer_preparation p = NULL;
                describing = true;
                enum pl_renderer_prepare_result result = mixing
                    ? pl_renderer_describe_image_mix(renderer, &mix, &target, &snapshot, &p)
                    : pl_renderer_describe_image(renderer, &image, &target, &snapshot, &p);
                describing = false;
                REQUIRE(!owner->ordinary && !owner->executions);
                if (legacy && mode == PL_RENDERER_PREPARE_ASYNC) {
                    REQUIRE(result == PL_RENDERER_PREPARE_UNSUPPORTED && !p);
                } else {
                    REQUIRE(result == PL_RENDERER_PREPARE_OK && p);
                    bool synchronous = mode == PL_RENDERER_PREPARE_SYNC || legacy;
                    REQUIRE(pl_renderer_prepare_mode(p) == (synchronous
                        ? PL_RENDERER_PREPARE_SYNC : PL_RENDERER_PREPARE_ASYNC));
                    REQUIRE(pl_renderer_prepare_poll(p) == PL_PASS_PREPARE_PENDING);
                    forbid_compile = true;
                    wait_prepared(p);
                    REQUIRE(!owner->ordinary && !owner->executions);
                    describing = true;
                    REQUIRE((mixing ? pl_renderer_preflight_image_mix(p, &mix, &target)
                                    : pl_renderer_preflight_image(p, &image, &target)) == PL_RENDERER_PREPARE_OK);
                    describing = false;
                    forbid_compile = !synchronous;
                    REQUIRE((mixing ? pl_render_image_mix_prepared(p, &mix, &target)
                                    : pl_render_image_prepared(p, &image, &target)) == PL_RENDERER_PREPARE_OK);
                    forbid_compile = false;
                    REQUIRE(synchronous ? owner->ordinary > 0 : owner->executions > 0);
                    REQUIRE(synchronous ? !owner->executions : !owner->ordinary);
                }
                pl_renderer_prepare_destroy(&p);
                REQUIRE(owner->refs == 1);
                pl_dispatch_destroy(&owner->auxiliary);
                pl_buf_destroy(gpu, &owner->output);
                free(owner);
            }
        }
    }
    struct snapshot_owner *owner = new_snapshot(gpu, true);
    owner->hook.describe = failed_description;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain,
        .release = release, .mode = PL_RENDERER_PREPARE_AUTO,
    };
    pl_renderer_preparation p = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot, &p) ==
            PL_RENDERER_PREPARE_FAILED);
    REQUIRE(!p && owner->refs == 1 && !owner->ordinary);
    snapshot.mode = -1;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot, &p) ==
            PL_RENDERER_PREPARE_INVALID);
    describing = false;
    owner->params.num_hooks = 0;
    for (int mode = PL_RENDERER_PREPARE_ASYNC; mode <= PL_RENDERER_PREPARE_SYNC; mode++) {
        snapshot.mode = mode;
        struct pl_frame_mix empty = {0};
        describing = true;
        enum pl_renderer_prepare_result result =
            pl_renderer_describe_image_mix(renderer, &empty, &target, &snapshot, &p);
        describing = false;
        REQUIRE(result == PL_RENDERER_PREPARE_OK);
        REQUIRE(pl_renderer_prepare_mode(p) == (mode == PL_RENDERER_PREPARE_SYNC
                ? PL_RENDERER_PREPARE_SYNC : PL_RENDERER_PREPARE_ASYNC));
        wait_prepared(p);
        REQUIRE(pl_render_image_mix_prepared(p, &empty, &target) == PL_RENDERER_PREPARE_OK);
        pl_renderer_prepare_destroy(&p);
    }
    REQUIRE(owner->refs == 1);
    pl_dispatch_destroy(&owner->auxiliary);
    pl_buf_destroy(gpu, &owner->output);
    free(owner);
    pl_tex_destroy(gpu, &output);
    pl_renderer_destroy(&renderer);
}

static void test_temporal_mix(pl_gpu gpu, const struct pl_filter_config *filter, bool dither, bool hdr, bool wide)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer native = pl_renderer_create(gpu->log, gpu);
    pl_tex textures[3];
    struct pl_frame inputs[3];
    for (int i = 0; i < 3; i++) {
        uint8_t pixels[8 * 8 * 4];
        for (int j = 0; j < 64; j++) {
            pixels[4*j] = i == 0 ? 240 : 20;
            pixels[4*j+1] = i == 1 ? 220 : 10;
            pixels[4*j+2] = i == 2 ? 200 : 30;
            pixels[4*j+3] = 255;
        }
        textures[i] = pl_tex_create(gpu, pl_tex_params(
            .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
            .sampleable = true, .initial_data = pixels));
        REQUIRE(textures[i]);
        inputs[i] = frame(textures[i]);
        if (hdr) inputs[i].color = pl_color_space_hdr10;
    }
    int mapped_sources = 0;
    struct frame_access source_access[3] = {0};
    for (int i = 0; i < 3; i++) {
        source_access[i].mapped_sources = &mapped_sources;
        inputs[i].user_data = &source_access[i];
        inputs[i].acquire = test_acquire;
        inputs[i].release = test_release;
    }
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true));
    pl_tex reference = pl_tex_create(gpu, &output->params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output), expected_target = frame(reference);
    struct snapshot_owner *owner = new_snapshot(gpu, true);
    owner->params.frame_mixer = filter;
    struct pl_peak_detect_params detect = pl_peak_detect_default_params;
    detect.allow_delayed = false;
    owner->params.peak_detect_params = hdr ? &detect : NULL;
    struct pl_dither_params temporal_dither = pl_dither_default_params;
    temporal_dither.method = PL_DITHER_ORDERED_FIXED;
    temporal_dither.temporal = true;
    owner->params.dither_params = dither ? &temporal_dither : NULL;
    owner->hook.stages |= PL_HOOK_OUTPUT;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    const struct pl_frame *frames[17];
    uint64_t signatures[17];
    float timestamps[17];
    for (int i = 0; i < 17; i++) {
        frames[i] = &inputs[i % 3];
        signatures[i] = 10 + i;
        timestamps[i] = i - 0.25f;
    }
    struct pl_frame_mix mix = {
        .num_frames = wide ? 17 : 3, .frames = frames, .signatures = signatures,
        .timestamps = timestamps, .vsync_duration = 1.0f,
    };
    pl_renderer_preparation preparation = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image_mix(renderer, &mix, &target, &snapshot,
                                           &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    forbid_compile = true;
    wait_prepared(preparation);
    forbid_compile = false;
    for (int step = 0; step < (wide ? 1 : 6); step++) {
        if (step == 1) { timestamps[0] = -0.75f; timestamps[1] = 0.25f; }
        if (step == 2) mix.num_frames = 1;
        if (step == 3) mix.num_frames = 3;
        if (step == 4) {
            frames[0] = &inputs[1]; frames[1] = &inputs[2]; frames[2] = &inputs[0];
            signatures[0] = 11; signatures[1] = 12; signatures[2] = 13;
        }
        int executions = owner->executions;
        describing = forbid_compile = true;
        enum pl_renderer_prepare_result result =
            pl_renderer_preflight_image_mix(preparation, &mix, &target);
        if (result != PL_RENDERER_PREPARE_OK)
            fprintf(stderr, "temporal preflight step %d: %d %s\n", step, result,
                    pl_renderer_prepare_error(preparation));
        REQUIRE(result == PL_RENDERER_PREPARE_OK);
        REQUIRE(owner->executions == executions);
        describing = false;
        result = pl_render_image_mix_prepared(preparation, &mix, &target);
        if (result != PL_RENDERER_PREPARE_OK)
            fprintf(stderr, "temporal execute step %d: %d %s\n", step, result,
                    pl_renderer_prepare_error(preparation));
        REQUIRE(result == PL_RENDERER_PREPARE_OK);
        forbid_compile = false;
        REQUIRE(mapped_sources == 0);
        for (int i = 0; i < 3; i++)
            REQUIRE(source_access[i].acquired == source_access[i].released);
        REQUIRE(owner->output_executions == step + 1);
        if (step == 1 || step == 2 || step == 5)
            REQUIRE(owner->executions == executions + 1);
        REQUIRE(pl_render_image_mix(native, &mix, &expected_target, &owner->params));
        REQUIRE(owner->ordinary == owner->executions);
        uint8_t actual[256], expected[256];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
        if (memcmp(actual, expected, sizeof(actual)))
            fprintf(stderr, "temporal pixel mismatch step=%d dither=%d\n", step, dither);
        REQUIRE_MEMEQ(actual, expected, sizeof(actual));
        if (!step) REQUIRE(actual[0] > 20 && actual[1] > 10);
        // A later source mismatch must reject the whole mix before the first
        // source is populated. Exercise the direct render entry point too.
        if (!step) {
            struct pl_frame changed = *frames[1];
            changed.planes[0].shift_x = 0.5f;
            const struct pl_frame *saved = frames[1];
            frames[1] = &changed;
            signatures[0] += 100;
            signatures[1] += 100;
            int runs = gpu_runs, before_writes = writes;
            executions = owner->executions;
            describing = forbid_compile = true;
            REQUIRE(pl_render_image_mix_prepared(preparation, &mix, &target) ==
                    PL_RENDERER_PREPARE_NOT_READY);
            describing = forbid_compile = false;
            REQUIRE(gpu_runs == runs && writes == before_writes);
            REQUIRE(owner->executions == executions);
            frames[1] = saved;
            signatures[0] -= 100;
            signatures[1] -= 100;
            REQUIRE(pl_renderer_prepare_poll(preparation) == PL_PASS_PREPARE_READY);
        }
    }
    pl_renderer_prepare_destroy(&preparation);
    REQUIRE(owner->refs == 1);
    pl_dispatch_destroy(&owner->auxiliary);
    pl_buf_destroy(gpu, &owner->output);
    free(owner);
    pl_tex_destroy(gpu, &output);
    pl_tex_destroy(gpu, &reference);
    for (int i = 0; i < 3; i++) pl_tex_destroy(gpu, &textures[i]);
    pl_renderer_destroy(&native);
    pl_renderer_destroy(&renderer);
}

static struct pl_hook_prepare_result large_graph(struct snapshot_owner *owner,
    const struct pl_hook_prepare_params *params, bool execute)
{
    for (int i = 0; i < 70; i++) {
        struct pl_hook_prepare_result result = auxiliary_hook(owner, params, execute);
        if (result.status != PL_DISPATCH_OK)
            return result;
    }
    return (struct pl_hook_prepare_result) { .status = PL_DISPATCH_OK };
}

static struct pl_hook_prepare_result describe_large_graph(void *priv,
    const struct pl_hook_prepare_params *params)
{
    return large_graph(priv, params, false);
}

static struct pl_hook_prepare_result execute_large_graph(void *priv,
    const struct pl_hook_prepare_params *params)
{
    return large_graph(priv, params, true);
}

static void test_large_graph(pl_gpu gpu, struct pl_frame image)
{
    struct snapshot_owner *owner = new_snapshot(gpu, true);
    owner->hook.describe = describe_large_graph;
    owner->hook.execute_prepared = execute_large_graph;
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true));
    REQUIRE(output);
    struct pl_frame target = frame(output);
    const struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    // Retire a partially admitted candidate, then complete the same large graph.
    for (int attempt = 0; attempt < 2; attempt++) {
        pl_renderer_preparation preparation = NULL;
        describing = true;
        REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot,
                                           &preparation) == PL_RENDERER_PREPARE_OK);
        describing = false;
        if (!attempt) {
            REQUIRE(pl_renderer_prepare_submit(preparation) == PL_RENDERER_PREPARE_OK);
        } else {
            wait_prepared(preparation);
            forbid_compile = true;
            int before = gpu_runs;
            REQUIRE(pl_render_image_prepared(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
            REQUIRE(gpu_runs - before >= 71);
            forbid_compile = false;
            uint32_t result = 0;
            REQUIRE(pl_buf_read(gpu, owner->output, 0, &result, sizeof(result)));
            REQUIRE(result == 42);
        }
        pl_renderer_prepare_destroy(&preparation);
        REQUIRE(owner->refs == 1);
    }
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &output);
    pl_buf_destroy(gpu, &owner->output);
    pl_dispatch_destroy(&owner->auxiliary);
    free(owner);
}

static void test_mpv_prepared(pl_gpu gpu, struct pl_frame image)
{
    static const char shader[] =
        "//!PARAM enabled\n"
        "//!TYPE CONSTANT int\n"
        "0\n"
        "//!PARAM gain\n"
        "//!TYPE DYNAMIC float\n"
        "1.0\n"
        "//!PARAM widthfactor\n"
        "//!TYPE DYNAMIC int\n"
        "1\n"
        "//!HOOK LUMA\n"
        "//!BIND HOOKED\n"
        "vec4 hook() { return HOOKED_texOff(0); }\n"
        "//!HOOK CHROMA\n"
        "//!BIND HOOKED\n"
        "vec4 hook() { return HOOKED_texOff(0); }\n"
        "//!HOOK NATIVE\n"
        "//!BIND HOOKED\n"
        "//!WIDTH HOOKED.w 2 * widthfactor *\n"
        "//!HEIGHT HOOKED.h 2 *\n"
        "//!SAVE UPSCALED\n"
        "//!WHEN enabled 0 =\n"
        "vec4 hook() { return HOOKED_texOff(0); }\n"
        "//!HOOK MAIN\n"
        "//!BIND HOOKED\n"
        "//!BIND UPSCALED\n"
        "//!BIND STATIC_TEX\n"
        "//!BIND STATIC_BUF\n"
        "//!COMPUTE 8 8\n"
        "//!WHEN UPSCALED.w 8 =\n"
        "void hook() { imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), "
        "gain * UPSCALED_tex(HOOKED_pos) + 0.0 * STATIC_TEX_tex(HOOKED_pos) + "
        "vec4(0.0 * unused)); }\n"
        "//!TEXTURE STATIC_TEX\n"
        "//!SIZE 1 1\n"
        "//!FORMAT rgba8\n"
        "//!FILTER NEAREST\n"
        "00000000\n"
        "//!BUFFER STATIC_BUF\n"
        "//!VAR float unused\n"
        "00000000\n";

    const struct pl_hook *hook = pl_mpv_user_shader_parse(gpu, shader, sizeof(shader) - 1);
    REQUIRE(hook && hook->describe && hook->execute_prepared);
    REQUIRE(hook->prepared_state_size > 0 && hook->reset_prepared);
    struct snapshot_owner *owner = new_snapshot(gpu, false);
    owner->params.dither_params = NULL;
    owner->params.hooks = &hook;
    owner->params.num_hooks = 1;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    struct pl_tex_params output_params = {
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex output = pl_tex_create(gpu, &output_params);
    pl_tex reference = pl_tex_create(gpu, &output_params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output), reference_target = frame(reference);
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer ordinary = pl_renderer_create(gpu->log, gpu);
    REQUIRE(renderer && ordinary);
    pl_renderer_preparation preparation = NULL;

    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot,
                                       &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(preparation && owner->refs == 2);
    wait_prepared(preparation);
    forbid_compile = describing = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(pl_render_image_prepared(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    forbid_compile = false;

    REQUIRE(pl_render_image(ordinary, &image, &reference_target, &owner->params));
    uint8_t expected[8 * 8 * 4], actual[8 * 8 * 4];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    REQUIRE_MEMEQ(expected, actual, sizeof(expected));

    for (int i = 0; i < 2; i++) {
        uint8_t previous[sizeof(actual)];
        memcpy(previous, actual, sizeof(actual));
        hook->parameters[1].data->f = i ? 0.0f : 0.5f;
        describing = forbid_compile = true;
        REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
                PL_RENDERER_PREPARE_OK);
        describing = false;
        REQUIRE(pl_render_image_prepared(preparation, &image, &target) ==
                PL_RENDERER_PREPARE_OK);
        forbid_compile = false;
        REQUIRE(pl_render_image(ordinary, &image, &reference_target, &owner->params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
        REQUIRE_MEMEQ(expected, actual, sizeof(expected));
        REQUIRE(memcmp(previous, actual, sizeof(actual)));
    }

    hook->parameters[2].data->i = 2;
    int runs_before = gpu_runs;
    describing = forbid_compile = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_NOT_READY);
    REQUIRE(gpu_runs == runs_before);
    hook->parameters[2].data->i = 1;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    describing = forbid_compile = false;

    hook->parameters[0].data->i = 1;
    describing = forbid_compile = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_NOT_READY);
    describing = forbid_compile = false;
    hook->parameters[0].data->i = 0;
    REQUIRE(pl_renderer_prepare_poll(preparation) == PL_PASS_PREPARE_READY);

    pl_renderer_prepare_destroy(&preparation);
    REQUIRE(owner->refs == 1);
    free(owner);
    pl_renderer_destroy(&ordinary);
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
    pl_mpv_user_shader_destroy(&hook);
}

static void test_mpv_corpus_shader(pl_gpu gpu, struct pl_frame image,
                                   const char *shader)
{
    const struct pl_hook *hook = pl_mpv_user_shader_parse(gpu, shader, strlen(shader));
    REQUIRE(hook && hook->describe && hook->execute_prepared);
    REQUIRE(hook->prepared_state_size > 0 && hook->reset_prepared);

    struct snapshot_owner *owner = new_snapshot(gpu, false);
    owner->params.dither_params = NULL;
    owner->params.hooks = &hook;
    owner->params.num_hooks = 1;
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    struct pl_tex_params output_params = {
        .w = 8, .h = 8, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true, .blit_dst = true,
    };
    pl_tex output = pl_tex_create(gpu, &output_params);
    pl_tex reference = pl_tex_create(gpu, &output_params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output), reference_target = frame(reference);
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer ordinary = pl_renderer_create(gpu->log, gpu);
    REQUIRE(renderer && ordinary);

    const int live_runs_before = gpu_runs;
    REQUIRE(pl_render_image(ordinary, &image, &reference_target, &owner->params));
    const int live_runs = gpu_runs - live_runs_before;
    REQUIRE(live_runs > 0);

    pl_renderer_preparation preparation = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot,
                                       &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(preparation && owner->refs == 2);
    wait_prepared(preparation);

    const int prepared_runs_before = gpu_runs;
    forbid_compile = describing = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(pl_render_image_prepared(preparation, &image, &target) ==
            PL_RENDERER_PREPARE_OK);
    forbid_compile = false;
    REQUIRE(gpu_runs - prepared_runs_before == live_runs);

    uint8_t expected[8 * 8 * 4], actual[8 * 8 * 4];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    REQUIRE_MEMEQ(expected, actual, sizeof(expected));

    pl_renderer_prepare_destroy(&preparation);
    REQUIRE(owner->refs == 1);
    free(owner);
    pl_renderer_destroy(&ordinary);
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
    pl_mpv_user_shader_destroy(&hook);
}

static void test_mpv_corpus(pl_gpu gpu, struct pl_frame image)
{
    for (int i = 0; i < PL_ARRAY_SIZE(user_shader_tests); i++)
        test_mpv_corpus_shader(gpu, image, user_shader_tests[i]);

    if (gpu->glsl.compute && gpu->limits.max_ssbo_size) {
        for (int i = 0; i < PL_ARRAY_SIZE(compute_shader_tests); i++)
            test_mpv_corpus_shader(gpu, image, compute_shader_tests[i]);
    }
}

static struct pl_hook_res identity_color_hook(void *priv, const struct pl_hook_params *p)
{
    REQUIRE(pl_shader_custom(p->sh, &(struct pl_custom_shader) {
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .body = "color.rgb += vec3(0.0);",
        .output_w = pl_rect_w(p->rect), .output_h = pl_rect_h(p->rect),
    }));
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = p->sh, .repr = p->repr,
        .color = p->color, .components = p->components, .rect = p->rect,
    };
}

static struct pl_hook_prepare_result identity_color_prepared(void *priv,
    const struct pl_hook_prepare_params *p)
{
    REQUIRE(pl_shader_custom(p->sh, &(struct pl_custom_shader) {
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .body = "color.rgb += vec3(0.0);",
        .output_w = pl_rect_w(p->rect), .output_h = pl_rect_h(p->rect),
    }));
    return (struct pl_hook_prepare_result) {
        .status = PL_DISPATCH_OK,
        .output = PL_HOOK_SIG_COLOR, .sh = p->sh, .repr = p->repr,
        .color = p->color, .components = p->components, .rect = p->rect,
    };
}

static void test_color_lut_prepared(pl_gpu gpu, struct pl_frame image,
                                    enum pl_lut_type type, int location, int injected, enum pl_color_working_space model)
{
    printf("LUT preparation: type=%d location=%d injected=%d\n", type, location, injected);
    struct snapshot_owner *owner = new_snapshot(gpu, false);
    owner->params.dither_params = NULL;
    owner->params.cone_params = &pl_vision_protanopia;
    float data[24];
    for (int b = 0; b < 2; b++) {
        for (int g = 0; g < 2; g++) {
            for (int r = 0; r < 2; r++) {
                int offset = 3 * (4*b + 2*g + r);
                data[offset] = 0.5f * r;
                data[offset+1] = 0.5f * g;
                data[offset+2] = 0.5f * b;
            }
        }
    }
    const struct pl_custom_lut lut = {
        .signature = 1234, .size = {2, 2, 2}, .data = data,
        .color_in = image.color, .color_out = image.color,
    };
    struct pl_hook replacements[3];
    const struct pl_hook *hooks[3];
    int count = 0;
    const enum pl_hook_stage stages[] = {PL_HOOK_TONE_MAP, PL_HOOK_GAMUT_MAP, PL_HOOK_COLOR_CONVERT};
    for (int i = 0; i < 3; i++) {
        replacements[i] = (struct pl_hook) {
            .stages = stages[i], .input = PL_HOOK_SIG_COLOR,
            .color_input = {.working_space = model, .color = image.color},
            .color_output = {.working_space = model, .color = image.color},
            .hook = identity_color_hook, .describe = identity_color_prepared,
            .execute_prepared = identity_color_prepared, .signature = 2345 + i,
        };
        if (injected & (1 << i))
            hooks[count++] = &replacements[i];
    }
    owner->params.hooks = hooks;
    owner->params.num_hooks = count;
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true));
    REQUIRE(output);
    pl_tex reference = pl_tex_create(gpu, &output->params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output);
    if (location == 0) {
        owner->params.lut = &lut;
        owner->params.lut_type = type;
    } else if (location == 1) {
        image.lut = &lut;
        image.lut_type = type;
    } else {
        target.lut = &lut;
        target.lut_type = type;
    }
    struct pl_frame reference_target = target;
    reference_target.planes[0].texture = reference;
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer ordinary = pl_renderer_create(gpu->log, gpu);
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    pl_renderer_preparation preparation = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot,
                                       &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    wait_prepared(preparation);
    forbid_compile = describing = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(pl_render_image_prepared(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    forbid_compile = false;
    REQUIRE(pl_render_image(ordinary, &image, &reference_target, &owner->params));
    uint8_t expected[64], actual[64];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    for (int i = 0; i < sizeof(actual); i++)
        REQUIRE(abs((int) actual[i] - expected[i]) <= 1);
    pl_renderer_prepare_destroy(&preparation);
    REQUIRE(owner->refs == 1);
    free(owner);
    pl_renderer_destroy(&ordinary);
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &output);
}

static void test_native_color(pl_gpu gpu, struct pl_frame image, int peak)
{
    struct snapshot_owner *owner = new_snapshot(gpu, false);
    struct pl_color_map_params map = pl_color_map_default_params;
    map.tone_mapping_function = &pl_tone_map_bt2446a;
    map.gamut_mapping = peak == 6 ? &pl_gamut_map_perceptual : &pl_gamut_map_desaturate;
    if (peak == 6) {
        owner->params.dynamic_constants = true;
        map.lut3d_size[0] = map.lut3d_size[1] = map.lut3d_size[2] = 16;
    }
    map.mastering_clip = 1.0f;
    map.contrast_recovery = peak >= 3 ? 0.5f : 0.0f;
    if (peak >= 3)
        map.contrast_smoothness = 2.0f;
    struct pl_peak_detect_params detect = pl_peak_detect_default_params;
    detect.allow_delayed = peak == 2 || peak == 4;
    if (detect.allow_delayed)
        detect.smoothing_period = 0.0f;
    owner->params.color_map_params = &map;
    owner->params.peak_detect_params = peak && peak != 5 ? &detect : NULL;
    owner->params.dither_params = NULL;
    image.color = pl_color_space_hdr10;
    image.color.hdr.max_luma = 1000.0f;
    if (peak == 6) {
        image.color.hdr.prim = *pl_raw_primaries_get(PL_COLOR_PRIM_DCI_P3);
    }
    pl_tex source = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .host_writable = true));
    REQUIRE(source);
    image.planes[0].texture = source;
    pl_tex output = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .renderable = true, .host_readable = true));
    pl_tex reference = pl_tex_create(gpu, &output->params);
    REQUIRE(output && reference);
    struct pl_frame target = frame(output), reference_target = frame(reference);
    target.color = reference_target.color = pl_color_space_bt709;
    if (peak == 6) target.color.hdr.max_luma = reference_target.color.hdr.max_luma = 203.0f;
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer ordinary = pl_renderer_create(gpu->log, gpu);
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    pl_renderer_preparation preparation = NULL;
    describing = true;
    enum pl_renderer_prepare_result result = pl_renderer_describe_image(
        renderer, &image, &target, &snapshot, &preparation);
    describing = false;
    if (result != PL_RENDERER_PREPARE_OK)
        fprintf(stderr, "native describe: %s\n", pl_renderer_prepare_error(preparation));
    REQUIRE(result == PL_RENDERER_PREPARE_OK);
    wait_prepared(preparation);
    pl_renderer without_contrast = peak >= 3 ? pl_renderer_create(gpu->log, gpu) : NULL;
    bool contrast_changed_pixels = false;
    for (int i = 0; i < 8; i++) {
        if (peak == 6) {
            image.color.hdr.prim.red.x = 0.68f - 0.001f * i;
            target.color.hdr.max_luma = reference_target.color.hdr.max_luma = 203.0f + 17.0f * i;
        }
        uint8_t pixels[64];
        // Delayed detection may legally consume the current or previous
        // completed measurement. Use stable content after warm-up for exact
        // parity; immediate detection still exercises changing brightness.
        const int level = detect.allow_delayed ? 4 : i;
        for (int j = 0; j < 16; j++) {
            pixels[4*j] = 30 + 25*level + (peak >= 3 ? 10 * (j % 4) : 0);
            pixels[4*j+1] = 50 + 20*level;
            pixels[4*j+2] = 60 + 18*level;
            pixels[4*j+3] = 255;
        }
        REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = source, .ptr = pixels)));
        forbid_compile = describing = true;
        result = pl_renderer_preflight_image(preparation, &image, &target);
        describing = false;
        if (result != PL_RENDERER_PREPARE_OK)
            fprintf(stderr, "native preflight %d: %s\n", i, pl_renderer_prepare_error(preparation));
        REQUIRE(result == PL_RENDERER_PREPARE_OK);
        result = pl_render_image_prepared(preparation, &image, &target);
        forbid_compile = false;
        if (result != PL_RENDERER_PREPARE_OK)
            fprintf(stderr, "native execute %d: %s\n", i, pl_renderer_prepare_error(preparation));
        REQUIRE(result == PL_RENDERER_PREPARE_OK);
        REQUIRE(pl_render_image(ordinary, &image, &reference_target, &owner->params));
        uint8_t actual[64], expected[64];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
        if (!detect.allow_delayed || i > 0)
            REQUIRE_MEMEQ(actual, expected, sizeof(actual));
        if (without_contrast) {
            struct pl_color_map_params disabled_map = map;
            disabled_map.contrast_recovery = 0.0f;
            struct pl_render_params disabled_params = owner->params;
            disabled_params.color_map_params = &disabled_map;
            REQUIRE(pl_render_image(without_contrast, &image, &reference_target,
                                    &disabled_params));
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
            contrast_changed_pixels |= memcmp(actual, expected, sizeof(actual)) != 0;
        }
    }
    if (without_contrast)
        REQUIRE(contrast_changed_pixels);
    pl_renderer_destroy(&without_contrast);
    pl_renderer_prepare_destroy(&preparation);
    free(owner);
    pl_renderer_destroy(&renderer);
    pl_renderer_destroy(&ordinary);
    pl_tex_destroy(gpu, &output);
    pl_tex_destroy(gpu, &reference);
    pl_tex_destroy(gpu, &source);
}

static void test_multiplane_target(pl_gpu gpu, struct pl_frame image)
{
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer ordinary = pl_renderer_create(gpu->log, gpu);
    struct snapshot_owner *owner = new_snapshot(gpu, false);
    owner->params.dither_params = NULL;
    struct pl_frame target = {
        .num_planes = 3, .color = pl_color_space_bt709,
        .repr = { .sys = PL_COLOR_SYSTEM_BT_709, .levels = PL_COLOR_LEVELS_LIMITED,
                  .bits = {8, 8, 0}},
        .crop = {0, 0, 8, 8},
    }, reference = target;
    for (int i = 0; i < 3; i++) {
        struct pl_tex_params params = {
            .w = i ? 4 : 8, .h = i ? 4 : 8,
            .format = pl_find_named_fmt(gpu, "r8"),
            .renderable = true, .host_readable = true, .blit_dst = true,
        };
        target.planes[i] = (struct pl_plane) {
            .texture = pl_tex_create(gpu, &params), .components = 1,
            .component_mapping = {i, -1, -1, -1},
        };
        reference.planes[i] = target.planes[i];
        reference.planes[i].texture = pl_tex_create(gpu, &params);
        REQUIRE(target.planes[i].texture && reference.planes[i].texture);
    }
    struct pl_renderer_snapshot snapshot = {
        .params = &owner->params, .owner = owner, .retain = retain, .release = release,
    };
    pl_renderer_preparation preparation = NULL;
    describing = true;
    REQUIRE(pl_renderer_describe_image(renderer, &image, &target, &snapshot,
                                      &preparation) == PL_RENDERER_PREPARE_OK);
    describing = false;
    wait_prepared(preparation);
    describing = forbid_compile = true;
    REQUIRE(pl_renderer_preflight_image(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    describing = false;
    REQUIRE(pl_render_image_prepared(preparation, &image, &target) == PL_RENDERER_PREPARE_OK);
    forbid_compile = false;
    REQUIRE(pl_render_image(ordinary, &image, &reference, &owner->params));
    for (int i = 0; i < 3; i++) {
        uint8_t actual[64], expected[64];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = target.planes[i].texture, .ptr = actual)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference.planes[i].texture, .ptr = expected)));
        REQUIRE_MEMEQ(actual, expected, i ? 16 : 64);
    }
    struct pl_frame changed = target;
    changed.planes[2].shift_x = 0.5f;
    const int before = gpu_runs;
    REQUIRE(pl_render_image_prepared(preparation, &image, &changed) == PL_RENDERER_PREPARE_NOT_READY);
    REQUIRE(gpu_runs == before);
    pl_renderer_prepare_destroy(&preparation);
    pl_renderer_destroy(&renderer);
    pl_renderer_destroy(&ordinary);
    free(owner);
    for (int i = 0; i < 3; i++) {
        pl_tex_destroy(gpu, &target.planes[i].texture);
        pl_tex_destroy(gpu, &reference.planes[i].texture);
    }
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_vulkan vulkan = pl_vulkan_create(log, pl_vulkan_params(
        .allow_software = true,
        .instance_params = pl_vk_inst_params(.debug = true),
    ));
    if (!vulkan)
        return SKIP;
    pl_gpu gpu = vulkan->gpu;
    struct pl_vk *priv = PL_PRIV(gpu);
    original = priv->impl;
    latch_init(&replacement_compile);
    priv->impl.pass_create_prepared = observe_prepare;
    priv->impl.tex_create = observe_texture;
    priv->impl.buf_create = observe_buffer;
    priv->impl.pass_create = observe_create;
    priv->impl.pass_run = observe_run;
    priv->impl.pass_run_prepared = observe_prepared_run;
    priv->impl.buf_write = observe_write;
    priv->impl.tex_clear_ex = observe_clear;
    priv->impl.tex_upload = observe_upload;
    uint8_t pixels[4 * 4 * 4];
    for (int i = 0; i < sizeof(pixels); i++)
        pixels[i] = (i * 17) % 256;
    pl_tex source = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .initial_data = pixels,
    ));
    REQUIRE(source);
    for (int location = 0; location < 3; location++) {
        for (int type = PL_LUT_NATIVE; type <= PL_LUT_CONVERSION; type++) {
            for (int injected = 0; injected < 2; injected++)
                test_color_lut_prepared(gpu, frame(source), type, location, injected, PL_COLOR_WORKING_JZAZBZ);
        }
    }
    for (int model = 0; model < PL_COLOR_WORKING_COUNT; model++)
        for (int mask = 1; mask < 8; mask++)
            test_color_lut_prepared(gpu, frame(source), PL_LUT_NATIVE, 0, mask, model);
    struct pl_tex_params neighbor_params = source->params;
    neighbor_params.host_writable = true;
    pl_tex previous = pl_tex_create(gpu, &neighbor_params);
    pl_tex next = pl_tex_create(gpu, &neighbor_params);
    REQUIRE(previous && next);
    uint8_t neighbors[sizeof(pixels)];
    for (int i = 0; i < sizeof(pixels); i++)
        neighbors[i] = 255 - pixels[i];
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = previous, .ptr = neighbors)));
    for (int i = 0; i < sizeof(pixels); i++)
        neighbors[i] = pixels[i] / 2;
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = next, .ptr = neighbors)));
    struct pl_frame prev_frame = frame(previous), next_frame = frame(next);
    struct pl_frame interlaced = frame(source);
    interlaced.prev = &prev_frame;
    interlaced.next = &next_frame;
    for (int mode = 15; mode <= 22; mode++)
        test_renderer_sampler(gpu, interlaced, 8, true, true, mode);
    pl_tex_destroy(gpu, &previous);
    pl_tex_destroy(gpu, &next);
    test_multiplane_target(gpu, frame(source));
    test_native_color(gpu, frame(source), false);
    test_native_color(gpu, frame(source), 1);
    test_native_color(gpu, frame(source), 2);
    test_native_color(gpu, frame(source), 3);
    test_native_color(gpu, frame(source), 4);
    test_native_color(gpu, frame(source), 5);
    test_native_color(gpu, frame(source), 6);
    test_renderer(gpu, frame(source), 4, false, false);
    test_renderer(gpu, frame(source), 8, false, false);
    test_renderer(gpu, frame(source), 8, true, false);
    test_renderer(gpu, frame(source), 8, true, true);
    test_renderer_sampler(gpu, frame(source), 8, true, true, 6);
    test_renderer_sampler(gpu, frame(source), 8, true, true, 10);
#ifdef PL_HAVE_LCMS
    for (int icc = 27; icc <= 30; icc++)
        test_renderer_sampler(gpu, frame(source), 8, true, true, icc);
#else
    printf("Skipping ICC preparation: LittleCMS support not built\n");
#endif
    test_renderer_sampler(gpu, frame(source), 8, false, true, 33);
    test_renderer_sampler(gpu, frame(source), 8, true, true, 34);
    test_renderer_sampler(gpu, frame(source), 8, true, true, 35);
    for (int grain = 31; grain <= 32; grain++)
        test_renderer_sampler(gpu, frame(source), 8, true, true, grain);
    for (int overlay = 23; overlay <= 26; overlay++)
        test_renderer_sampler(gpu, frame(source), 8, true, true, overlay);
    for (int border = 11; border <= 14; border++)
        test_renderer_sampler(gpu, frame(source), 8, true, true, border);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 1);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 2);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 3);
    test_renderer_sampler(gpu, frame(source), 4, false, true, 4);
    test_renderer_sampler(gpu, frame(source), 2, false, true, 4);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 5);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 36);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 37);
    test_renderer_sampler(gpu, frame(source), 8, false, true, 38);
    for (int filter = 2; filter < 6; filter++) {
        test_renderer_sampler(gpu, frame(source), 8, false, filter, 36);
        test_renderer_sampler(gpu, frame(source), 8, false, filter, 37);
    }
    uint8_t odd_pixels[5 * 3 * 4];
    for (int i = 0; i < sizeof(odd_pixels); i++) odd_pixels[i] = (i * 19) % 256;
    pl_tex odd = pl_tex_create(gpu, pl_tex_params(
        .w = 5, .h = 3, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .initial_data = odd_pixels));
    REQUIRE(odd);
    struct pl_frame cropped = frame(odd);
    cropped.crop = (pl_rect2df) {0.25, 0.5, 4.75, 2.5};
    cropped.planes[0].flipped = true;
    test_renderer_sampler(gpu, cropped, 8, false, true, 1);
    for (int rotation = PL_ROTATION_90; rotation <= PL_ROTATION_270; rotation++) {
        struct pl_frame rotated = cropped;
        rotated.rotation = rotation;
        test_renderer(gpu, rotated, 8, true, true);
        test_renderer_sampler(gpu, cropped, 8, true, true, 6 + rotation);
    }
    pl_tex_destroy(gpu, &odd);
    struct pl_frame dolby = frame(source);
    dolby.repr.sys = PL_COLOR_SYSTEM_DOLBYVISION;
    dolby.repr.dovi = &dovi_meta;
    dolby.color = pl_color_space_hdr10;
    test_renderer(gpu, dolby, 8, true, true);
    pl_tex enhancement_texture = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = source->params.format, .sampleable = true, .initial_data = pixels));
    REQUIRE(enhancement_texture);
    struct pl_frame enhancement = frame(enhancement_texture);
    struct pl_dovi_metadata fel = dovi_meta;
    fel.nlq_active = true;
    for (int c = 0; c < 3; c++) {
        fel.nlq[c].offset = 0.5f;
        fel.nlq[c].deadzone_slope = 0.25f;
    }
    dolby.repr.dovi = &fel;
    dolby.enhancement_layer = &enhancement;
    test_renderer(gpu, dolby, 8, true, true);
    pl_tex_destroy(gpu, &enhancement_texture);
    test_empty_render(gpu, frame(source));
    test_frame_access(gpu, frame(source), false);
    test_frame_access(gpu, frame(source), true);
    test_replacement_lifecycle(gpu, frame(source));
    test_preparation_policy(gpu, frame(source));
    for (int i = 0; i < pl_num_frame_mixers; i++)
        if (pl_frame_mixers[i].filter)
            test_temporal_mix(gpu, pl_frame_mixers[i].filter, false, false, false);
    test_temporal_mix(gpu, &pl_filter_oversample, false, true, false);
    test_temporal_mix(gpu, &pl_filter_oversample, true, false, false);
    test_temporal_mix(gpu, &pl_filter_mitchell, false, false, false);
    test_temporal_mix(gpu, &pl_filter_oversample, false, false, true);
    test_mix(gpu, frame(source), false, true);
    test_mix(gpu, frame(source), true, true);
    test_mix(gpu, frame(source), false, false);
    test_large_graph(gpu, frame(source));
    test_mpv_prepared(gpu, frame(source));
    test_mpv_corpus(gpu, frame(source));
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
        test_renderer_sampler(gpu, yuv, 8, true, true, 6);
        test_renderer_sampler(gpu, yuv, 8, true, true, 10);
        test_renderer_sampler(gpu, yuv, 8, true, true, 31);
        test_renderer_sampler(gpu, yuv, 8, true, true, 32);
        test_renderer_sampler(gpu, yuv, 4, false, true, 1);
        test_renderer_sampler(gpu, yuv, 8, false, true, 1);
        test_renderer_sampler(gpu, yuv, 8, false, true, 2);
        test_renderer_sampler(gpu, yuv, 8, false, true, 5);
        test_renderer_sampler(gpu, yuv, 8, false, true, 36);
        test_renderer_sampler(gpu, yuv, 8, false, true, 37);
        test_renderer_sampler(gpu, yuv, 8, false, true, 38);
        test_mix(gpu, yuv, false, true);
        if (!planar)
            test_mpv_prepared(gpu, yuv);
    }
    uint8_t uv422_data[16];
    for (int i = 0; i < 8; i++) {
        uv422_data[2*i] = 90 + i * 9;
        uv422_data[2*i+1] = 160 - i * 8;
    }
    pl_tex uv422 = pl_tex_create(gpu, pl_tex_params(
        .w = 2, .h = 4, .format = pl_find_named_fmt(gpu, "rg8"),
        .sampleable = true, .initial_data = uv422_data));
    REQUIRE(uv422);
    struct pl_frame yuv422 = frame(source);
    yuv422.num_planes = 2;
    yuv422.repr = (struct pl_color_repr) {
        .sys = PL_COLOR_SYSTEM_BT_709, .levels = PL_COLOR_LEVELS_LIMITED,
    };
    yuv422.planes[0] = (struct pl_plane) {
        .texture = planes[0], .components = 1, .component_mapping = {0},
    };
    yuv422.planes[1] = (struct pl_plane) {
        .texture = uv422, .components = 2, .component_mapping = {1, 2},
        .shift_x = 0.5,
    };
    test_renderer_sampler(gpu, yuv422, 8, false, true, 5);
    pl_tex_destroy(gpu, &uv422);
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
    latch_destroy(&replacement_compile);
    pl_log_destroy(&log);
}
