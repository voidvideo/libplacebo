#include "vulkan/gpu.h"
#include "shaders.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/shaders/sampling.h>
#include "pass_prepare_helpers.h"

// Guard description/match calls against resource work. LUTs are explicitly
// staged by the preceding real-source build; cold LUT discovery has a separate
// contract and test fixture.
static _Thread_local bool forbid_gpu;
static struct pl_gpu_fns original;
static pl_tex observed_texture(pl_gpu gpu, const struct pl_tex_params *params)
{
    REQUIRE(!forbid_gpu);
    return original.tex_create(gpu, params);
}
static pl_buf observed_buffer(pl_gpu gpu, const struct pl_buf_params *params)
{
    REQUIRE(!forbid_gpu);
    return original.buf_create(gpu, params);
}
static pl_pass observed_pass(pl_gpu gpu, const struct pl_pass_params *params)
{
    REQUIRE(!forbid_gpu);
    return original.pass_create(gpu, params);
}
static void observed_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(!forbid_gpu);
    original.pass_run(gpu, params);
}
static bool observed_prepared_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    REQUIRE(!forbid_gpu);
    return original.pass_run_prepared(gpu, params);
}
static void observed_write(pl_gpu gpu, pl_buf buf, size_t offset, const void *data, size_t size)
{
    REQUIRE(!forbid_gpu);
    original.buf_write(gpu, buf, offset, data, size);
}

enum sampling_kind {
    DIRECT, NEAREST, BILINEAR, BICUBIC, HERMITE, GAUSSIAN, OVERSAMPLE,
    POLAR_GATHER, POLAR_COMPUTE, ORTHO, KIND_COUNT,
};

static pl_shader sample(pl_dispatch dp, enum sampling_kind kind,
                         const struct pl_sample_src *src, pl_shader_obj *lut)
{
    pl_shader sh = pl_dispatch_begin(dp);
    bool ok = false;
    switch (kind) {
    case DIRECT: ok = pl_shader_sample_direct(sh, src); break;
    case NEAREST: ok = pl_shader_sample_nearest(sh, src); break;
    case BILINEAR: ok = pl_shader_sample_bilinear(sh, src); break;
    case BICUBIC: ok = pl_shader_sample_bicubic(sh, src); break;
    case HERMITE: ok = pl_shader_sample_hermite(sh, src); break;
    case GAUSSIAN: ok = pl_shader_sample_gaussian(sh, src); break;
    case OVERSAMPLE: ok = pl_shader_sample_oversample(sh, src, 0.0); break;
    case POLAR_GATHER:
    case POLAR_COMPUTE:
        ok = pl_shader_sample_polar(sh, src, pl_sample_filter_params(
            .filter = pl_filter_ewa_lanczos, .lut = lut,
            .no_compute = kind == POLAR_GATHER,
        ));
        break;
    case ORTHO:
        ok = pl_shader_sample_ortho2(sh, src, pl_sample_filter_params(
            .filter = pl_filter_catmull_rom, .lut = lut,
        ));
        break;
    case KIND_COUNT: abort();
    }
    REQUIRE(ok);
    return sh;
}

static pl_dispatch_description describe(pl_dispatch dp, pl_shader sh,
                                         const struct pl_tex_params *target)
{
    pl_dispatch_description desc = NULL;
    REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &sh),
                                        target, &desc) == PL_DISPATCH_OK);
    REQUIRE(!sh && desc);
    return desc;
}

static pl_dispatch_prepared prepare(pl_dispatch_description *desc)
{
    pl_dispatch_preparation request = NULL;
    REQUIRE(pl_dispatch_prepare_submit(desc, &request) == PL_PASS_PREPARE_ACCEPTED);
    pl_clock_t start = pl_clock_now();
    while (pl_dispatch_prepare_poll(request) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    pl_dispatch_prepared prepared = NULL;
    REQUIRE(pl_dispatch_prepare_take(&request, &prepared) == PL_PASS_PREPARE_READY);
    REQUIRE(!request && prepared);
    return prepared;
}

static void test_sampling(pl_gpu gpu, pl_tex source, enum sampling_kind kind)
{
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader_obj lut = NULL;
    struct pl_tex_params target = source->params;
    target.initial_data = NULL;
    target.w = 8;
    target.h = kind == ORTHO ? 4 : 8;
    target.storable = kind == POLAR_COMPUTE;
    target.host_readable = true;
    pl_tex output = pl_tex_create(gpu, &target);
    pl_tex reference = pl_tex_create(gpu, &target);
    REQUIRE(output && reference);
    struct pl_sample_src real = {
        .tex = source, .new_w = target.w, .new_h = target.h,
        .rect = {4,0,0,4}, // flipped source exercises coordinate normalization
    };
    struct pl_sample_src metadata = real;
    metadata.tex = NULL;
    metadata.texture = &source->params;
    metadata.sampler_type = source->sampler_type;

    // Actual source generation stages any sampler LUT, then both descriptions
    // must normalize to the identical complete pass.
    pl_dispatch_description actual = describe(dp, sample(dp, kind, &real, &lut), &target);
    forbid_gpu = true;
    pl_dispatch_description desc = describe(dp, sample(dp, kind, &metadata, &lut), &target);
    const struct pl_pass_params *a = pl_dispatch_description_params(actual);
    const struct pl_pass_params *b = pl_dispatch_description_params(desc);
    REQUIRE(a->type == b->type);
    REQUIRE(!strcmp(a->glsl_shader, b->glsl_shader));
    if (a->vertex_shader)
        REQUIRE(b->vertex_shader && !strcmp(a->vertex_shader, b->vertex_shader));
    REQUIRE(a->num_descriptors == b->num_descriptors && a->num_constants == b->num_constants);
    if (kind == POLAR_GATHER)
        REQUIRE(strstr(b->glsl_shader, "textureGather"));
    if (kind == POLAR_COMPUTE)
        REQUIRE(b->type == PL_PASS_COMPUTE);
    forbid_gpu = false;
    pl_dispatch_prepared prepared = prepare(&desc);
    forbid_gpu = true;
    REQUIRE(pl_dispatch_prepared_matches(actual, prepared));
    REQUIRE(pl_dispatch_prepared_matches(actual, prepared)); // non-consuming
    REQUIRE(!pl_dispatch_prepared_matches(actual, NULL));
    REQUIRE(!pl_dispatch_prepared_matches(NULL, prepared));
    forbid_gpu = false;

    pl_shader sh = sample(dp, kind, &real, &lut);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = reference)));
    sh = sample(dp, kind, &real, &lut);
    REQUIRE(pl_dispatch_finish_prepared(dp, prepared,
        pl_dispatch_params(.shader = &sh, .target = output)) == PL_DISPATCH_OK);
    uint8_t expected[8*8*4], result[8*8*4];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = result)));
    REQUIRE_MEMEQ(result, expected, target.w * target.h * 4);

    // Same normalized shader from another dispatcher is not owned by this one.
    pl_dispatch other = pl_dispatch_create(gpu->log, gpu);
    pl_dispatch_description foreign = describe(other, sample(other, kind, &metadata, &lut), &target);
    forbid_gpu = true;
    REQUIRE(!pl_dispatch_prepared_matches(foreign, prepared));
    forbid_gpu = false;
    pl_dispatch_description_destroy(&foreign);
    pl_dispatch_destroy(&other);
    // Different sampler type changes source declaration and coordinates.
    metadata.sampler_type = PL_SAMPLER_RECT;
    forbid_gpu = true;
    pl_dispatch_description changed = describe(dp, sample(dp, kind, &metadata, &lut), &target);
    REQUIRE(!pl_dispatch_prepared_matches(changed, prepared));
    forbid_gpu = false;
    pl_dispatch_description_destroy(&changed);
    pl_dispatch_description_destroy(&actual);
    pl_dispatch_prepared_destroy(&prepared);
    pl_dispatch_destroy(&dp);
    pl_shader_obj_destroy(&lut);
    pl_tex_destroy(gpu, &output);
    pl_tex_destroy(gpu, &reference);
}

static void test_external_and_precedence(pl_gpu gpu, pl_tex source)
{
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(
        .tex_w = 4, .tex_h = 4, .format = PL_FMT_UNORM,
        .sampler = PL_SAMPLER_NORMAL, .mode = PL_TEX_SAMPLE_NEAREST,
    )));
    const struct pl_shader_res *res = pl_shader_finalize(sh);
    REQUIRE(res && res->input == PL_SHADER_SIG_SAMPLER && res->num_descriptors == 0);
    pl_dispatch_abort(dp, &sh);
    // Real resources take precedence over invalid, unused metadata.
    sh = pl_dispatch_begin(dp);
    struct pl_tex_params invalid = {0};
    REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(
        .tex = source, .texture = &invalid, .sampler_type = PL_SAMPLER_TYPE_COUNT,
    )));
    pl_dispatch_abort(dp, &sh);
    sh = pl_dispatch_begin(dp);
    REQUIRE(!pl_shader_sample_nearest(sh, pl_sample_src(.texture = &invalid)));
    REQUIRE(pl_shader_is_failed(sh));
    pl_dispatch_abort(dp, &sh);
    pl_dispatch_destroy(&dp);
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(.allow_software = true));
    if (!vk)
        return SKIP;
    pl_gpu gpu = vk->gpu;
    struct pl_vk *priv = PL_PRIV(gpu);
    original = priv->impl;
    priv->impl.tex_create = observed_texture;
    priv->impl.buf_create = observed_buffer;
    priv->impl.pass_create = observed_pass;
    priv->impl.pass_run = observed_run;
    priv->impl.pass_run_prepared = observed_prepared_run;
    priv->impl.buf_write = observed_write;
    uint8_t pixels[4*4*4];
    for (int i = 0; i < sizeof(pixels); i++)
        pixels[i] = i * 29;
    pl_tex source = pl_tex_create(gpu, pl_tex_params(
        .w = 4, .h = 4, .format = pl_find_named_fmt(gpu, "rgba8"),
        .sampleable = true, .renderable = true, .initial_data = pixels,
    ));
    REQUIRE(source);
    for (enum sampling_kind kind = 0; kind < KIND_COUNT; kind++)
        test_sampling(gpu, source, kind);
    test_external_and_precedence(gpu, source);
    pl_tex_destroy(gpu, &source);
    pl_pass_prepare_uninit(gpu);
    pl_vulkan_destroy(&vk);
    pl_log_destroy(&log);
}
