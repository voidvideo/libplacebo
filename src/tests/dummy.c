#include "gpu_tests.h"
#include "shaders.h"

#include <libplacebo/dummy.h>
#include <libplacebo/renderer.h>

static void matrix_chain_tests(pl_log log)
{
    const pl_matrix3x3 matrices[] = {
        {{{1, 2, 0}, {0, 1, 0}, {0, 0, 1}}},
        {{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}}},
        {{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}},
    };
    for (int count = 0; count <= PL_ARRAY_SIZE(matrices); count++) {
        pl_shader sh = pl_shader_alloc(log, NULL);
        REQUIRE(sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0));
        for (int i = 0; i < count; i++)
            sh_color_matrix(sh, matrices[i]);
        const struct pl_shader_res *res = pl_shader_finalize(sh);
        REQUIRE(res);
        REQUIRE(res->num_variables == (count ? 1 : 0));
        if (!count) {
            pl_shader_free(&sh);
            continue;
        }
        const float *data = res->variables[0].data;
        // Check the registered column-major shader matrix against sequential
        // transforms of every basis vector, including empty and single chains.
        for (int c = 0; c < 3; c++) {
            float expected[3] = {0};
            expected[c] = 1;
            for (int i = 0; i < count; i++)
                pl_matrix3x3_apply(&matrices[i], expected);
            for (int r = 0; r < 3; r++)
                REQUIRE(fabsf(data[c * 3 + r] - expected[r]) < 1e-6f);
        }
        pl_shader_free(&sh);
    }
}

static pl_cache_obj color_cache_miss(void *priv, uint64_t key)
{
    (*(int *) priv)++;
    return (pl_cache_obj) {.key = key};
}

static void color_transform_tests(pl_log log, pl_gpu gpu)
{
    const pl_transform3x3 transforms[] = {
        {.mat = {{{1, 2, 0}, {0, 1, 0}, {0, 0, 1}}}, .c = {0.1, -0.2, 0.3}},
        {.mat = {{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}}}, .c = {-0.5, 0.2, 0.1}},
    };
    pl_shader sh = pl_shader_alloc(log, NULL);
    REQUIRE(sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0));
    for (int i = 0; i < PL_ARRAY_SIZE(transforms); i++)
        sh_color_transform(sh, &transforms[i]);
    const struct pl_shader_res *res = pl_shader_finalize(sh);
    REQUIRE(res && res->num_variables == 2);
    const float *matrix = res->variables[0].data;
    const float *offset = res->variables[1].data;
    for (int basis = -1; basis < 3; basis++) {
        float expected[3] = {0};
        if (basis >= 0) expected[basis] = 1;
        for (int i = 0; i < PL_ARRAY_SIZE(transforms); i++)
            pl_transform3x3_apply(&transforms[i], expected);
        for (int c = 0; c < 3; c++)
            REQUIRE_FEQ(expected[c], offset[c] + (basis < 0 ? 0 : matrix[basis*3+c]), 1e-6);
    }

    // Arbitrary body code is a barrier; reset discards an unfinished chain.
    pl_shader_reset(sh, NULL);
    REQUIRE(sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0));
    sh_color_transform(sh, &transforms[0]);
    GLSL("color.rgb *= color.rgb;\n");
    sh_color_transform(sh, &transforms[1]);
    res = pl_shader_finalize(sh);
    REQUIRE(res && res->num_variables == 4);
    pl_shader_reset(sh, NULL);
    sh_color_transform(sh, &transforms[0]);
    pl_shader_reset(sh, NULL);
    res = pl_shader_finalize(sh);
    REQUIRE(res && res->num_variables == 0);

    // Existing native decode/conversion/encode calls share one affine chain.
    pl_shader_reset(sh, NULL);
    struct pl_color_repr repr = {
        .sys = PL_COLOR_SYSTEM_BT_709, .levels = PL_COLOR_LEVELS_LIMITED,
        .alpha = PL_ALPHA_INDEPENDENT,
    };
    struct pl_color_space src = pl_color_space_srgb, dst = src;
    src.transfer = dst.transfer = PL_COLOR_TRC_LINEAR;
    dst.primaries = PL_COLOR_PRIM_BT_2020;
    pl_shader_decode_color(sh, &repr, NULL);
    pl_shader_color_convert(sh, &src, &dst);
    repr.sys = PL_COLOR_SYSTEM_BT_601;
    repr.levels = PL_COLOR_LEVELS_LIMITED;
    pl_shader_encode_color(sh, &repr);
    res = pl_shader_finalize(sh);
    REQUIRE(res && res->num_variables == 2);
    int misses = 0;
    pl_cache cache = pl_cache_create(pl_cache_params(
        .get = color_cache_miss, .priv = &misses,
    ));
    pl_gpu_set_cache(gpu, cache);
    for (int i = 0; i < 4; i++) {
        pl_shader_reset(sh, pl_shader_params(.gpu = gpu));
        REQUIRE(sh_require(sh, PL_SHADER_SIG_COLOR, 0, 0));
        pl_transform3x3 changed = transforms[1];
        if (i == 2) changed.c[0] += 0.5f;
        sh_color_transform(sh, &transforms[0]);
        sh_color_transform(sh, &changed);
        res = pl_shader_finalize(sh);
        REQUIRE(res && res->num_variables == 2);
        REQUIRE(misses == (i < 2 ? 1 : 2));
        REQUIRE(pl_cache_objects(cache) == (i < 2 ? 1 : 2));
        float expected[3] = {0};
        pl_transform3x3_apply(&transforms[0], expected);
        pl_transform3x3_apply(&changed, expected);
        const float *cached_offset = res->variables[1].data;
        for (int c = 0; c < 3; c++)
            REQUIRE_FEQ(cached_offset[c], expected[c], 1e-6);
    }
    pl_gpu_set_cache(gpu, NULL);
    pl_cache_destroy(&cache);
    pl_shader_free(&sh);
}

int main()
{
    pl_log log = pl_test_logger();
    matrix_chain_tests(log);
    pl_gpu gpu = pl_gpu_dummy_create(log, NULL);
    color_transform_tests(log, gpu);
    pl_buffer_tests(gpu);
    pl_texture_tests(gpu);

    // Attempt creating a shader and accessing the resulting LUT
    pl_tex dummy = pl_tex_dummy_create(gpu, pl_tex_dummy_params(
        .w = 100,
        .h = 100,
        .format = pl_find_named_fmt(gpu, "rgba8"),
    ));

    struct pl_sample_src src = {
        .tex = dummy,
        .new_w = 1000,
        .new_h = 1000,
    };

    pl_shader_obj lut = NULL;
    struct pl_sample_filter_params filter_params = {
        .filter = pl_filter_ewa_lanczos,
        .lut = &lut,
    };

    pl_shader sh = pl_shader_alloc(log, pl_shader_params( .gpu = gpu ));
    REQUIRE(pl_shader_sample_polar(sh, &src, &filter_params));
    const struct pl_shader_res *res = pl_shader_finalize(sh);
    REQUIRE(res);

    for (int n = 0; n < res->num_descriptors; n++) {
        const struct pl_shader_desc *sd = &res->descriptors[n];
        if (sd->desc.type != PL_DESC_SAMPLED_TEX)
            continue;

        pl_tex tex = sd->binding.object;
        const float *data = (float *) pl_tex_dummy_data(tex);
        if (!data)
            continue; // means this was the `dummy` texture

#ifdef PRINT_LUTS
        for (int i = 0; i < tex->params.w; i++)
            printf("lut[%d] = %f\n", i, data[i]);
#endif
    }

    // Try out generation of the sampler2D interface
    src.tex = NULL;
    src.tex_w = 100;
    src.tex_h = 100;
    src.format = PL_FMT_UNORM;
    src.sampler = PL_SAMPLER_NORMAL;
    src.mode = PL_TEX_SAMPLE_LINEAR;

    pl_shader_reset(sh, pl_shader_params( .gpu = gpu ));
    REQUIRE(pl_shader_sample_polar(sh, &src, &filter_params));
    REQUIRE((res = pl_shader_finalize(sh)));
    REQUIRE_CMP(res->input, ==, PL_SHADER_SIG_SAMPLER, "u");

    pl_shader_free(&sh);
    pl_shader_obj_destroy(&lut);
    pl_tex_destroy(gpu, &dummy);
    pl_gpu_dummy_destroy(&gpu);
    pl_log_destroy(&log);
}
