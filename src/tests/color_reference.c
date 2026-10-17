#include "utils.h"
#include "pl_thread.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/sampling.h>

static pl_shader shader(pl_dispatch dp, pl_gpu gpu, pl_tex source,
                        struct pl_color_space input, struct pl_color_space target, bool dynamic)
{
    pl_shader sh = pl_dispatch_begin(dp);
    pl_shader_reset(sh, pl_shader_params(.gpu = gpu, .dynamic_constants = dynamic));
    REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
    pl_shader_color_map(sh, pl_color_map_params(.gamut_mapping = &pl_gamut_map_clip),
                        input, target, NULL, false);
    return sh;
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
    return prepared;
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(.allow_software = true));
    if (!vk) return SKIP;
    pl_gpu gpu = vk->gpu;
    const float pixels[][4] = {
        {0,0,0,1}, {0.02,0.04,0.06,1}, {0.5,0.5,0.5,1}, {1,1,1,1},
        {1,0,0,1}, {0,1,0,1}, {0,0,1,1}, {-0.1,1.1,0.4,1},
    };
    struct pl_tex_params params = {
        .w = PL_ARRAY_SIZE(pixels), .h = 1, .format = pl_find_named_fmt(gpu, "rgba32f"),
        .sampleable = true, .renderable = true, .host_readable = true,
        .initial_data = pixels,
    };
    pl_tex source = pl_tex_create(gpu, &params);
    params.initial_data = NULL;
    pl_tex output = pl_tex_create(gpu, &params), reference = pl_tex_create(gpu, &params);
    REQUIRE(source && output && reference);
    pl_dispatch dp = pl_dispatch_create(log, gpu);
    const float blacks[] = {0.1f, PL_COLOR_HDR_BLACK, 0.203f};
    const float whites[] = {100, 10, 200, 202.999f, 203, 203.001f, 250, 10000, 100, 203};
    for (int b = 0; b < PL_ARRAY_SIZE(blacks); b++) {
        pl_dispatch_prepared prepared = NULL;
        for (int w = 0; w < PL_ARRAY_SIZE(whites); w++) {
            struct pl_color_space target = pl_color_space_srgb;
            target.hdr.max_luma = whites[w];
            target.hdr.min_luma = blacks[b];
            pl_shader sh = shader(dp, gpu, source, pl_color_space_srgb, target, true);
            pl_dispatch_description desc = NULL;
            REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &sh),
                                                &params, &desc) == PL_DISPATCH_OK);
            if (!prepared) prepared = prepare(&desc);
            else {
                REQUIRE(pl_dispatch_prepared_matches(desc, prepared));
                pl_dispatch_description_destroy(&desc);
            }
            sh = shader(dp, gpu, source, pl_color_space_srgb, target, false);
            REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = reference)));
            sh = shader(dp, gpu, source, pl_color_space_srgb, target, true);
            REQUIRE(pl_dispatch_finish_prepared(dp, prepared,
                pl_dispatch_params(.shader = &sh, .target = output)) == PL_DISPATCH_OK);
            float expected[PL_ARRAY_SIZE(pixels)][4], actual[PL_ARRAY_SIZE(pixels)][4];
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
            for (int i = 0; i < PL_ARRAY_SIZE(pixels); i++)
                for (int c = 0; c < 4; c++)
                    REQUIRE(fabsf(expected[i][c] - actual[i][c]) < 2e-5f);
        }
        pl_dispatch_prepared_destroy(&prepared);
    }
    // HTPC regression: BT.709/BT.1886 with P3 mastering metadata loses that
    // metadata for one frame, then regains it. Both preparation directions must
    // keep the same dispatch and match ordinary rendering of each current frame.
    for (int initial = 0; initial < 2; initial++) {
        pl_dispatch_prepared prepared = NULL;
        for (int step = 0; step < 3; step++) {
            struct pl_color_space input = pl_color_space_bt709;
            struct pl_color_space target = pl_color_space_bt709;
            if ((initial + step) % 2)
                input.hdr.prim = (struct pl_raw_primaries) {
                    .red = {0.68f, 0.32f}, .green = {0.265f, 0.69f},
                    .blue = {0.15f, 0.06f}, .white = {0.3127f, 0.329f},
                };
            pl_shader sh = shader(dp, gpu, source, input, target, true);
            pl_dispatch_description desc = NULL;
            REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &sh),
                                                &params, &desc) == PL_DISPATCH_OK);
            if (!prepared) prepared = prepare(&desc);
            else {
                REQUIRE(pl_dispatch_prepared_matches(desc, prepared));
                pl_dispatch_description_destroy(&desc);
            }
            sh = shader(dp, gpu, source, input, target, false);
            REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = reference)));
            sh = shader(dp, gpu, source, input, target, true);
            REQUIRE(pl_dispatch_finish_prepared(dp, prepared,
                pl_dispatch_params(.shader = &sh, .target = output)) == PL_DISPATCH_OK);
            float expected[PL_ARRAY_SIZE(pixels)][4], actual[PL_ARRAY_SIZE(pixels)][4];
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = reference, .ptr = expected)));
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
            for (int i = 0; i < PL_ARRAY_SIZE(pixels); i++)
                for (int c = 0; c < 4; c++)
                    REQUIRE(fabsf(expected[i][c] - actual[i][c]) < 2e-5f);
        }
        pl_dispatch_prepared_destroy(&prepared);
    }
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &source); pl_tex_destroy(gpu, &output); pl_tex_destroy(gpu, &reference);
    pl_vulkan_destroy(&vk); pl_log_destroy(&log);
}
