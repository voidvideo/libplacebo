#include "utils.h"
#include "pl_thread.h"
#include "fruc.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/fruc.h>

static struct pl_hook_texture texture(pl_tex tex)
{
    return (struct pl_hook_texture) {
        .texture = tex, .params = tex->params, .sampler_type = tex->sampler_type,
    };
}

static void optical_flow_test(pl_gpu gpu, pl_dispatch dp)
{
    enum { N = 128 };
    struct pl_hook_texture gray[2], flow[2];
    if (!pl_fruc_describe(gpu, N, N, gray, flow)) {
        printf("FRUC hardware execution unavailable on this device\n");
        return;
    }
    struct pl_fruc *engine = pl_fruc_create(gpu, N, N);
    REQUIRE(engine);
    float pixels[N][N][4], output[N][N][4];
    for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        float value = ((x * 13 + y * 37 + x * y * 17) % 251) / 250.0f;
        for (int c = 0; c < 3; c++) pixels[y][x][c] = value;
        pixels[y][x][3] = 1.0f;
    }
    struct pl_fruc_source source = {.width = N, .height = N, .t = 0.5f};
    pl_tex inputs[2];
    for (int i = 0; i < 2; i++) {
        inputs[i] = pl_tex_create(gpu, pl_tex_params(
            .w = N, .h = N, .format = pl_find_named_fmt(gpu, "rgba32f"),
            .sampleable = true, .host_writable = true, .initial_data = pixels));
        REQUIRE(inputs[i]);
        source.frames[i] = texture(inputs[i]);
        source.flow[i] = texture(engine->flow[i]);
    }
    pl_tex target = pl_tex_create(gpu, pl_tex_params(
        .w = N, .h = N, .format = pl_find_named_fmt(gpu, "rgba32f"),
        .storable = true, .renderable = true, .host_readable = true));
    REQUIRE(target);
    for (int iteration = 0; iteration < 4; iteration++) {
        for (int i = 0; i < 2; i++) {
            pl_shader sh = pl_dispatch_begin(dp);
            pl_shader_fruc_grayscale(sh, &source.frames[i], N, N);
            REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = engine->gray[i])));
        }
        REQUIRE(pl_fruc_execute(gpu, engine));
        pl_shader sh = pl_dispatch_begin(dp);
        pl_shader_fruc_interpolate(sh, &source);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = target)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = target, .ptr = output)));
        double error = 0;
        for (int y = 8; y < N-8; y++) for (int x = 8; x < N-8; x++)
            error += fabsf(output[y][x][0] - pixels[y][x][0]);
        REQUIRE(error / ((N-16)*(N-16)) < 0.01);
    }
    // A four-pixel translation must produce the two-pixel intermediate image.
    float shifted[N][N][4];
    for (int y = 0; y < N; y++) for (int x = 0; x < N; x++)
        memcpy(shifted[y][x], pixels[y][PL_MAX(0, x-4)], sizeof(shifted[y][x]));
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = inputs[1], .ptr = shifted)));
    for (int i = 0; i < 2; i++) {
        pl_shader sh = pl_dispatch_begin(dp);
        pl_shader_fruc_grayscale(sh, &source.frames[i], N, N);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = engine->gray[i])));
    }
    REQUIRE(pl_fruc_execute(gpu, engine));
    pl_shader sh = pl_dispatch_begin(dp);
    pl_shader_fruc_interpolate(sh, &source);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = target)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = target, .ptr = output)));
    double motion_error = 0;
    for (int y = 16; y < N-16; y++) for (int x = 16; x < N-16; x++)
        motion_error += fabsf(output[y][x][0] - pixels[y][x-2][0]);
    REQUIRE(motion_error / ((N-32)*(N-32)) < 0.04);

    // Teardown must also be safe with optical-flow work still queued.
    REQUIRE(pl_fruc_execute(gpu, engine));
    pl_fruc_destroy(gpu, &engine);
    for (int i = 0; i < 2; i++) pl_tex_destroy(gpu, &inputs[i]);
    pl_tex_destroy(gpu, &target);
    pl_gpu_finish(gpu);
    printf("FRUC hardware execution passed\n");
}

int main(void)
{
    pl_log log = pl_test_logger();
    pl_vk_inst instance = pl_vk_inst_create(log, pl_vk_inst_params(.debug = true));
    REQUIRE(instance);
    pl_vulkan vk = pl_vulkan_create_with_instance_extensions(log, pl_vulkan_params(
        .instance = instance->instance, .get_proc_addr = instance->get_proc_addr,
        .allow_software = true), instance->extensions, instance->num_extensions);
    if (!vk) {
        pl_vk_inst_destroy(&instance);
        pl_log_destroy(&log);
        return SKIP;
    }
    pl_gpu gpu = vk->gpu;
    pl_dispatch dp = pl_dispatch_create(log, gpu);
    optical_flow_test(gpu, dp);
    enum { W = 64, H = 8, PAD = 4 };
    float pixels[2][H][W+PAD][4], result[H][W][4];
    int16_t vectors[2][H][W][2];
    for (int f = 0; f < 2; f++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W+PAD; x++) {
                pixels[f][y][x][0] = (float)(x - f * 4) / W;
                pixels[f][y][x][1] = 0.25f;
                pixels[f][y][x][2] = 0.75f;
                pixels[f][y][x][3] = 1.0f;
                if (x < W) {
                    vectors[f][y][x][0] = (f ? -4 : 4) * 32;
                    vectors[f][y][x][1] = 0;
                }
            }
        }
    }
    pl_fmt rgb = pl_find_named_fmt(gpu, "rgba32f");
    pl_fmt flowfmt = pl_find_fmt(gpu, PL_FMT_SINT, 2, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    REQUIRE(rgb && flowfmt);
    struct pl_fruc_source src = {.width = W, .height = H};
    pl_tex frames[2], flows[2];
    for (int i = 0; i < 2; i++) {
        frames[i] = pl_tex_create(gpu, pl_tex_params(
            .w = W+PAD, .h = H, .format = rgb,
            .sampleable = true, .host_writable = true, .initial_data = pixels[i]));
        flows[i] = pl_tex_create(gpu, pl_tex_params(
            .w = W, .h = H, .format = flowfmt,
            .sampleable = true, .host_writable = true, .initial_data = vectors[i]));
        REQUIRE(frames[i] && flows[i]);
        src.frames[i] = texture(frames[i]);
        src.flow[i] = texture(flows[i]);
    }
    pl_tex out = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = rgb, .storable = true,
        .renderable = true, .host_readable = true));
    REQUIRE(out);
    struct pl_fruc_source metadata = src;
    for (int i = 0; i < 2; i++) {
        metadata.frames[i].texture = NULL;
        metadata.flow[i].texture = NULL;
    }
    pl_shader described = pl_dispatch_begin(dp);
    pl_shader_fruc_interpolate(described, &metadata);
    pl_dispatch_description description = NULL;
    REQUIRE(pl_dispatch_describe_finish(dp, pl_dispatch_params(.shader = &described),
        &out->params, &description) == PL_DISPATCH_OK);
    pl_dispatch_preparation request = NULL;
    REQUIRE(pl_dispatch_prepare_submit(&description, &request) == PL_PASS_PREPARE_ACCEPTED);
    pl_clock_t start = pl_clock_now();
    while (pl_dispatch_prepare_poll(request) == PL_PASS_PREPARE_PENDING) {
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 10.0);
        pl_thread_sleep(0.001);
    }
    pl_dispatch_prepared prepared = NULL;
    REQUIRE(pl_dispatch_prepare_take(&request, &prepared) == PL_PASS_PREPARE_READY);
    for (int step = 0; step <= 4; step++) {
        src.t = step * 0.25f;
        pl_shader sh = pl_dispatch_begin(dp);
        pl_shader_fruc_interpolate(sh, &src);
        REQUIRE(pl_dispatch_finish_prepared(dp, prepared,
            pl_dispatch_params(.shader = &sh, .target = out)) == PL_DISPATCH_OK);
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = out, .ptr = result)));
        // Constant translation of a ramp reconstructs its intermediate position.
        // The six damped iterations leave at most 1/64 of the displacement.
        for (int y = 0; y < H; y++) for (int x = 5; x < W-5; x++) {
            REQUIRE(fabsf(result[y][x][0] - (x - src.t * 4) / W) < 0.0011f);
            REQUIRE(fabsf(result[y][x][1] - 0.25f) < 0.00001f);
            REQUIRE(fabsf(result[y][x][3] - 1.0f) < 0.00001f);
        }
    }
    pl_dispatch_prepared_destroy(&prepared);
    // Identical textured images with bogus large flow must preserve upstream's
    // static-region guard, rather than warp the otherwise stationary image.
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = frames[1], .ptr = pixels[0])));
    src.t = 0.5f;
    for (int i = 0; i < 2; i++) {
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
            vectors[i][y][x][0] = (i ? -48 : 48) * 32;
        REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = flows[i], .ptr = vectors[i])));
    }
    pl_shader guard = pl_dispatch_begin(dp);
    pl_shader_fruc_interpolate(guard, &src);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &guard, .target = out)));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = out, .ptr = result)));
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        REQUIRE(fabsf(result[y][x][0] - pixels[0][y][x][0]) < 0.00001f);

    pl_tex gray = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = pl_find_named_fmt(gpu, "r8"),
        .storable = true, .renderable = true, .host_readable = true));
    REQUIRE(gray);
    pl_shader sh = pl_dispatch_begin(dp);
    pl_shader_fruc_grayscale(sh, &src.frames[0], W, H);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = gray)));
    uint8_t proxy[H][W];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = gray, .ptr = proxy)));
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        float expected = (0.2126f * x / W + 0.7152f * 0.25f + 0.0722f * 0.75f) * 255;
        // Allow one UNORM quantization step across GPU conversion rules.
        REQUIRE(fabsf(proxy[y][x] - expected) <= 1.0f);
    }
    pl_tex_destroy(gpu, &gray);
    pl_tex_destroy(gpu, &out);
    for (int i = 0; i < 2; i++) {
        pl_tex_destroy(gpu, &frames[i]);
        pl_tex_destroy(gpu, &flows[i]);
    }
    pl_dispatch_destroy(&dp);
    pl_vulkan_destroy(&vk);
    pl_vk_inst_destroy(&instance);
    pl_log_destroy(&log);
}
