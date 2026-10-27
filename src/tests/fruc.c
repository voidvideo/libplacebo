#include "utils.h"
#include "pl_thread.h"
#include "fruc.h"
#include <libplacebo/vulkan.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/frame_queue.h>
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

static void snapshot_ref(void *owner) { (void) owner; }

static struct pl_frame rgb_frame(pl_tex tex)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes = {{.texture = tex, .components = 4, .component_mapping = {0,1,2,3}}},
        .repr = {.sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_PC,
                 .alpha = PL_ALPHA_PREMULTIPLIED},
        .color = {.primaries = PL_COLOR_PRIM_BT_709, .transfer = PL_COLOR_TRC_LINEAR},
    };
}

static bool map_source(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src, struct pl_frame *out)
{
    *out = *(struct pl_frame *) src->frame_data;
    return true;
}

static void renderer_test(pl_gpu gpu)
{
    enum { N = 128, SOURCES = 7 };
    struct pl_hook_texture gray[2], flow[2];
    if (!pl_fruc_describe(gpu, N, N, gray, flow)) return;
    float base[N][N], pixels[N][N][4], actual[N][N][4], expected[N][N][4];
    for (int y = 0; y < N; y++) for (int x = 0; x < N; x++)
        base[y][x] = ((x * 13 + y * 37 + x * y * 17) % 251) / 250.0f;
    pl_tex inputs[SOURCES];
    struct pl_frame frames[SOURCES];
    for (int n = 0; n < SOURCES; n++) {
        for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
            for (int c = 0; c < 3; c++) pixels[y][x][c] = base[y][PL_MAX(0,x-4*n)];
            pixels[y][x][3] = 1.0f;
        }
        inputs[n] = pl_tex_create(gpu, pl_tex_params(.w=N, .h=N,
            .format=pl_find_named_fmt(gpu, "rgba32f"), .sampleable=true, .initial_data=pixels));
        REQUIRE(inputs[n]);
        frames[n] = rgb_frame(inputs[n]);
    }
    pl_tex output = pl_tex_create(gpu, pl_tex_params(.w=N, .h=N,
        .format=pl_find_named_fmt(gpu, "rgba32f"), .renderable=true, .storable=true, .host_readable=true));
    pl_tex reference = pl_tex_create(gpu, &output->params);
    REQUIRE(output && reference);
    struct pl_frame target = rgb_frame(output), native_target = rgb_frame(reference);
    struct pl_render_params params = pl_render_default_params;
    params.frame_mixer = &pl_filter_fruc;
    params.dither_params = NULL;
    struct pl_renderer_snapshot snapshot = {.params=&params, .owner=&params,
        .retain=snapshot_ref, .release=snapshot_ref, .mode=PL_RENDERER_PREPARE_ASYNC};
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    pl_renderer native = pl_renderer_create(gpu->log, gpu);
    const struct pl_frame *pair[2] = {&frames[0], &frames[1]};
    uint64_t signatures[2] = {1,2};
    float timestamps[2] = {-0.4f, 0.6f};
    struct pl_frame_mix mix = {.num_frames=2, .frames=pair, .signatures=signatures,
        .timestamps=timestamps, .vsync_duration=24.0f/60.0f};
    pl_renderer_preparation prep = NULL;
    REQUIRE(pl_renderer_describe_image_mix(renderer, &mix, &target, &snapshot, &prep) == PL_RENDERER_PREPARE_OK);
    REQUIRE(pl_renderer_prepare_submit(prep) == PL_RENDERER_PREPARE_OK);
    enum pl_pass_prepare_state state;
    while ((state = pl_renderer_prepare_poll(prep)) == PL_PASS_PREPARE_PENDING)
        pl_thread_sleep(0.001);
    REQUIRE(state == PL_PASS_PREPARE_READY);
    pl_queue queue = pl_queue_create(gpu);
    struct pl_queue_params settings = {.interpolation_threshold = 2.0f};
    struct pl_queue_plan_cadence cadence = {.source_duration=1.0f/24,
        .output_duration=1.0f/60, .intent=PL_QUEUE_PLAN_INTENT_BRACKET};
    pl_queue_push(queue, &(struct pl_source_frame) {.pts=0, .duration=1.0f/24,
        .frame_data=&frames[0], .map=map_source});
    pl_queue_plan plan = NULL;
    REQUIRE(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
    REQUIRE(pl_queue_plan_get_info(plan)->support_radius == cadence.source_duration);
    REQUIRE(pl_queue_plan_set_target(plan, 1.0/60) == PL_QUEUE_PLAN_PENDING);
    REQUIRE(pl_queue_plan_request(plan) == PL_QUEUE_PLAN_PENDING);
    pl_queue_plan_destroy(&plan);
    for (int n=1; n<SOURCES; n++)
        pl_queue_push(queue, &(struct pl_source_frame) {.pts=n/24.0, .duration=1.0f/24,
            .frame_data=&frames[n], .map=map_source});
    // Fifteen output timestamps at 60 Hz span six source intervals at 24 Hz.
    for (int tick = 0; tick < 15; tick++) {
        int source = tick * 2 / 5;
        float t = (tick * 2 % 5) / 5.0f;
        pair[0] = &frames[source]; pair[1] = &frames[source+1];
        signatures[0] = source+1; signatures[1] = source+2;
        timestamps[0] = -t; timestamps[1] = 1.0f-t;
        REQUIRE(pl_queue_plan_prepare(queue, &settings, &cadence, &plan) == PL_QUEUE_PLAN_PREPARED);
        REQUIRE(pl_queue_plan_set_target(plan, tick/60.0) == PL_QUEUE_PLAN_READY);
        REQUIRE(pl_queue_plan_consume(plan, &mix) == PL_QUEUE_PLAN_READY);
        REQUIRE(mix.num_frames == (t == 0.0f ? 1 : 2));
        if (mix.num_frames == 2)
            REQUIRE(fabsf(-mix.timestamps[0] / (mix.timestamps[1]-mix.timestamps[0]) - t) < 1e-6);
        pl_queue_plan_destroy(&plan);
        REQUIRE(pl_renderer_preflight_image_mix(prep, &mix, &target) == PL_RENDERER_PREPARE_OK);
        REQUIRE(pl_render_image_mix_prepared(prep, &mix, &target) == PL_RENDERER_PREPARE_OK);
        REQUIRE(pl_render_image_mix(native, &mix, &native_target, &params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex=output, .ptr=actual)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex=reference, .ptr=expected)));
        double motion_error = 0;
        for (int y = 24; y < N-24; y++) for (int x = 40; x < N-24; x++) {
            REQUIRE(fabsf(actual[y][x][0] - expected[y][x][0]) < 0.003);
            float position = x - 4*(source+t);
            int left = floorf(position);
            float value = base[y][left]*(1-(position-left)) + base[y][left+1]*(position-left);
            motion_error += fabsf(actual[y][x][0]-value);
        }
        REQUIRE(motion_error / ((N-48)*(N-64)) < 0.05);
    }
    // An absent future frame is pending, never a successful substituted frame.
    mix.num_frames = 1; mix.timestamps = timestamps; timestamps[0] = -0.4f;
    REQUIRE(pl_renderer_preflight_image_mix(prep, &mix, &target) == PL_RENDERER_PREPARE_NOT_READY);
    REQUIRE(!pl_render_image_mix(native, &mix, &native_target, &params));
    pl_queue_destroy(&queue);
    pl_renderer_prepare_destroy(&prep);
    pl_renderer_destroy(&native);
    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &reference); pl_tex_destroy(gpu, &output);
    for (int n=0; n<SOURCES; n++) pl_tex_destroy(gpu, &inputs[n]);
    puts("FRUC native and prepared 24-to-60 rendering passed");
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
    renderer_test(gpu);
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
