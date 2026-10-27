#include "utils.h"
#include <time.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>
#include "frame_mixer_blend.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
static void pause_poll(void)
{
#ifdef _WIN32
    Sleep(1);
#else
    struct timespec duration = {.tv_nsec = 1000000};
    nanosleep(&duration, NULL);
#endif
}
struct owner { int refs; };
static void retain(void *p) { ((struct owner *)p)->refs++; }
static void release(void *p) { ((struct owner *)p)->refs--; }
struct render_counts { unsigned sources, output; };
static void render_info(void *priv, const struct pl_render_info *info)
{
    struct render_counts *counts = priv;
    if (info->stage == PL_RENDER_STAGE_FRAME) counts->sources++;
    if (info->stage == PL_RENDER_STAGE_BLEND) counts->output++;
}
struct source { int acquired, released; };
static bool acquire(pl_gpu gpu, struct pl_frame *frame)
{
    ((struct source *)frame->user_data)->acquired++;
    return true;
}
static void unmap(pl_gpu gpu, struct pl_frame *frame)
{
    ((struct source *)frame->user_data)->released++;
}
static struct pl_frame frame(pl_tex texture)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes = {{.texture = texture, .components = 4, .component_mapping = {0,1,2,3}}},
        .repr = {.sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_PC, .alpha = PL_ALPHA_INDEPENDENT},
        .color = pl_color_space_srgb,
    };
}
static float decode(float x) { return x <= .04045f ? x / 12.92f : powf((x + .055f)/1.055f, 2.4f); }
static float encode(float x) { return x <= .0031308f ? x * 12.92f : 1.055f * powf(x, 1/2.4f) - .055f; }
static void prepare(pl_renderer_mixer m)
{
    REQUIRE(pl_renderer_mixer_stage(m) == PL_RENDERER_PREPARE_OK);
    pl_clock_t start = pl_clock_now();
    enum pl_pass_prepare_state s;
    do {
        s = pl_renderer_mixer_prepare_poll(m);
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 20);
        if (s == PL_PASS_PREPARE_PENDING) pause_poll();
    } while (s == PL_PASS_PREPARE_PENDING);
    REQUIRE(s == PL_PASS_PREPARE_READY);
}
static enum pl_mixer_status complete(pl_renderer_mixer m)
{
    pl_clock_t start = pl_clock_now();
    enum pl_mixer_status s;
    do {
        s = pl_renderer_mixer_poll(m);
        REQUIRE(pl_clock_diff(pl_clock_now(), start) < 20);
        if (s == PL_MIXER_PENDING) pause_poll();
    } while (s == PL_MIXER_PENDING);
    return s;
}
// Compare LUT placement against ordinary renderer calibration. Hardware LUT
// interpolation can quantize even float targets, so analytic LUT arithmetic is
// not an exact reference. Gamma and blending still have independent CPU oracles.
static void calibrate(pl_gpu gpu, const struct pl_custom_lut *lut, float rgb[3])
{
    float rgba[4] = {rgb[0], rgb[1], rgb[2], 1};
    struct pl_tex_params tp = {.w=1, .h=1, .format=pl_find_named_fmt(gpu,"rgba32f"),
        .sampleable=true, .renderable=true, .host_readable=true, .initial_data=rgba};
    pl_tex source = pl_tex_create(gpu, &tp); tp.initial_data = NULL;
    pl_tex output = pl_tex_create(gpu, &tp); REQUIRE(source && output);
    struct pl_frame input = frame(source), target = frame(output);
    target.lut = lut; target.lut_type = PL_LUT_NORMALIZED;
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    REQUIRE(pl_render_image(rr, &input, &target, &pl_render_fast_params));
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex=output, .ptr=rgba)));
    memcpy(rgb, rgba, 3*sizeof(float));
    pl_renderer_destroy(&rr); pl_tex_destroy(gpu, &source); pl_tex_destroy(gpu, &output);
}
static void pixels(pl_gpu gpu, pl_tex output, float a, float b, float t,
                   bool linear, float gain, const struct pl_custom_lut *lut)
{
    int count = output->params.w * output->params.h;
    float *actual = malloc(count * 4 * sizeof(float)); REQUIRE(actual);
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = output, .ptr = actual)));
    float x[3], y[3], expected[3];
    for (int c=0; c<3; c++) { x[c]=a+c*.03f; y[c]=b+c*.03f; }
    if (lut && !linear) { calibrate(gpu, lut, x); calibrate(gpu, lut, y); }
    for (int c=0; c<3; c++)
        // Default SDR metadata has a 1000:1 contrast ratio. The cache holds
        // absolute linear light including black; gain scales that black too.
        expected[c] = linear ? encode(((1-t)*decode(x[c]) + t*decode(y[c]))*gain +
                                      (gain - 1.0f) * .001f / .999f)
                             : ((1-t)*x[c]+t*y[c])*gain;
    if (lut && linear) calibrate(gpu, lut, expected);
    for (int p=0; p<count; p++) {
        for (int c=0; c<3; c++) {
            if (fabsf(actual[p*4+c] - expected[c]) >= 4e-4f)
                fprintf(stderr, "pixel %d channel %d actual %.8f expected %.8f domain %d a=%f b=%f t=%f gain=%f lut=%d size=%d\n",
                        p,c,actual[p*4+c],expected[c],linear,a,b,t,gain,!!lut,output->params.w);
            REQUIRE(fabsf(actual[p*4+c] - expected[c]) < 4e-4f);
        }
        REQUIRE(fabsf(actual[p*4+3]-1) < 1e-5f);
    }
    free(actual);
}
static void run(pl_gpu gpu, pl_mixer_registry registry, bool linear, bool lut)
{
    struct source counters[3] = {0};
    pl_tex textures[3]; struct pl_frame sources[3];
    struct pl_tex_params tp = {.w = 4, .h = 4, .format = pl_find_named_fmt(gpu,"rgba32f"),
        .sampleable = true, .renderable = true, .host_readable = true, .blit_dst = true};
    for (int n = 0; n < 3; n++) {
        float data[16][4];
        for (int p = 0; p < 16; p++) {
            for (int c = 0; c < 3; c++) data[p][c] = .15f + n * .2f + c * .03f;
            data[p][3] = 1;
        }
        tp.initial_data = data;
        textures[n] = pl_tex_create(gpu, &tp); REQUIRE(textures[n]);
        sources[n] = frame(textures[n]);
        sources[n].user_data = &counters[n];
        sources[n].acquire = acquire; sources[n].release = unmap;
    }
    tp.initial_data = NULL;
    pl_tex output = pl_tex_create(gpu, &tp); REQUIRE(output);
    struct pl_frame target = frame(output);
    struct source destination = {0};
    target.user_data = &destination; target.acquire = acquire; target.release = unmap;
    pl_tex_clear(gpu, output, (float[4]) {.7f,.2f,.9f,1});
    float table[24];
    for (int b = 0; b < 2; b++) for (int g = 0; g < 2; g++) for (int r = 0; r < 2; r++) {
        int i = 3*(4*b+2*g+r); table[i] = r*g; table[i+1] = g*b; table[i+2] = b*r;
    }
    struct pl_custom_lut calibration = {.signature = 55234, .size = {2,2,2},
        .data = table, .interpolation = PL_LUT_LINEAR};
    if (lut) { target.lut = &calibration; target.lut_type = PL_LUT_NORMALIZED; }
    struct pl_render_params params = pl_render_fast_params;
    struct render_counts counts = {0};
    params.info_callback = render_info; params.info_priv = &counts;
    params.frame_mixer_input = linear ? PL_FRAME_MIXER_INPUT_LINEAR : PL_FRAME_MIXER_INPUT_COMPLETED;
    struct owner owner = {0};
    struct pl_renderer_snapshot snapshot = {.params = &params, .owner = &owner,
        .retain = retain, .release = release, .mode = PL_RENDERER_PREPARE_ASYNC};
    struct blend_stats stats = {0};
    struct blend_config settings = {.domain = linear ? PL_MIXER_COLOR_LINEAR : PL_MIXER_COLOR_COMPLETED,
                                    .gain = 1, .stats = &stats};
    struct pl_mixer_config config = {.revision = 1, .data = &settings, .size = sizeof(settings)};
    pl_mixer mixer = NULL;
    REQUIRE(pl_mixer_create(registry, "test.vulkan-blend", gpu, &config, &mixer) == PL_MIXER_OK);
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    const struct pl_frame *frames[2] = {&sources[0], &sources[1]};
    uint64_t signatures[2] = {10, 11};
    float relative[2] = {-.5f, .5f};
    struct pl_frame_mix mix = {.num_frames = 2, .frames = frames, .signatures = signatures,
                              .timestamps = relative, .vsync_duration = 1};
    double times[2] = {0, 1};
    pl_renderer_mixer m = NULL;
    REQUIRE(pl_renderer_mixer_describe(rr, mixer, &mix, &target, &snapshot, &m) == PL_RENDERER_PREPARE_OK);
    REQUIRE(pl_mixer_destroy(&mixer) == PL_MIXER_BUSY); // retained even before any jobs
    prepare(m);
    REQUIRE(counters[0].acquired == 0 && counters[1].acquired == 0);
    uint64_t content[2] = {0}, allocation[2] = {0};
    unsigned source_dispatches = 0;
    for (int n = 1; n <= 3; n++) {
        double t = n * .25;
        float before[64], after[64];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex=output, .ptr=before)));
        REQUIRE(pl_renderer_mixer_begin(m, &mix, times, t, 1, &target) == PL_MIXER_OK);
        REQUIRE(counters[0].acquired == 1 && counters[1].acquired == 1);
        if (n == 1) { source_dispatches = counts.sources; REQUIRE(source_dispatches > 0); }
        else REQUIRE(counts.sources == source_dispatches);
        REQUIRE(destination.acquired == n - 1 && destination.released == n - 1);
        stats.hold = true;
        REQUIRE(pl_renderer_mixer_submit(m) == PL_MIXER_PENDING);
        REQUIRE(pl_renderer_mixer_submit(m) == PL_MIXER_INVALID);
        REQUIRE(pl_renderer_mixer_poll(m) == PL_MIXER_PENDING);
        REQUIRE(pl_renderer_mixer_finish(m, &target) == PL_MIXER_PENDING);
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex=output, .ptr=after)));
        REQUIRE(!memcmp(before, after, sizeof(before)));
        REQUIRE(pl_renderer_mixer_begin(m, &mix, times, t, 1, &target) == PL_MIXER_BUSY);
        REQUIRE(pl_renderer_mixer_discard(m) == PL_MIXER_BUSY);
        REQUIRE(pl_renderer_mixer_destroy(&m) == PL_MIXER_BUSY);
        if (n == 1) {
            memcpy(content, stats.content, sizeof(content));
            memcpy(allocation, stats.allocation, sizeof(allocation));
        } else {
            REQUIRE(!memcmp(content, stats.content, sizeof(content)));
            REQUIRE(!memcmp(allocation, stats.allocation, sizeof(allocation)));
        }
        stats.hold = false;
        REQUIRE(complete(m) == PL_MIXER_OK);
        REQUIRE(pl_renderer_mixer_finish(m, &target) == PL_MIXER_OK);
        pixels(gpu, output, .15f, .35f, t, linear, 1, target.lut);
    }
    // A sliding source window reuses the shared frame, then epoch invalidates it.
    frames[0] = &sources[1]; frames[1] = &sources[2];
    signatures[0] = 11; signatures[1] = 12; times[0] = 1; times[1] = 2;
    REQUIRE(pl_renderer_mixer_begin(m, &mix, times, 1.5, 1, &target) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_submit(m) == PL_MIXER_PENDING);
    REQUIRE(stats.content[0] == content[1]);
    REQUIRE(counters[1].acquired == 1 && counters[2].acquired == 1);
    REQUIRE(complete(m) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_finish(m, &target) == PL_MIXER_OK);
    pixels(gpu, output, .35f, .55f, .5f, linear, 1, target.lut);
    uint64_t previous = stats.content[0];
    REQUIRE(pl_renderer_mixer_begin(m, &mix, times, 1.5, 2, &target) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_submit(m) == PL_MIXER_PENDING);
    REQUIRE(stats.epoch == 2 && stats.content[0] != previous);
    REQUIRE(counters[1].acquired == 2 && counters[2].acquired == 2);
    stats.fail = true;
    REQUIRE(complete(m) == PL_MIXER_FAILED);
    REQUIRE(pl_renderer_mixer_finish(m, &target) == PL_MIXER_FAILED);
    REQUIRE(pl_renderer_mixer_discard(m) == PL_MIXER_OK);
    stats.fail = false;
    // Explicitly discarded failed output did not replace the prior target pixels.
    pixels(gpu, output, .35f, .55f, .5f, linear, 1, target.lut);
    // A new immutable mixer configuration can complete while old work is held.
    REQUIRE(pl_renderer_mixer_begin(m, &mix, times, 1.5, 2, &target) == PL_MIXER_OK);
    stats.hold = true;
    REQUIRE(pl_renderer_mixer_submit(m) == PL_MIXER_PENDING);
    struct blend_stats next_stats = {0};
    struct blend_config next_settings = settings;
    next_settings.stats = &next_stats; next_settings.gain = .5f;
    struct pl_mixer_config next_config = {.revision = 2, .data = &next_settings, .size = sizeof(next_settings)};
    pl_mixer next_mixer = NULL; pl_renderer_mixer next = NULL;
    REQUIRE(pl_mixer_create(registry, "test.vulkan-blend", gpu, &next_config, &next_mixer) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_describe(rr, next_mixer, &mix, &target, &snapshot, &next) == PL_RENDERER_PREPARE_OK);
    prepare(next);
    REQUIRE(pl_renderer_mixer_begin(next, &mix, times, 1.5, 2, &target) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_submit(next) == PL_MIXER_PENDING);
    REQUIRE(complete(next) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_finish(next, &target) == PL_MIXER_OK);
    pixels(gpu, output, .35f, .55f, .5f, linear, .5f, target.lut);
    REQUIRE(pl_renderer_mixer_finish(m, &target) == PL_MIXER_PENDING);
    pixels(gpu, output, .35f, .55f, .5f, linear, .5f, target.lut);
    stats.hold = false;
    REQUIRE(complete(m) == PL_MIXER_OK);
    // A changed finishing configuration cannot consume an older request.
    struct pl_frame changed = target;
    changed.color.transfer = PL_COLOR_TRC_GAMMA22;
    REQUIRE(pl_renderer_mixer_finish(m, &changed) == PL_MIXER_INVALID);
    REQUIRE(pl_renderer_mixer_finish(m, &target) == PL_MIXER_OK);
    pixels(gpu, output, .35f, .55f, .5f, linear, 1, target.lut);
    REQUIRE(pl_renderer_mixer_destroy(&next) == PL_MIXER_OK);
    REQUIRE(pl_mixer_destroy(&next_mixer) == PL_MIXER_OK);
    // Resize requires a new prepared layout. Rejected begin runs no source work.
    tp.w = tp.h = 8;
    pl_tex resized = pl_tex_create(gpu, &tp); REQUIRE(resized);
    struct pl_frame resized_target = target; resized_target.planes[0].texture = resized;
    int prior = counters[1].acquired + counters[2].acquired;
    REQUIRE(pl_renderer_mixer_begin(m, &mix, times, 1.5, 2, &resized_target) == PL_MIXER_INVALID);
    REQUIRE(counters[1].acquired + counters[2].acquired == prior);
    REQUIRE(pl_renderer_mixer_describe(rr, mixer, &mix, &resized_target, &snapshot, &next) == PL_RENDERER_PREPARE_OK);
    prepare(next);
    REQUIRE(pl_renderer_mixer_begin(next, &mix, times, 1.5, 3, &resized_target) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_submit(next) == PL_MIXER_PENDING);
    REQUIRE(complete(next) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_finish(next, &resized_target) == PL_MIXER_OK);
    pixels(gpu, resized, .35f, .55f, .5f, linear, 1, target.lut);
    REQUIRE(pl_renderer_mixer_destroy(&next) == PL_MIXER_OK);
    pl_tex_destroy(gpu, &resized);
    // Explicitly discard prepared but unsubmitted work, then retire normally.
    REQUIRE(pl_renderer_mixer_begin(m, &mix, times, 1.5, 2, &target) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_discard(m) == PL_MIXER_OK);
    REQUIRE(pl_renderer_mixer_destroy(&m) == PL_MIXER_OK);
    REQUIRE(pl_mixer_destroy(&mixer) == PL_MIXER_OK);
    REQUIRE(stats.instances == 0 && stats.submissions == stats.releases && owner.refs == 0);
    for (int n = 0; n < 3; n++) {
        REQUIRE(counters[n].acquired == counters[n].released);
        pl_tex_destroy(gpu, &textures[n]);
    }
    pl_tex_destroy(gpu, &output);
    pl_renderer_destroy(&rr);
}
int main(int argc, char **argv)
{
    REQUIRE(argc == 2);
#ifdef _WIN32
    HMODULE library = LoadLibraryA(argv[1]); REQUIRE(library);
    pl_mixer_module_entry_fn entry = (pl_mixer_module_entry_fn)GetProcAddress(library, PL_MIXER_MODULE_ENTRY);
#else
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL); REQUIRE(library);
    pl_mixer_module_entry_fn entry = (pl_mixer_module_entry_fn)dlsym(library, PL_MIXER_MODULE_ENTRY);
#endif
    REQUIRE(entry);
    struct pl_mixer_module module = {.struct_size = sizeof(module)};
    REQUIRE(entry(PL_API_VER, PL_MIXER_ABI_VERSION, &module) == PL_MIXER_OK);
    pl_log log = pl_test_logger();
    pl_vulkan vk = pl_vulkan_create(log, pl_vulkan_params(.allow_software = true));
    if (!vk) return SKIP;
    if (!pl_pass_prepare_supported(vk->gpu)) return SKIP;
    pl_mixer_registry registry = pl_mixer_registry_create();
    REQUIRE(pl_mixer_register(registry, module.mixers) == PL_MIXER_OK);
    for (int linear = 0; linear < 2; linear++)
        for (int lut = 0; lut < 2; lut++) run(vk->gpu, registry, linear, lut);
    REQUIRE(pl_mixer_unregister(registry, module.mixers->id) == PL_MIXER_OK);
    REQUIRE(pl_mixer_registry_destroy(&registry) == PL_MIXER_OK);
    pl_vulkan_destroy(&vk); pl_log_destroy(&log);
#ifdef _WIN32
    REQUIRE(FreeLibrary(library));
#else
    REQUIRE(dlclose(library) == 0);
#endif
    return 0;
}
