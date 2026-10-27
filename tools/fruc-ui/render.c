#include <assert.h>
#include <libplacebo/filters.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void ref(void *x) {}
static struct pl_frame frame(pl_tex t) {
  return (struct pl_frame){.num_planes = 1,
                           .planes = {{.texture = t,
                                       .components = 4,
                                       .component_mapping = {0, 1, 2, 3}}},
                           .repr = pl_color_repr_rgb,
                           .color = pl_color_space_srgb};
}
int main(int argc, char **argv) {
  assert(argc == 3 || argc == 4);
  FILE *times = argc == 4 ? fopen(argv[3], "r") : NULL;
  assert(argc == 3 || times);
  enum { W = 960, H = 540, N = 12 };
  size_t bytes = W * H * 4;
  unsigned char *pixels = malloc(bytes);
  assert(pixels);
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  assert(in && out);
  pl_log log =
      pl_log_create(PL_API_VER, pl_log_params(.log_cb = pl_log_simple,
                                              .log_level = PL_LOG_WARN));
  pl_vk_inst inst = pl_vk_inst_create(log, pl_vk_inst_params(.debug = true));
  assert(inst);
  pl_vulkan vk = pl_vulkan_create_with_instance_extensions(
      log,
      pl_vulkan_params(.instance = inst->instance,
                       .get_proc_addr = inst->get_proc_addr),
      inst->extensions, inst->num_extensions);
  assert(vk);
  pl_gpu gpu = vk->gpu;
  pl_renderer rr = pl_renderer_create(log, gpu);
  pl_tex inputs[N];
  struct pl_frame frames[N];
  for (int i = 0; i < N; i++) {
    assert(fread(pixels, 1, bytes, in) == bytes);
    inputs[i] = pl_tex_create(
        gpu,
        pl_tex_params(.w = W, .h = H, .format = pl_find_named_fmt(gpu, "rgba8"),
                      .sampleable = true, .initial_data = pixels));
    assert(inputs[i]);
    frames[i] = frame(inputs[i]);
  }
  pl_tex output = pl_tex_create(
      gpu,
      pl_tex_params(.w = W, .h = H, .format = pl_find_named_fmt(gpu, "rgba8"),
                    .renderable = true, .storable = true,
                    .host_readable = true));
  assert(output);
  struct pl_frame target = frame(output);
  struct pl_render_params params = pl_render_default_params;
  params.frame_mixer = &pl_filter_fruc;
  params.dither_params = NULL;
  struct pl_renderer_snapshot snap = {.params = &params,
                                      .owner = &params,
                                      .retain = ref,
                                      .release = ref,
                                      .mode = PL_RENDERER_PREPARE_ASYNC};
  const struct pl_frame *pair[2] = {&frames[0], &frames[1]};
  uint64_t sig[2] = {1, 2};
  float ts[2] = {-.4, .6};
  struct pl_frame_mix mix = {.num_frames = 2,
                             .frames = pair,
                             .signatures = sig,
                             .timestamps = ts,
                             .vsync_duration = 24.0 / 60};
  pl_renderer_preparation prep = NULL;
  assert(pl_renderer_describe_image_mix(rr, &mix, &target, &snap, &prep) ==
         PL_RENDERER_PREPARE_OK);
  assert(pl_renderer_prepare_submit(prep) == PL_RENDERER_PREPARE_OK);
  enum pl_pass_prepare_state state;
  while ((state = pl_renderer_prepare_poll(prep)) == PL_PASS_PREPARE_PENDING)
    usleep(1000);
  assert(state == PL_PASS_PREPARE_READY);
  for (int tick = 0; tick < 28; tick++) {
    double pts = tick / 60.0;
    if (times && fscanf(times, "%lf", &pts) != 1)
      break;
    int i = (int)floor(pts * 24);
    float t = times ? (float)(pts * 24 - i) : (tick * 2 % 5) / 5.0f;
    pair[0] = &frames[i % N];
    pair[1] = &frames[(i + 1) % N];
    sig[0] = i + 1;
    sig[1] = i + 2;
    ts[0] = -t;
    ts[1] = 1 - t;
    mix.num_frames = t == 0 ? 1 : 2;
    assert(pl_renderer_preflight_image_mix(prep, &mix, &target) ==
           PL_RENDERER_PREPARE_OK);
    assert(pl_render_image_mix_prepared(prep, &mix, &target) ==
           PL_RENDERER_PREPARE_OK);
    assert(pl_tex_download(
        gpu, pl_tex_transfer_params(.tex = output, .ptr = pixels)));
    assert(fwrite(pixels, 1, bytes, out) == bytes);
  }
  fclose(in);
  fclose(out);
  pl_renderer_prepare_destroy(&prep);
  pl_renderer_destroy(&rr);
  for (int i = 0; i < N; i++)
    pl_tex_destroy(gpu, &inputs[i]);
  pl_tex_destroy(gpu, &output);
  pl_vulkan_destroy(&vk);
  pl_vk_inst_destroy(&inst);
  pl_log_destroy(&log);
  free(pixels);
}
