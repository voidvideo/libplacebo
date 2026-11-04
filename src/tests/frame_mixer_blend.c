// Independently loaded Vulkan mixer fixture, using only public libplacebo APIs.
#include <stdlib.h>
#include <string.h>
#include <libplacebo/vulkan.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/custom.h>
#include "frame_mixer_blend.h"
struct instance { pl_gpu gpu; pl_dispatch dispatch; struct blend_config config; };
struct ticket { pl_tex output; };
static void retain(void *p) {}
static void release(void *p) {}
static enum pl_mixer_status negotiate(void *p, pl_gpu gpu,
    const struct pl_mixer_config *config, struct pl_mixer_requirements *out)
{
    if (config->size != sizeof(struct blend_config) || !pl_vulkan_get(gpu))
        return PL_MIXER_UNSUPPORTED;
    const struct blend_config *c = config->data;
    *out = (struct pl_mixer_requirements) {
        .backend = PL_MIXER_BACKEND_VULKAN, .domain = c->domain,
        .radius = 1, .max_frames = 2,
        .max_in_flight = 2, .format = pl_find_named_fmt(gpu, "rgba32f"),
    };
    return out->format ? PL_MIXER_OK : PL_MIXER_UNSUPPORTED;
}
static enum pl_mixer_status create(void *p, pl_gpu gpu,
    const struct pl_mixer_config *config, const struct pl_mixer_requirements *q, void **out)
{
    struct instance *i = calloc(1, sizeof(*i));
    if (!i) return PL_MIXER_FAILED;
    i->gpu = gpu;
    memcpy(&i->config, config->data, sizeof(i->config));
    i->dispatch = pl_dispatch_create(gpu->log, gpu);
    i->config.stats->instances++;
    *out = i;
    return PL_MIXER_OK;
}
static enum pl_mixer_status submit(void *priv, const struct pl_mixer_job_params *job, void **out)
{
    struct instance *i = priv;
    struct ticket *ticket = calloc(1, sizeof(*ticket));
    if (!ticket) return PL_MIXER_FAILED;
    float t = (job->pts - job->frames[0].pts) / (job->frames[1].pts - job->frames[0].pts);
    float gain = i->config.gain;
    struct pl_shader_desc descriptors[2] = {
        {.desc = {.name = "a", .type = PL_DESC_SAMPLED_TEX},
         .binding = {.object = job->frames[0].texture}},
        {.desc = {.name = "b", .type = PL_DESC_SAMPLED_TEX},
         .binding = {.object = job->frames[1].texture}},
    };
    struct pl_shader_var variables[2] = {
        {.var = pl_var_float("t"), .data = &t, .dynamic = true},
        {.var = pl_var_float("gain"), .data = &gain, .dynamic = true},
    };
    pl_shader shader = pl_dispatch_begin(i->dispatch);
    bool ok = pl_shader_custom(shader, &(struct pl_custom_shader) {
        .description = "external Vulkan blend fixture",
        .body = "color = mix(texelFetch(a, ivec2(gl_FragCoord.xy), 0), "
                "texelFetch(b, ivec2(gl_FragCoord.xy), 0), t); color.rgb *= gain;",
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR,
        .descriptors = descriptors, .num_descriptors = 2,
        .variables = variables, .num_variables = 2,
    });
    if (ok) ok = pl_dispatch_finish(i->dispatch, &(struct pl_dispatch_params) {
        .shader = &shader, .target = job->output,
    });
    if (!ok) {
        pl_dispatch_abort(i->dispatch, &shader);
        pl_gpu_finish(i->gpu); // fixture failure must have no outstanding access
        free(ticket);
        return PL_MIXER_FAILED;
    }
    pl_gpu_flush(i->gpu);
    ticket->output = job->output;
    struct blend_stats *s = i->config.stats;
    s->submissions++;
    s->epoch = job->epoch;
    for (int f = 0; f < 2; f++) {
        s->content[f] = job->frames[f].content_id;
        s->allocation[f] = job->frames[f].allocation_id;
    }
    *out = ticket;
    return PL_MIXER_PENDING;
}
static enum pl_mixer_status poll(void *priv, void *ptr)
{
    struct instance *i = priv;
    struct ticket *ticket = ptr;
    i->config.stats->polls++;
    if (i->config.stats->hold || pl_tex_poll(i->gpu, ticket->output, 0))
        return PL_MIXER_PENDING;
    return i->config.stats->fail ? PL_MIXER_FAILED : PL_MIXER_OK;
}
static void cleanup(void *priv, void *ptr)
{
    struct instance *i = priv;
    i->config.stats->releases++;
    free(ptr);
}
static void destroy(void *priv)
{
    struct instance *i = priv;
    i->config.stats->instances--;
    pl_dispatch_destroy(&i->dispatch);
    free(i);
}
static const struct pl_mixer_descriptor desc = {
    .abi_version = PL_MIXER_ABI_VERSION, .struct_size = sizeof(desc),
    .id = "test.vulkan-blend", .name = "Vulkan blend test",
    .retain = retain, .release = release, .negotiate = negotiate, .create = create,
    .submit = submit, .poll = poll, .release_job = cleanup, .destroy = destroy,
};
enum pl_mixer_status pl_mixer_module_entry(uint32_t api, uint32_t abi, struct pl_mixer_module *out);
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
enum pl_mixer_status pl_mixer_module_entry(uint32_t api, uint32_t abi, struct pl_mixer_module *out)
{
    if (api != PL_API_VER || abi != PL_MIXER_ABI_VERSION) return PL_MIXER_UNSUPPORTED;
    if (!out || out->struct_size < sizeof(*out)) return PL_MIXER_INVALID;
    out->num_mixers = 1; out->mixers = &desc;
    return PL_MIXER_OK;
}
