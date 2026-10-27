// Separately built external module: it does not link to private renderer code.
#include <libplacebo/frame_mixer.h>
static void retain(void *p) {}
static void release(void *p) {}
static enum pl_mixer_status negotiate(void *p, pl_gpu gpu,
    const struct pl_mixer_config *c, struct pl_mixer_requirements *r)
{
    // This loader fixture deliberately supplies no processing implementation.
    return PL_MIXER_UNSUPPORTED;
}
static enum pl_mixer_status create(void *p, pl_gpu g,
    const struct pl_mixer_config *c, const struct pl_mixer_requirements *r, void **out)
{ return PL_MIXER_UNSUPPORTED; }
static enum pl_mixer_status submit(void *p, const struct pl_mixer_job_params *j, void **t)
{ return PL_MIXER_UNSUPPORTED; }
static enum pl_mixer_status poll(void *p, void *t) { return PL_MIXER_FAILED; }
static void release_job(void *p, void *t) {}
static void destroy(void *p) {}
static const struct pl_mixer_descriptor descriptor = {
    .abi_version = PL_MIXER_ABI_VERSION, .struct_size = sizeof(descriptor),
    .id = "test.loaded", .name = "External loader test",
    .retain = retain, .release = release, .negotiate = negotiate,
    .create = create, .submit = submit, .poll = poll,
    .release_job = release_job, .destroy = destroy,
};
enum pl_mixer_status pl_mixer_module_entry(uint32_t api, uint32_t abi,
                                          struct pl_mixer_module *out);
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
enum pl_mixer_status pl_mixer_module_entry(uint32_t api, uint32_t abi,
                                          struct pl_mixer_module *out)
{
    if (api != PL_API_VER || abi != PL_MIXER_ABI_VERSION)
        return PL_MIXER_UNSUPPORTED;
    if (!out || out->struct_size < sizeof(*out)) return PL_MIXER_INVALID;
    out->num_mixers = 1;
    out->mixers = &descriptor;
    return PL_MIXER_OK;
}
