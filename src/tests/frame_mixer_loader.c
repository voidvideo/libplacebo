#include "utils.h"
#include <assert.h>
#include <string.h>
#include <libplacebo/frame_mixer.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
static void load_cycle(const char *path)
{
#ifdef _WIN32
    HMODULE library = LoadLibraryA(path);
    assert(library);
    pl_mixer_module_entry_fn entry = (pl_mixer_module_entry_fn)
        GetProcAddress(library, PL_MIXER_MODULE_ENTRY);
#else
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    assert(library);
    pl_mixer_module_entry_fn entry = (pl_mixer_module_entry_fn)
        dlsym(library, PL_MIXER_MODULE_ENTRY);
#endif
    assert(entry);
    struct pl_mixer_module module = {.struct_size = sizeof(module)};
    assert(entry(PL_API_VER, PL_MIXER_ABI_VERSION + 1, &module) == PL_MIXER_UNSUPPORTED);
    assert(entry(PL_API_VER - 1, PL_MIXER_ABI_VERSION, &module) == PL_MIXER_UNSUPPORTED);
    assert(entry(PL_API_VER, PL_MIXER_ABI_VERSION, &module) == PL_MIXER_OK);
    assert(module.num_mixers == 1);
    struct pl_mixer_module too_small = {.struct_size = sizeof(size_t)};
    assert(entry(PL_API_VER, PL_MIXER_ABI_VERSION, &too_small) == PL_MIXER_INVALID);
    assert(entry(PL_API_VER, PL_MIXER_ABI_VERSION, NULL) == PL_MIXER_INVALID);
    pl_mixer_registry r = pl_mixer_registry_create();
    assert(pl_mixer_register(r, &module.mixers[0]) == PL_MIXER_OK);
    const struct pl_mixer_descriptor *d = pl_mixer_registry_get(r, 0);
    assert(d && !strcmp(d->id, "test.loaded"));
    const struct pl_gpu_t gpu = {0};
    struct pl_mixer_config config = {0};
    pl_mixer mixer = NULL;
    assert(pl_mixer_create(r, d->id, &gpu, &config, &mixer) == PL_MIXER_UNSUPPORTED);
    assert(!mixer);
    assert(pl_mixer_unregister(r, d->id) == PL_MIXER_OK);
    assert(pl_mixer_registry_destroy(&r) == PL_MIXER_OK);
#ifdef _WIN32
    assert(FreeLibrary(library));
#else
    assert(dlclose(library) == 0);
#endif
}
int main(int argc, char **argv)
{
    assert(argc == 2);
    for (unsigned i = 0; i < 32; i++) load_cycle(argv[1]);
    return 0;
}
