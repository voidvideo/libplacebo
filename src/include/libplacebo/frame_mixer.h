/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBPLACEBO_FRAME_MIXER_H_
#define LIBPLACEBO_FRAME_MIXER_H_

#include <libplacebo/gpu.h>
#include <libplacebo/colorspace.h>

PL_API_BEGIN

// External implementations use the same libplacebo ABI as their host. This
// additional version covers the callback table, not an ABI-independent GPU SDK.
#define PL_MIXER_ABI_VERSION 2

typedef struct pl_mixer_registry_t *pl_mixer_registry;
typedef struct pl_mixer_t *pl_mixer;
typedef struct pl_mixer_job_t *pl_mixer_job;

enum pl_mixer_status {
    PL_MIXER_OK = 0,
    PL_MIXER_PENDING,
    PL_MIXER_UNSUPPORTED,
    PL_MIXER_INVALID,
    PL_MIXER_FAILED,
    PL_MIXER_BUSY,
};

enum pl_mixer_backend {
    PL_MIXER_BACKEND_VULKAN = 1,
    PL_MIXER_BACKEND_CUDA,
};

enum pl_mixer_color_domain {
    PL_MIXER_COLOR_COMPLETED = 0,
    PL_MIXER_COLOR_LINEAR,
};

struct pl_mixer_config {
    uint64_t revision;
    // Implementation-defined settings matching descriptor.config_schema.
    // Borrowed for negotiate/create only. The implementation copies retained data.
    const void *data;
    size_t size;
};

struct pl_mixer_requirements {
    enum pl_mixer_backend backend;
    enum pl_mixer_color_domain domain;
    // Temporal support in source-frame units, identical to an internal frame
    // mixer's filter radius. The ordinary queue selects the chronological mix.
    float radius;
    // Maximum variable-length mix accepted by the implementation. Every
    // nonempty ordinary mix up to this size must be accepted.
    uint32_t max_frames;
    uint32_t max_in_flight;
    // One handle type or zero. Must be chosen BEFORE allocating shared images.
    enum pl_handle_type export_handle;
    // Format belongs to the supplied gpu; applies to every input and output.
    pl_fmt format;
};

struct pl_mixer_frame {
    pl_tex texture;
    uint64_t content_id;    // changes when rendered content changes
    uint64_t allocation_id; // changes when storage is replaced, even at same address
    double pts;            // seconds in the job's epoch, not normalized filter units
};

struct pl_mixer_job_params {
    uint64_t epoch; // changes on seek/reset; caches must not cross epochs
    double pts;
    const struct pl_mixer_frame *frames;
    uint32_t num_frames;
    pl_tex output; // separate writable destination; never aliases an input
    uint64_t output_allocation_id;
    struct pl_color_space color;
    enum pl_alpha_mode alpha;
    // Owns ALL textures and related synchronization resources. The core retains
    // once before submit and releases only after release_job, including failures.
    // Caller must prevent writes/recycling while this lease exists.
    void *owner;
    void (*retain)(void *owner);
    void (*release)(void *owner);
};

struct pl_mixer_descriptor {
    uint32_t abi_version;
    size_t struct_size;
    const char *id;
    const char *name;
    const char *config_schema; // optional UTF-8 JSON schema; settings semantics are plugin-owned
    void *priv;
    // Keeps descriptor strings, callbacks and native library mapped. Both
    // functions are mandatory. Registration holds one reference until unregister.
    void (*retain)(void *priv);
    void (*release)(void *priv);

    // Synchronous capability query; must not allocate shared frame resources.
    enum pl_mixer_status (*negotiate)(void *priv, pl_gpu gpu,
        const struct pl_mixer_config *, struct pl_mixer_requirements *out);
    // Creates one immutable configured instance. Expensive preparation belongs
    // here; the application invokes this off its render thread. On failure the
    // implementation cleans up and leaves *out NULL. No pending create in v1.
    enum pl_mixer_status (*create)(void *priv, pl_gpu gpu,
        const struct pl_mixer_config *, const struct pl_mixer_requirements *, void **out);
    // Job metadata remains valid until release_job. OK means GPU COMPLETE.
    // PENDING means admitted work: *ticket may be NULL if unnecessary.
    // Any other return means no outstanding GPU/resource use and no ticket.
    // A failure after GPU submission MUST remain PENDING until poll can report
    // terminal failure with no outstanding resource use.
    enum pl_mixer_status (*submit)(void *instance,
        const struct pl_mixer_job_params *, void **ticket);
    // Nonblocking. Returns only PENDING, OK or FAILED. Terminal means all external
    // GPU access has ended AND images have been returned to libplacebo ownership.
    enum pl_mixer_status (*poll)(void *instance, void *ticket);
    void (*release_job)(void *instance, void *ticket);
    void (*destroy)(void *instance);
};

// Optional dynamic-library discovery convention. Void (or another application)
// loads the module and resolves PL_MIXER_MODULE_ENTRY. The module rejects an
// incompatible libplacebo API or mixer ABI; no C++/Rust objects cross this call.
// The returned array and descriptors remain borrowed while the module is loaded.
// Registration retains each implementation through its retain/release callbacks.
// The loader keeps its own module reference until all registrations are removed.
#define PL_MIXER_MODULE_ENTRY "pl_mixer_module_entry"
struct pl_mixer_module {
    size_t struct_size; // caller sets capacity before invoking entry
    uint32_t num_mixers;
    const struct pl_mixer_descriptor *mixers;
};
typedef enum pl_mixer_status (*pl_mixer_module_entry_fn)(
    uint32_t libplacebo_api_version, uint32_t mixer_abi_version,
    struct pl_mixer_module *out);

// All calls for a registry and its instances are caller-serialized. No callbacks
// run on an implicit worker. The caller chooses the execution thread. Device and
// queue synchronization still follows the gpu/vulkan API contracts.
PL_API pl_mixer_registry pl_mixer_registry_create(void);
// BUSY preserves the registry if registered implementations remain.
PL_API enum pl_mixer_status pl_mixer_registry_destroy(pl_mixer_registry *registry);
PL_API enum pl_mixer_status pl_mixer_register(pl_mixer_registry,
    const struct pl_mixer_descriptor *);
// BUSY preserves registration while any instance exists.
PL_API enum pl_mixer_status pl_mixer_unregister(pl_mixer_registry, const char *id);
PL_API int pl_mixer_registry_count(pl_mixer_registry);
// Borrowed until unregister. The callback table is copied at registration.
PL_API const struct pl_mixer_descriptor *pl_mixer_registry_get(pl_mixer_registry, int index);
PL_API enum pl_mixer_status pl_mixer_create(pl_mixer_registry, const char *id,
    pl_gpu, const struct pl_mixer_config *, pl_mixer *out);
PL_API const struct pl_mixer_requirements *pl_mixer_get_requirements(pl_mixer);
// Retained renderer integrations use these paired references. Each retain must
// have one release; the original owner still calls destroy after references end.
PL_API pl_gpu pl_mixer_get_gpu(pl_mixer);
PL_API void pl_mixer_retain(pl_mixer);
PL_API void pl_mixer_release(pl_mixer);
// BUSY preserves the instance until all jobs and retained owners have released.
PL_API enum pl_mixer_status pl_mixer_destroy(pl_mixer *);
PL_API enum pl_mixer_status pl_mixer_submit(pl_mixer,
    const struct pl_mixer_job_params *, pl_mixer_job *out);
PL_API enum pl_mixer_status pl_mixer_poll(pl_mixer_job);
// BUSY preserves pending jobs and their leases. No implicit cancellation or wait.
PL_API enum pl_mixer_status pl_mixer_job_destroy(pl_mixer_job *);

// Vulkan implementations can use pl_vulkan_unwrap/hold_ex/release_ex on the
// supplied gpu/textures. CUDA implementations use exportable texture storage
// and external semaphores through those same Vulkan APIs. CUDA imports, streams,
// device matching, layout conversion and handle duplication belong to the plugin.
// This lifecycle API does not load libraries or select queue frames. The
// pl_renderer_mixer API in renderer.h supplies prepared renderer cache integration.
// Plugins must end imported-resource lifetimes before releasing the owning
// job, or explicitly retain its owner for the entire longer import lifetime.
// Completion must leave no semaphore dependency referencing an object about
// to be destroyed. In v1, plugins may return images after observing completion
// with no remaining release semaphore dependency.
// Those integration operations are separate from this implementation lifecycle.

PL_API_END
#endif
