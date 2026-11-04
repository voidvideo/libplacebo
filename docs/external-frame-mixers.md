# External frame mixer C interface

The public contract in [frame_mixer.h](../src/include/libplacebo/frame_mixer.h)
supplies registration, configuration negotiation, and retained asynchronous jobs.
The adapter in [renderer.h](../src/include/libplacebo/renderer.h) connects those
jobs to prepared source caches and the renderer's color-finishing stage.
The application loads modules and supplies the queue-selected frame mix. Existing
`pl_render_image_mix` calls continue to use their ordinary filter path.

## Load and control an implementation

The application loads a library and resolves `PL_MIXER_MODULE_ENTRY`.
The entry function receives `PL_API_VER`, `PL_MIXER_ABI_VERSION`, and a
caller-sized `pl_mixer_module`. It returns descriptors for its implementations.
The module must reject incompatible versions. The loader test builds and opens
an actual separate shared library using this convention.

The application registers descriptors with `pl_mixer_register`. It can enumerate
them with `pl_mixer_registry_count` and `pl_mixer_registry_get`. Each descriptor
has an implementation ID, display name, optional JSON settings schema, and C
callbacks. No Void types cross the interface. Libplacebo does not load libraries
or introduce a second application plugin manager.

`pl_mixer_create` passes the application's immutable configuration bytes and
revision to negotiation, validates returned requirements, and creates an instance.
Configuration updates create another instance. The application decides when that
instance becomes active; this API does not substitute an earlier configuration
when creation fails. Creation is synchronous and may compile pipelines or load
models. Call it on the application's preparation worker.

The implementation chooses a Vulkan or CUDA backend, a completed or linear RGB
contract, a texture format, a temporal radius, a maximum input capacity and a maximum number of outstanding jobs,
and an optional export handle type. Negotiate these before allocating shared
frames. Linear input requires a floating-point format. CUDA requires an admitted
export handle; that check alone does not prove CUDA device or format compatibility.
The implementation must check those additional requirements during creation.

All API calls for a registry and its instances are caller-serialized. The API
creates no worker threads and does not determine which application thread runs
callbacks. The application must also honor libplacebo device/queue threading rules.

## Submit and retain frames

A job contains an output timestamp in seconds, an epoch, chronological input
frames, and a separate output texture. Each input has a content identity and an
allocation identity. All frames use the job's color space and alpha convention.
The caller supplies textures from the instance's GPU. Aliasing the output with
any input is invalid, including through another wrapper over the same allocation.
The core rejects identical input/output texture handles; the caller remains
responsible for excluding storage aliases hidden behind different handles.

The implementation declares the same temporal radius used by an internal frame
mixer and a maximum input capacity. The application uses the ordinary queue path
to select a chronological `pl_frame_mix`. Every nonempty ordinary mix within the
declared capacity is valid. The interface does not prescribe a pair, directional
counts, a motion field, or an inference method.

The core copies input descriptors and retains the supplied resource owner before
calling the implementation. The owner must keep all textures and associated
synchronization objects alive and prevent conflicting writes. The core does not
increment an implicit `pl_tex` reference count: no such ownership is assumed.

`submit` can return:

- `OK`: output is complete and GPU access has ended; a releasable job is returned.
- `PENDING`: work is admitted and a job is returned. Poll without blocking.
- An error: no admitted work, ticket, or outstanding resource access remains.

A failure after GPU submission must stay pending until external resource use has
ended. `poll` then returns `FAILED`. Neither submission acceptance nor pending
work means that an output may be presented. This initial API uses completion
polling; consumer-side GPU dependency chaining is not implemented here.

Destroying a pending job returns `BUSY` and preserves the job. After terminal
completion, job destruction calls the implementation's ticket cleanup, then
releases the resource owner. Instance destruction refuses live jobs and retained adapter owners; unregister
refuses live instances; registry destruction refuses live registrations. There
is no implicit cancellation, wait, retry, or alternate mixer selection.

## Connect jobs to the prepared renderer

Create a `pl_renderer_mixer` for an immutable mixer configuration and renderer
snapshot. The snapshot must use `PL_RENDERER_PREPARE_ASYNC`, and its mixer color
domain must match the implementation's negotiated requirements.

1. Call `pl_renderer_mixer_describe` with representative source frames and target.
   It describes source-cache and output passes without running the mixer.
2. Call `pl_renderer_mixer_stage` to allocate textures and submit shader preparation.
   Poll `pl_renderer_mixer_prepare_poll` until ready. A failed preparation is not usable.
3. Call `pl_renderer_mixer_begin` with the queue-selected mix, source timestamps,
   requested timestamp, epoch, and target. It renders cache misses and reserves
   those textures. It does not call the external implementation.
4. Call `pl_renderer_mixer_submit`. Poll `pl_renderer_mixer_poll` while pending.
   These callbacks may run on another worker after a serialized ownership handoff.
5. After `OK`, call `pl_renderer_mixer_finish` on the renderer owner. It acquires
   the destination, runs the prepared output stage, and releases the request.
6. Destroy the adapter after all requests and retained resource leases have ended.
   Then destroy the mixer instance and unload its module through the core lifecycle.

Source timestamps and the requested timestamp are absolute seconds.
`pl_frame_mix.timestamps` does not participate in this adapter. The application
passes the same ordinary queue selection it would give an internal frame mixer.
The adapter does not select presentation deadlines or invent missing source frames.

The adapter owns cached rendered textures, a separate output texture, and the
prepared graph. It borrows source frame descriptors only during describe/begin;
it does not retain borrowed queue pointers. The caller must maintain the snapshot's
referenced dependencies and GPU for the adapter lifetime. The first source's
inferred color metadata supplies the output hook mapping context.

Unchanged source signatures reuse rendered pixels and content IDs. A newly rendered
source receives a new content ID. Allocation IDs stay stable while those textures
exist. An epoch change invalidates cached contents. A pending request or longer
plugin lease prevents the adapter from reusing its textures.

Completed RGB uses the existing completed-frame cache contract. Linear RGB uses
the explicit linear cache contract and runs deferred color finishing after mixing.
Both paths reuse the renderer's source and output builders, including their LUT
placement. The mixer receives the actual cache color metadata and premultiplied
alpha. Export requirements apply when allocating both cache and output textures.

One request is admitted per adapter. Independent adapters can overlap, including
old and new immutable configurations. A resize or incompatible target/color
configuration requires a newly described adapter. Target LUT and ICC configuration
must remain consistent with the described graph.

`finish` returns `PENDING` without acquiring or writing the target while the job is
pending. A failed request remains available for explicit `discard`. Discard does
not cancel GPU work: it returns `BUSY` until pending work and retained leases end.
A request prepared by `begin` but not submitted can also be discarded. Destruction
returns `BUSY` while a request or retained lease exists. No alternate output is
selected when these operations fail.

The current adapter requires sampleable, renderable four-component textures and
enabled intermediate framebuffers. It supports at most 16 cache slots, with no more
inputs per request than the initially described source count. Nonzero source
rotation is explicitly unsupported. It does not add automatic queue selection or
application configuration-generation management.

## Backend responsibilities

A Vulkan implementation receives `pl_gpu` at creation and `pl_tex` values in jobs.
It can use the existing Vulkan APIs to obtain device/image information and to
transfer image ownership with `pl_vulkan_hold_ex` and `pl_vulkan_release_ex`.
Raw image access requires a successful hold; unwrapping alone is insufficient.
Native queue access must obey the Vulkan context's queue synchronization contract.

A CUDA implementation lives in the external library. Libplacebo does not link
CUDA or TensorRT. The plugin owns matching-device selection, CUDA contexts and
streams, external memory/semaphore imports, tensor conversion, inference, and
cleanup. Export capability is an allocation requirement, not a property that can
be added to an arbitrary cached image afterward. Memory handle ownership follows
`pl_shared_mem`; clone a handle when the importing API takes ownership.

The plugin must return images to libplacebo ownership before terminal completion.
Any semaphore dependency must remain valid until libplacebo has consumed it.
Observing external GPU completion and releasing images without a remaining
semaphore dependency is one valid implementation for this initial completion API.
The plugin may retain a job owner explicitly for a longer import lifetime, but
must release that reference before final retirement. GPU-only tensor conversions
are permitted; the interface does not require CPU readback.

## Verified scope and remaining integration

`renderer_mixer` loads a separate Vulkan blend module and executes GPU shaders.
It checks completed and linear RGB, nonlinear target LUT placement, source render
reuse, allocation/content identities, pending output protection, explicit failure,
configuration overlap, resize, and resource retirement. Twenty consecutive runs
passed on an NVIDIA GeForce RTX 5070. This is a correctness fixture with small
textures, not a 4K throughput or fullscreen playback benchmark.

All 30 configured tests subsequently passed using Void's staged Vulkan loader
and Khronos validation layer, both version 1.4.363, with validation enabled.
The logs contained no validation errors or VUID messages. This includes the
prepared-render numerical regression and external mixer fixture. The queue-plan
regression tests the agreed available-frame policy; publication demand tests use
the explicit bracket intent. The queue implementation is unchanged.

CUDA resource sharing has not been executed on hardware by these tests. Void's
application-side admission, worker handoff, and configuration retirement still
need integration. The adapter does not select an external mixer through
`pl_render_params`, and these changes do not update Void's dependency pin or binaries.
