# External frame mixer C interface

This is the first implementation of the external implementation lifecycle.
The public contract is in [frame_mixer.h](../src/include/libplacebo/frame_mixer.h).
It supplies registration, enumeration, configuration negotiation, and retained
asynchronous jobs. It does **not yet connect to the renderer's internal frame
cache or select an external implementation through `pl_render_params`**.
Existing rendering behavior is unchanged.

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
contract, a texture format, a frame window, a maximum number of outstanding jobs,
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

The required past/future counts are exact counts strictly before/after the target.
An exact timestamp sample is additional and counts toward the total frame limit.
These are admission requirements, not instructions to fabricate missing frames.
A caller that cannot assemble the required window must not submit the job.
The interface does not prescribe a pair, a motion field, or an inference method.

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
releases the resource owner. Instance destruction refuses live jobs; unregister
refuses live instances; registry destruction refuses live registrations. There
is no implicit cancellation, wait, retry, or alternate mixer selection.

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

These are plugin obligations, not hardware validation results. Current tests
exercise the C ABI, dynamic loading, validation, and lifetime rules with fixtures.
They do not execute Vulkan shaders or CUDA inference.

## Remaining renderer integration

The next implementation must connect this lifecycle to the prepared renderer:

1. Admit the negotiated source window through the queue and render/cache each
   source under the selected completed or linear color contract.
2. Include export requirements in cache allocation and identity. Keep allocations
   leased across asynchronous jobs without retaining borrowed queue pointers.
3. Dispatch the external job, preserve its pending state, and consume only its
   matching completed output before the appropriate color-finishing stage.
4. Tie instance generations and retirement to Void's existing preparation owner.
5. Verify an actual external Vulkan implementation and CUDA sharing on hardware.

The existing queue prefix limit does not establish an arbitrary required temporal
window. The existing prepared-render pending status is not an external job ticket.
Both connections require explicit implementation; this lifecycle does not pretend
those integration steps already exist.
