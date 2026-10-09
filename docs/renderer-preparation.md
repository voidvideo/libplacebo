# Renderer preparation in this fork

Use the retained renderer preparation API to compile a rendering graph before presenting it. Select the preparation policy through `pl_renderer_snapshot.mode`. Existing ordinary `pl_render_image` and `pl_render_image_mix` calls keep their synchronous behavior.

## Choose the preparation policy

| Mode | Supported graph | Graph cannot be prepared |
| --- | --- | --- |
| `PL_RENDERER_PREPARE_ASYNC` | Compile passes through the preparation worker pool | Return `PL_RENDERER_PREPARE_UNSUPPORTED` |
| `PL_RENDERER_PREPARE_AUTO` | Use the same asynchronous path | Retain a synchronous renderer for execution |
| `PL_RENDERER_PREPARE_SYNC` | Use ordinary rendering at execution | Follow ordinary renderer support and errors |

The zero value is ASYNC, preserving the strict behavior of existing fork callers. Set AUTO explicitly to permit fallback. These policies govern preparation, not GPU execution. Resource staging can still occur on the submitting thread.

AUTO selects fallback only for `UNSUPPORTED` during description. Invalid inputs, shader failures, worker failures, and runtime binding mismatches are not silently retried through synchronous rendering. Its fallback covers the whole selected graph; it does not divide one graph into synchronous and asynchronous regions.

Call `pl_renderer_prepare_mode` on a successfully described handle to inspect the effective ASYNC or SYNC mode before executing. AUTO is a requested policy, never an effective mode.

## Retain, describe, submit, poll, execute

The snapshot owner retains immutable render parameters, all referenced options, hook contexts, and layout resources. Both retain and release callbacks are required. The source and target frame pointers passed to description are not retained; supply valid runtime frames again at preflight and execution. GPU dependencies must outlive the preparation handle. Calls on a handle are serialized by its owner.

Description performs no GPU allocation, compilation, dispatch, or ordinary hook invocation. ASYNC records the selected graph. SYNC retains the snapshot without invoking the ordinary renderer.

Submit stages ASYNC resources and queues compilation. Poll collects completed passes and admits more work when worker capacity becomes available. A SYNC handle becomes ready after submission without compiling or rendering.

For ASYNC, execution first preflights the entire selected graph, including every contributing source and the mixed output, before issuing GPU work. A changed layout returns NOT_READY and requires a new candidate. Description requires valid texture/layout metadata up front. Execution acquires frames using native ownership rules and revalidates acquired bindings before dispatch. Description and standalone preflight never acquire frames. All successful and failed acquisition attempts are released. Temporal sources are acquired and released sequentially, and cached sources need no acquisition. A later acquisition failure can occur after an earlier source submitted GPU work; execution is not a rollback transaction.

For SYNC, preflight checks readiness and basic arguments only. Execution calls the ordinary renderer and can allocate, compile, invoke ordinary hooks, or acquire/release frames on the calling thread. It does not provide the ASYNC guarantee of complete validation before GPU work.

Destroy releases worker requests, prepared passes, textures, renderer state, and retained snapshot references. A candidate can be destroyed before preparation completes.

## Temporal mixing preserves native state

Enabled mixing prepares a source cache for each supplied frame and an output variant for each possible contributor count. The source window can exceed 16 frames; output variants retain the native limit of 16 simultaneous contributors. Empty destinations or unavailable intermediate formats use the same nearest-image fallback as ordinary rendering, with asynchronous preparation. Different weights and smaller contributor counts reuse prepared passes. More contributors or incompatible source layouts need a new candidate.

Native and prepared mixing share filter blur and weight calculation. Both perform the mixed-output conversion through the same builder. Source preparations share renderer state, including peak-detection history. Source cache population and output presentation advance one temporal shader sequence. Cache eligibility and eviction follow native mixing, including retaining eligible frames below the contribution cutoff.

Callers must change frame signatures when source content or upstream hook values change. Cache reuse skips source processing and source hooks; output hooks still execute for each presentation.

## Validation and remaining limits

`src/tests/renderer_prepare.c` checks:

- Every exposed non-null frame-mixer preset, plus Mitchell, against ordinary rendering using distinct source colors; a 17-frame source window with fewer active contributors.
- Changing weights, one-to-many contributor transitions, rolling signatures, source cache reuse, and ordinary/prepared hook counts.
- Exact temporal-dither output and HDR mixing with immediate peak detection against ordinary rendering.
- Rejection of an incompatible later source before any source dispatch or buffer write.
- ASYNC, AUTO, and SYNC with supported hooks and legacy hooks, through image and mix entry points.
- No ordinary callbacks during description/submission/preflight; no synchronous pass compilation during ASYNC execution; snapshot release after destruction.

The renderer test and full suite are run with `uv run meson test -C build-full --print-errorlogs`. Local verification uses the configured Vulkan GPU and LittleCMS build. Passing this suite does not establish every feature combination or hardware backend.

Additional tests cover acquire/release errors and interlaced neighbors, absent input and empty crops with clearing/overlays, and Dolby Vision base and enhancement layers. AUTO can select ordinary rendering for explicitly unsupported description paths. Coverage gaps do not establish missing feature support. Feature parity is determined against ordinary rendering behavior; arbitrary legacy hooks still require AUTO fallback or explicit preparation callbacks.

Exact functions, line references, file hashes, and test evidence are recorded in the [implementation record](renderer-preparation-implementation.md).

## Parallel pass compilation

The per-GPU preparation service uses `max(1, min(4, available CPUs - 1))`
workers, bounded separately from its 64-request capacity. CPU count uses Linux
thread affinity or Windows process affinity; other POSIX systems use the online
CPU count. Detection failure falls back conservatively to one worker. This
counts logical CPUs and does not interpret container CPU-time quotas; it reduces
contention but does not reserve a CPU exclusively for rendering. Each worker claims a pending request under the
service mutex, then releases that mutex before shader translation and pipeline
creation. Independent passes can finish out of order; readiness and take remain
per-request. Shaderc options/results and Vulkan pipeline caches are per-job.

Cancellation or release does not wait for an in-flight compiler/driver call.
The owning worker finishes that call and retires its abandoned result. GPU
shutdown cancels queued work, wakes all workers and joins every started worker
before releasing the service. If thread creation partially succeeds, the service
uses the workers that started; if none start, preparation remains unavailable.
No public API or SONAME change is required.

This removes the single-worker bottleneck. It does not move renderer graph
description, intermediate allocation or LUT staging off the calling thread,
and it does not parallelize the internals of one shader compilation.
