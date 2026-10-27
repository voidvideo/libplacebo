# Native 4K ncnn/Vulkan speed test

On this RTX 5070, the fastest tested serial GPU-resident graph was RIFE 4.4 at
17.79 generated frames/s. Exact 3840×2160, 24→60 conversion requires 48 generated
frames/s: twelve of the sixty output timestamps coincide with source frames.
None of these four serial configurations meets that rate.

Inputs and outputs remain on the GPU throughout the timed interval. These
results do not include CPU uploads or readback. They measure the unmodified
exported graphs, not an integrated libplacebo frame queue. They do not establish
the maximum performance of a concurrent or differently optimized implementation.

| Model export | Generated frames/s | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| rife-v4.4_ensembleFalse_fastTrue | 17.79 | 53.61 | 74.72 |
| rife-v4.6_ensembleFalse | 15.26 | 62.11 | 80.17 |
| rife-v4.25-lite_ensembleFalse | 10.43 | 95.93 | 97.89 |
| rife-v4.25_ensembleFalse | 9.50 | 103.65 | 111.16 |

## Measurement boundary

The harness uses ncnn release 20250503, revision
`305837fd4a722ebc47c5d72e72d8ec9ae970e932`, which matches the upstream plugin's
pinned ncnn revision. Model exports and the Warp shader come unchanged from
VapourSynth-RIFE-ncnn-Vulkan revision
`c3ec6aabc07c8fa37a4f58d7fed9e2ad1fc1b13f`. FP16 storage and packing are enabled;
FP16 arithmetic is disabled, matching upstream. Inputs are native 3840×2160;
model-specific padding is applied without reducing image resolution.

One model runs at a time. Input frames and four timestep maps are uploaded before
nine warmups. The timer then covers at least fifteen seconds and 48 inferences,
cycling fractional times 0.4, 0.8, 0.2 and 0.6. Each inference records commands,
submits them and waits for completion. Model initialization, input upload and
output download are outside the timer. A single output is downloaded afterward
only to verify that the benchmark produced a finite, nonconstant image.

All four runs completed and passed those execution checks. The GPU was an RTX
5070 with driver 615.71.09 and 12227 MiB VRAM. The desktop remained active. This
was not a long thermal soak or live playback test.

## Secondary filter timings

The retained `timing/` records include the VapourSynth plugin's CPU packing,
upload and readback. They are diagnostic measurements and do not decide
GPU-resident feasibility. Worker counts 1, 2 and 3 ran separately. Three workers
failed for both 4.25 variants with `vkAllocateMemory failed -2`, followed by
SIGSEGV. Those failed runs have no reported throughput. No inference about the
resident path's memory limit follows from those plugin failures.

The user clarified that this task is speed-only. Additional quality evaluation
was stopped. Its local files remain outside this report and outside Git. Earlier
PyTorch results are explicitly excluded from the Vulkan feasibility decision.

## Evidence and reproduction

`runs.json` retains raw timing samples and settings; `summary.json` removes the
samples. `provenance.json` records revisions and artifact hashes.
`validation.json` records the four resident execution checks. No application,
libplacebo algorithm, dependency pin, model or shader was changed.

See [the benchmark commands](../../README.md#persistent-native-4k-ncnnvulkan-evaluation).
The local speed report is
`/home/nick/.cache/voidvideo-fruc-benchmark/ncnn-4k/index.html`.
Downloaded binaries and raw output buffers remain in that cache.

Sources: [VapourSynth RIFE ncnn/Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan),
[ncnn SDK 20250503](https://github.com/Tencent/ncnn/releases/tag/20250503).
