# Native 4K TensorRT and concurrent Vulkan speed

TensorRT RIFE 4.4 meets the measured inference-throughput requirement for native
3840×2160, 24→60 on this RTX 5070. The one-stream run achieved 51.42 generated
frames/s; a separate 60.15-second confirmation achieved
50.94 generated frames/s (3064 completed inferences).
The target is 48 generated frames/s because twelve of the sixty output timestamps
coincide with source frames. Inputs and outputs stay on the GPU during timing.

This changes the feasibility finding: the earlier serial ncnn results did not
cover TensorRT. The passing result establishes backend inference throughput,
not libplacebo integration or live playback latency. The measured margin above
48/s is limited.

## TensorRT results

All rates are completed generated frames/s. Engine compilation, initialization,
upload and readback are excluded. Each matrix run lasts at least fifteen seconds
after warmup. Models are the unchanged upstream V2 ONNX exports linked by SVP.

| Model | 1 stream | 2 streams | 3 streams |
| --- | ---: | ---: | ---: |
| rife44 | 51.42 | 50.17 | 50.01 |
| rife46 | 41.94 | 40.55 | 40.28 |
| rife425-lite | Failed check | Failed check | Failed check |
| rife425 | 26.00 | 25.28 | 25.12 |

RIFE 4.25 Lite failed the basic finite/nonconstant-output assertion for all three
stream counts. Those runs have no valid throughput result. The assertion did not
record which condition failed; the cause remains unresolved. This was an
execution-validity check, not an image-quality evaluation. No quality suite ran.

## Concurrent ncnn/Vulkan comparison

RIFE 4.4 measured 17.58, 18.10 and 18.01 generated frames/s with one, two and three
independent workers respectively. All outputs passed the execution check.
These runs also keep inputs and outputs resident on the GPU throughout timing.
The exported models and runtime execution strategies differ between backends;
the results compare those configurations rather than isolating a compiler change.

## Timing method and limits

TensorRT 10.16.1.11 uses FP16, disables TF32, and uses CUDA graphs with separate
execution contexts and CUDA streams. Each stream queues eight inferences per
batch, cycling fractional times 0.4, 0.8, 0.2 and 0.6. Every batch completes on
all streams before its frames are counted. Throughput uses total wall time,
including submission and synchronization. Queued throughput is not a latency
measurement. Input and output dimensions remain 3840×2160; padding is internal
to the V2 models. No input downscaling is used.

Engine builds use a fixed 4K profile, optimization level 3, a 4 GiB workspace
limit and upstream FP32 constraints for applicable floating-point layers.
Integer shape/index operations retain their required types. Build duration is
recorded separately. The ncnn harness shares model weights and resident inputs,
uses separate worker allocators, and starts workers together after warmup.
Each worker submits and waits independently.

The GPU is an RTX 5070 with driver 615.71.09 and 12227 MiB VRAM. Engine builds and
benchmarks ran sequentially. The desktop and VoidVideo remained active; this was
not an exclusive GPU environment. The one-minute confirmation is not a long
thermal soak or a live video pipeline test.

## Evidence and reproduction

`runs.json` contains raw samples; `summary.json` contains concise records.
`validation.json` checks completed frame counts, elapsed times, calculated rates
and output-validity flags. `provenance.json` records model/archive hashes,
runtime versions, compiler settings and engine-build records. Two initial
harness setup errors were corrected before building the measured RIFE 4.4 engine;
the failed setup records are retained and excluded from speed results.

See [the commands](../../README.md#gpu-resident-tensorrt-and-concurrent-vulkan-speed-tests).
The local report is
`/home/nick/.cache/voidvideo-fruc-benchmark/gpu-speed/index.html`.
Model downloads, engine caches, logs and earlier harness snapshots remain outside
Git in that cache. Changes are limited to benchmark tools and evidence; no
application or interpolation algorithm was modified.

Sources: [SVP RIFE settings](https://www.svp-team.com/docs/rife-ai/),
[SVP-linked ONNX exports](https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models),
[vs-mlrt runtime configuration](https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py).
