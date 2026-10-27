# Native 4K Practical-RIFE evaluation

**PyTorch-only historical results. Excluded from the Vulkan feasibility decision.**
The required deployment runtime is ncnn/Vulkan. These measurements do not assess
that runtime and must not be used to reject real-time Vulkan interpolation.
They remain here only to preserve the original experiment and its limitations.

None of the six tested configurations meets the inference throughput needed for
3840×2160, 24→60 conversion on this RTX 5070. The fastest is RIFE 4.25 Lite,
half-scale estimation, FP16, at 19.47 generated frames/s. The cadence needs 48
generated frames/s. This configuration reaches about 24.33 equivalent output
frames/s when counting the twelve source-coincident timestamps per sixty outputs.
It would need roughly 2.47 times its measured generation throughput just to meet
the inference workload, before the rest of the video pipeline.

This conclusion applies to these PyTorch/CUDA configurations. It does not measure
an optimized TensorRT or Vulkan implementation, other models, or another GPU.

## Results

All inputs and outputs are 3840×2160. `half` means internal motion-estimation
scale 0.5; `full` means scale 1.0. Configurations without an `fp16` suffix use FP32.
Peak tensor memory is PyTorch allocated memory, not total board VRAM use. Edge
error is an equal-weight average of the six scene means on a 0–255 RGB scale;
lower is better and the number is not a perceptual quality percentage.

| Configuration | Generated FPS | Median ms | p95 ms | Maximum ms | Peak tensor GiB | Edge error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rife425-lite-half-fp16 | 19.47 | 51.21 | 52.12 | 52.35 | 3.03 | 11.878 |
| rife425-half-fp16 | 18.49 | 53.90 | 54.82 | 55.80 | 3.04 | 11.515 |
| rife425-lite-half | 10.94 | 91.28 | 92.02 | 92.26 | 6.18 | 9.829 |
| rife425-half | 10.43 | 95.67 | 96.62 | 97.25 | 6.49 | 9.445 |
| rife425-lite-full | 8.86 | 113.03 | 113.40 | 114.63 | 7.12 | 6.096 |
| rife425-full | 7.20 | 133.94 | 159.62 | 177.17 | 7.96 | 7.072 |

The saved fast-scroll images show damaged text and deformed stationary cyan
outlines in both Lite half-scale variants. FP16 increases speed but does not
resolve those artifacts. Full-scale settings preserve some details better, but
no tested setting provides both the required throughput and clean menu output.

## Method and limits

The GPU was an RTX 5070 with driver 615.71.09 and 12227 MiB total VRAM. Runs used
PyTorch 2.7.1+cu128. All six configurations ran serially. The desktop remained
active; this was not an exclusive GPU environment.

The framework draws geometry and font glyphs directly at 4K and rounds positions
to native pixels. It does not resize 540p bitmaps. Source images follow six defined
menu animations at 24 FPS. References are rendered at exact output times. Each
configuration scores all 48 intermediate timestamps per scene: 288 images total.
Twelve output timestamps per second coincide with sources and need no inference.
The viewer retains twelve sampled intermediate timestamps per scene, with their
references. These synthetic menus are not an Android TV capture.

The resident timing test uses a fast-scroll source pair already on the GPU. After
one cold inference and eight warmups, it alternates the four fractional times
0.4, 0.8, 0.2, and 0.6 for at least fifteen seconds and 48 inferences. Each inference
is synchronized. Throughput uses total elapsed wall time; latency statistics use
the recorded per-inference durations. Quality runs use changing source pairs and
record their inference durations separately.

Timing excludes source uploads, output readback, capture, rendering, and
presentation. It is not a live fullscreen test or a long thermal soak. The tested
serial inference path already misses the target before these additional costs.
PyTorch allocated/reserved memory excludes other processes and driver allocations.

The framework uses upstream scale-dependent zero padding and cuDNN algorithm
selection. FP16 follows upstream default tensor precision, including warp grids.
Model code and weights remain unchanged. Every inference weight loads strictly;
known unused teacher/time-estimation training weights are recorded separately.
No candidate was integrated into VoidVideo and no application dependency changed.

## Validation and evidence

All six runs completed. Every run has 288 finite score records at the expected
timestamps, 72 saved candidate PNGs, and 72 matching reference PNGs. All saved
images passed dimension and integrity checks. Reference hashes match across all
configurations. A separate CPU calculation from saved PNGs reproduced the GPU
quality metrics within 0.0001 RGB code values. Existing checked 540p input fixtures
remain pixel-identical after adding native-resolution rendering.

`runs.json` contains raw timings, per-frame quality, model/code hashes, status,
precision, padding, and memory. `summary.json` omits individual timing and quality
samples for easier reading. `provenance.json` records the original model downloads,
archive hashes, upstream revision, dependency versions, and framework commit.
`validation.json` records output checks.

The first full-model smoke run used logical-pixel geometry rounding and disabled
cuDNN algorithm benchmarking. It is excluded from this matrix and preserved in
`4k-initial-fixture-smoke` outside Git. A loader validation attempt is also retained
separately. All six reported runs use the corrected native-pixel fixture and
upstream cuDNN setting.

Open `/home/nick/.cache/voidvideo-fruc-benchmark/4k/index.html` for the local visual
comparison. Full-size PNGs, logs, and downloaded artifacts remain in that cache.
See [the framework instructions](../../README.md#native-4k-rife-evaluation) to
reproduce a configuration. The original model release and supported scale/FP16
options are documented in [Practical-RIFE](https://github.com/hzwer/Practical-RIFE).
