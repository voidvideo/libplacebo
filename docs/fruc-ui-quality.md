# Vulkan FRUC UI reconstruction experiment

This note records the first motion-quality experiment for developers working on
scrolling UI distortion. The candidate remains local and is not installed in
VoidVideo or published as a dependency revision. It does not complete motion
vector repair or occlusion-aware image reconstruction.

## Evidence and implementation

The inherited static-region guard compares colors at the same pixel in both
input images. Equal colors can also occur in a moving repeated pattern. A
controlled test translates repeated strokes by 48 pixels: both endpoints show
background at a pixel where the intermediate image must show a stroke. The
original shader fails that test.

The candidate checks each reconstructed trajectory against the reverse flow
at its displaced endpoint. It also compares RGB colors at those corresponding
source positions. These residuals weight the two warped samples. The guard
allows the motion reconstruction through only when both trajectories have low
geometric and color residuals.

The weights stay positive and normalized at every interpolation position.
They use a one-pixel round-trip error scale and the inherited 0.03 color scale.
Guard validation transitions across squared motion errors of 0.5 to 2.0 and
mean absolute RGB differences of 0.01 to 0.03. These are experimental heuristics,
not NVIDIA-provided settings. The shader still uses the original six damped
inverse-warp iterations. It does not repair inconsistent vectors or construct
explicit visibility masks. Those limitations remain visible around object edges.

The implementation is original Vulkan shader code. No MVTools code, NVIDIA
runtime library, CUDA dependency, extra source frame, or presentation change is
included.

## Reference implementations

- [NVIDIA's FRUC guide](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvfruc-programming-guide/index.html#inside-fruc-library)
  separates flow validation, vector repair, interpolation, and image hole filling.
- [MVTools MaskFun.cpp](https://github.com/dubhatervapoursynth/vapoursynth-mvtools/blob/master/src/MaskFun.cpp)
  constructs time-dependent occlusion masks from neighboring motion vectors.
- [FFmpeg minterpolate](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_minterpolate.c)
  uses matching error to adjust overlapped motion-compensation weights in AOBMC.

These references guide the investigation. The candidate does not reproduce their
complete algorithms and is not equivalent to NVIDIA's NvFRUC library.

## Synthetic evidence

The fixture contains 12 source frames at 24 FPS and 28 output targets at 60 FPS.
Only the 22 intermediate targets are scored. Known images at the requested times
provide the reference. RGB mean absolute error is measured on a 0–255 code-value
scale; it is not a percentage of perceptual quality.

| Region | Original | Candidate |
|---|---:|---:|
| Scrolling scene, whole image | 4.201 | 2.903 |
| Scrolling text | 9.814 | 6.612 |
| Stationary header in scrolling scene | 0.237 | 0.215 |
| Selection scene, whole image | 6.510 | 6.267 |
| Selection-scene menu text | 16.105 | 15.362 |
| Selection-scene moving panel | 4.389 | 4.480 |

The moving-panel region regresses by about 2.1%. Visible artifacts remain. The
selection fixture includes abrupt focus changes; interpolation cannot infer
their exact timing from endpoint images alone. These measurements support further
experimentation, not a claim that Android TV scrolling is fixed.

The GPU tests cover repeated strokes, inconsistent motion, the original static
guard, fixed translation, hardware optical flow, cache invalidation, and prepared
24-to-60 rendering. The original shader fails the repeated-stroke regression;
the candidate passes. `fruc.c` and `renderer_prepare.c` pass on the local NVIDIA
GPU with Vulkan validation. VoidVideo's `render_loop_playback_tests --fruc` also
passes when explicitly loaded with the candidate library. This covers 15 output
timestamps and 12 intermediate images, plus the controlled transition tests.
Physical-display playback and Windows execution are not verified for this change.

## Reproduce the comparison

`tools/fruc-ui/generate.py` requires Pillow and the DejaVu Sans fonts at
`/usr/share/fonts/truetype/dejavu`. `evaluate.py` requires NumPy. `render.c`
requires the VoidVideo libplacebo fork headers and NVIDIA Vulkan optical flow.
Compile it with assertions enabled against that SDK:

```sh
cc tools/fruc-ui/render.c -o /tmp/fruc-ui-render $(pkg-config --cflags --libs libplacebo) -lm
python tools/fruc-ui/generate.py /tmp/fruc-ui-fixtures
mkdir -p /tmp/fruc-ui-baseline /tmp/fruc-ui-candidate
```

For each library, run the renderer once per fixture, selecting that library with
the existing dynamic-loader mechanism. For example:

```sh
LD_PRELOAD=/absolute/path/to/candidate/libplacebo.so.3750000 /tmp/fruc-ui-render /tmp/fruc-ui-fixtures/scroll.rgba /tmp/fruc-ui-candidate/scroll.rgba
LD_PRELOAD=/absolute/path/to/candidate/libplacebo.so.3750000 /tmp/fruc-ui-render /tmp/fruc-ui-fixtures/selection.rgba /tmp/fruc-ui-candidate/selection.rgba
```

Repeat with the original library and the baseline output directory. Then compare:

```sh
python tools/fruc-ui/evaluate.py /tmp/fruc-ui-fixtures /tmp/fruc-ui-baseline /tmp/fruc-ui-candidate
```
