# Compare interpolation tools on menus

This benchmark helps VoidVideo developers compare existing interpolation tools
before choosing further mixer work. It does not change an interpolation algorithm,
install an application dependency, or establish real-display playback performance.

## Inputs and comparison tracks

`suite.py` generates six deterministic, 960×540 menu sequences: slow and fast
horizontal scrolling under a fixed outline, vertical scrolling, a moving focus
outline, crossing panels, and a stationary control. Each sequence contains 25
source images at 24 FPS. The manifest records input and reference image hashes.

The midpoint track requests 24 frames at `(source_index + 0.5) / 24` seconds.
The 24→60 track requests 60 frames at `output_index / 60` seconds. Both use
reference images rendered directly at those times. Original-frame timestamps are
excluded from the 24→60 quality score.

These fixtures test defined, continuous menu animations. They do not establish
the quality of Android TV capture, natural video, HDR, 1080p, or 4K output.

## Run a candidate

Use Python with NumPy and Pillow. Keep downloaded tools, models, environments,
fixtures, and outputs outside the source checkout. Each run needs a new output
directory so that previous output cannot contaminate a result.

```sh
python3 tools/fruc-ui/suite.py /absolute/path/to/suite
python3 tools/fruc-ui/bench_adapter.py ffmpeg /absolute/path/to/suite /absolute/path/to/results/ffmpeg-midpoint --track midpoint
python3 tools/fruc-ui/bench_adapter.py mvtools /absolute/path/to/suite /absolute/path/to/results/mvtools-midpoint --track midpoint --resource /absolute/path/to/mvtools.so
python3 tools/fruc-ui/bench_report.py /absolute/path/to/suite /absolute/path/to/results
```

MVTools needs VapourSynth. Its adapter uses full-range BT.709 YUV444P16, a
two-pixel subpixel grid, 16×16 blocks with eight-pixel overlap, and `FlowFPS` with
`mask=2`. The FFmpeg adapter uses `minterpolate` with AOBMC, bilateral motion
estimation, UMH search with radius 64, and variable-size block compensation.
All settings and actual commands appear in each run's `run.json`.

The `rife-ncnn` adapter takes an executable in `--resource` and a model directory
in `--model`. It requests exact fractional timestamps with `-s`. The `film`
adapter takes a TorchScript model in `--model`. The `rife` adapter takes the
Practical-RIFE checkout in `--resource`, with the downloaded release's
`train_log` directory inside that checkout. Both Python model adapters require
CUDA-enabled PyTorch. RIFE follows its upstream video adapter's right/bottom
zero-padding to multiples of 128. FILM follows its upstream centered zero-padding
to multiples of 64. Both crop that padding from the output. They do not change
model weights or algorithms.

The `bim` adapter takes the official BiM-VFI checkout in `--resource` and its
checkpoint in `--model`. It loads the inference component directly, uses five
pyramid levels, and leaves training-only package initializers unloaded. It passes
unpadded RGB to the model so that the model normalizes before its own padding.

The `gimm-port` adapter evaluates the maintained ComfyUI GIMM-VFI port separately
from the original repository. It imports the port's inference modules without
installing ComfyUI. Supply its checkout in `--resource` and a directory containing
`gimmvfi_r_arb_lpips_fp32.safetensors` and `raft-things_fp32.safetensors` in
`--model`. It uses the RAFT variant, FP32, `ds_factor=1`, and the port's 32-pixel
replicate padding. It loads the separate RAFT checkpoint as the port does.

The `libplacebo` adapter takes the compiled `render.c` executable in `--resource`
and an existing shared library in `--model`. It selects that library only for
the benchmark process. The helper reads all source frames and the explicit
timestamp file; it does not loop the input to manufacture missing frames.
The adapter records synchronized render duration, including optical flow and
dispatch but excluding output readback. This differs from the Python adapters'
model-only duration, which also excludes source upload.

Run each candidate again with `--track 24to60` and a different output directory
to test fractional-rate conversion. A tool that supports only midpoint output
must remain absent from that track unless it supplies native fractional timing.

## Interpret the report

`bench_report.py` writes `scores.json` and `index.html`. The HTML view shows
lossless PNGs at synchronized timestamps. Its playback button advances through
the scored frames at a fixed inspection speed; it is not a real-time playback test.
Use `--encode-video` to add compressed, 60 FPS previews of each candidate's actual
24→60 output. Missing tail frames remain missing in those previews. The enlarged
text crops and scores still use lossless PNGs.

Scores use common available timestamps across candidates in each scene and track.
FFmpeg can end before the final requested timestamp because of lookahead. The
adapter records `partial_output` and the actual frame count. It does not pad output.
Failed runs remain visible in the report and do not enter quality rankings.

Errors are RGB code values on a 0–255 scale. Edge error uses reference edges
expanded by two pixels. The text region includes the surrounding menu area.
Temporal error measures the change in reconstruction error across scored frames;
the 24→60 scoring intervals are unequal after original frames are excluded.
Inspect text, outlines, and occlusion boundaries before interpreting averages.

Run wall times include startup, file handling, and encoding. The original RIFE-ncnn adapter
starts a process per requested intermediate frame, so those wall times cannot
estimate integrated inference speed. Python model adapters also record synchronized
GPU inference duration, separating the first inference from subsequent samples.
Run candidates serially on an otherwise idle GPU before comparing performance.

## Candidate sources

- [MVTools](https://github.com/dubhatervapoursynth/vapoursynth-mvtools)
- [FFmpeg minterpolate](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_minterpolate.c)
- [RIFE-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- [FILM PyTorch conversion](https://github.com/dajes/frame-interpolation-pytorch),
  an unofficial conversion of Google's FILM model
- [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI)
- [GIMM-VFI ComfyUI port](https://github.com/kijai/ComfyUI-GIMM-VFI)
- [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI)
- [NVIDIA Optical Flow SDK](https://developer.nvidia.com/optical-flow-sdk)

Listing a candidate is not evidence that it ran successfully. Check its run status,
source revision, model checksum, and output count. Evaluation does not approve
shipping a candidate or importing its dependencies into VoidVideo.

## Native 4K RIFE evaluation

These historical PyTorch measurements are excluded from the ncnn/Vulkan
feasibility decision. Use the persistent ncnn/Vulkan evaluation below.

`bench_rife_4k.py` draws the same six menu layouts directly at 3840×2160.
Geometry rounds to native pixels and fonts rasterize at the target resolution.
It evaluates all 48 intermediate timestamps of each one-second 24→60 sequence
and saves twelve representative timestamps per scene. The other twelve output
timestamps coincide with source frames and need no inference.

Run each configuration in a separate process, serially on the GPU:

```sh
python tools/fruc-ui/bench_rife_4k.py /path/to/Practical-RIFE /path/to/results/full --scale 1
python tools/fruc-ui/bench_rife_4k.py /path/to/Practical-RIFE /path/to/results/half --scale 0.5
python tools/fruc-ui/bench_rife_4k.py /path/to/Practical-RIFE-lite /path/to/results/lite-half-fp16 --scale 0.5 --precision fp16
python3 tools/fruc-ui/report_rife_4k.py /path/to/results
```

Use the downloaded model's own `train_log` code and weights with the upstream
`model` helpers. The adapter validates all inference weights and records unused
training-only teacher/time-estimation weights. It uses upstream scale-dependent
padding, cuDNN algorithm selection, and the upstream FP16 default tensor type
when requested. Candidate algorithm files remain unchanged.

The continuous timing test warms up the model, then repeatedly interpolates a
resident fast-scroll frame pair at four fractional times for at least 15 seconds
and 48 inferences. It records every synchronized inference time, total throughput,
and PyTorch peak allocated/reserved GPU memory. It excludes input transfer, output
readback, capture, rendering, and presentation. The memory figures exclude other
processes and non-PyTorch driver allocations. This is an inference feasibility
test, not proof of sustained application playback or a long thermal soak.

The 24→60 cadence requires 48 generated frames per second. Report this generated
rate separately from the equivalent output rate, which also counts the twelve
source-coincident frames. Quality measurements run separately over changing frame
pairs. They use rounded RGB8 output and edge masks expanded by eight native pixels,
matching the two-logical-pixel region used by the smaller benchmark.


## Persistent native 4K ncnn/Vulkan evaluation

Use `ncnn_resident.cpp` for GPU-resident speed measurements.
`bench_ncnn_4k.py` supplies secondary full-filter timings. The PyTorch adapter
above measures a different runtime and cannot establish Vulkan feasibility.
Install NumPy, Pillow and VapourSynth in the test Python environment.
Load the upstream VapourSynth-RIFE-ncnn-Vulkan plugin and its matching model
exports. The recorded run used release `r9_mod_v33` and models from revision
`c3ec6aabc07c8fa37a4f58d7fed9e2ad1fc1b13f`.

```sh
python tools/fruc-ui/bench_ncnn_4k.py /path/to/librife.so /path/to/model /path/to/results/timing/model-t2 --workers 2 --mode timing
python tools/fruc-ui/report_ncnn_4k.py /path/to/results
```

Each output directory must be new. Test worker counts 1, 2 and 3 in separate
processes. Run GPU tests serially. Timing disables frame caches and requests
unique generated output indices. It includes CPU packing, GPU transfers,
inference and readback. Request latency includes queueing; it is not network-only
latency. Total board memory includes the desktop and other processes.
Scene detection, frame skipping, TTA and UHD mode are explicitly disabled.
This adapter now runs speed tests only. Earlier quality work remains in the
local cache and is excluded from this speed report.

If a process crashes, its initial `running` record is incomplete evidence.
Record the process exit code and mark the run failed before reporting results.
Never count an incomplete run as successful or reuse its directory.

`ncnn_resident.cpp` measures a separate boundary: a persistent model with input
frames and timestep maps already on the GPU. It uses upstream `Warp` unchanged.
Build against the Vulkan-enabled ncnn 20250503 shared SDK; the upstream Warp API
is incompatible with the 20261020 SDK. Set `sdk` to the extracted SDK directory
and `rife` to the upstream checkout:

```sh
g++ -std=c++17 -O3 -fopenmp tools/fruc-ui/ncnn_resident.cpp "$rife/RIFE/warp.cpp" -I"$rife/RIFE" -I"$sdk/include/ncnn" -L"$sdk/lib" -Wl,-rpath,"$sdk/lib" -lncnn -o /path/to/ncnn-resident
/path/to/ncnn-resident /path/to/model /path/to/pair.f32 /path/to/output.f32 /path/to/results/resident/model/run.json
```

Create the resident output directory first. The input file contains two planar
RGB float32 frames, each 3840×2160 with values normalized to 0–1, in source order.
For the recorded comparison these are `scene(10, 'scroll_fast', 4)` and
`scene(11, 'scroll_fast', 4)` from `suite.py`. The harness applies model padding
and FP16 storage. It warms up nine times, then cycles fractions 0.4, 0.8, 0.2,
and 0.6 for at least fifteen seconds and 48 inferences. Each inference submits
and waits for completion. The timed interval excludes upload and download.
The final output file contains the cropped planar float32 result at fraction
0.4. The final download occurs after the timer stops and serves only to check
that the benchmark produced a finite image.

The SDK and prebuilt plugin are distinct builds. Report their measurements
separately; their difference cannot be attributed solely to transfer costs.
Neither test establishes live playback performance or authorizes integration.
