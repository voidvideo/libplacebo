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

Run wall times include startup, file handling, and encoding. RIFE-ncnn currently
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
