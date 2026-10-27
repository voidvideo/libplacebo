# Synthetic menu evaluation, 2026-10-27

MVTools has the lowest average edge error in this suite. BiM has the lowest
24→60 error on both horizontal scroll scenes. RIFE 4.25 has the lowest 24→60
error on the vertical scene. No tested candidate eliminates the visible outline
and text distortions in fast scrolling. These results do not establish a
replacement suitable for shipping.

Each row represents one tested configuration, not the best possible tuning of
that tool. RIFE 4.6 and 4.25 use different models as well as different backends;
their difference cannot be attributed to Vulkan versus CUDA.

Only the benchmark framework changed during this evaluation. Candidate algorithm
sources and weights were not modified. Nothing was integrated into VoidVideo.

## Method

The machine has an RTX 5070, driver 615.71.09, and 12227 MiB VRAM. Six deterministic
960×540 scenes contain 25 input images at 24 FPS. References are rendered directly
at each target time. See the [framework instructions](../../README.md) for the
scene definitions, adapters, and metrics.

Eight configurations produced images for both midpoint and native 24→60 tracks.
FFmpeg emitted 23 of 24 midpoint frames and 58 of 60 native frames because of EOF
lookahead. All tools are therefore scored on the same 23 midpoint frames or
46 native intermediate frames per scene. Original-frame timestamps are excluded
from native quality scores. No missing frames were padded.

## Edge-error comparison

Values are RGB code values on a 0–255 scale; lower is better. Each value below is
an equal-weight average of the six scene means. This is a diagnostic ranking,
not a perceptual quality percentage.

| Configuration | Midpoint | 24→60 |
| --- | ---: | ---: |
| MVTools 29, FlowFPS mask 2 | 2.394 | 4.093 |
| BiM-VFI, five pyramid levels | 2.830 | 4.292 |
| GIMM-VFI RAFT, maintained ComfyUI port, FP32 | 3.162 | 4.694 |
| RIFE 4.25, official PyTorch release | 3.880 | 6.250 |
| FILM, unofficial PyTorch conversion, FP32 | 4.128 | 7.233 |
| FFmpeg 7.1.5 minterpolate, AOBMC/UMH | 4.857 | 8.622 |
| RIFE 4.6, ncnn Vulkan | 4.945 | 6.041 |
| Original FFmpeg Vulkan port snapshot | 5.777 | 6.332 |

MVTools leads the native focus-outline, crossing-panel, and stationary scenes.
BiM leads native slow and fast horizontal scrolling; RIFE 4.25 leads native
vertical scrolling. The ranking changes between midpoint and fractional output,
so a frame-doubling result cannot stand in for a 24→60 result.

Visual inspection of fast-scroll midpoint frame 10 shows deformed cyan outlines
or damaged text in every inspected family. The original Vulkan port loses much
more text detail in that frame. Lower average error does not mean a tool keeps
every menu element stable.

## Timing and execution limits

Native model-only median durations, excluding the first inference, were about
8.1 ms for RIFE 4.25, 128.4 ms for BiM, 167.0 ms for FILM, and 592.8 ms for the
GIMM port. These synchronized measurements exclude upload and readback. They
are indicative offline measurements, not application frame budgets. Some runs
overlapped other benchmark activity; this was not an isolated performance lab.

The original Vulkan port's separate render measurement had a 0.264 ms median
and 5.140 ms p95. That path includes optical flow and cache reuse, and has a
different measurement boundary from the Python models. Do not compare these
numbers as equivalent pipeline costs. RIFE-ncnn starts a process per output;
its wall time cannot estimate integrated inference speed.

The NVIDIA Optical Flow SDK 5.0.7 sample aborted inside the proprietary library
before producing scored output. Both RGB and NV12 attempts failed. The stack
includes `CLogger::LogMsg`, `NvGPURGB2Y::Process`, and `NvOFFRUCPreProc::Execute`.
The cause is unresolved; NVIDIA quality remains unmeasured. The backtrace is
retained in this directory.

The original GIMM repository failed to import on Python 3.11 because of mutable
dataclass defaults. The maintained ComfyUI port ran successfully and is explicitly
identified as a separate implementation. Its inference modules ran without a
ComfyUI installation.

## Evidence and validation

`scores.json` contains every scene metric and scored index. `runs.json` records
settings, statuses, timings, hashes, and representative commands. `provenance.json`
records source revisions, model hashes, and the Python package environment.
`output-validation.json` records frame counts and source-coincident errors.

All 16 image-producing runs passed PNG readability, 960×540 dimension, contiguous
index, and recorded-count checks. At native original-frame timestamps, all tools
reproduced the source exactly except FFmpeg's RGB/YUV conversion, whose largest
per-frame mean absolute error was 0.5005 code values. Identity scoring returned
zero; cached edge masks reproduced previous metrics exactly. The HTML comparison
was rendered and inspected in Chrome.

Initial generic-padding RIFE, FILM, and BiM runs were excluded and preserved under
`superseded-preprocessing` in the local cache. The results here use reruns with
upstream preprocessing: RIFE right/bottom 128 alignment, FILM centered 64 alignment,
and BiM's internal padding after normalization.

Full PNG sequences, run command logs, lossless comparison HTML, compressed 60 FPS
previews, and downloaded tools remain outside Git at
`/home/nick/.cache/voidvideo-fruc-benchmark`. Open `results/index.html` there.
Video previews are for viewing only; scores and enlarged crops use PNGs.

This evaluation does not cover actual Android TV capture, 1080p/4K, HDR, live
fullscreen playback, or an integrated Vulkan implementation of the neural models.
NVIDIA's failed run is an outstanding comparison, not a poor quality score.
