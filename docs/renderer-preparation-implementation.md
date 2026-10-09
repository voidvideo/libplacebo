# Temporal preparation implementation record

<!-- Audience: fork maintainers. Goal: locate the code and evidence for temporal preparation and policy changes. Type: reference. Evidence: current source snapshot and local test logs. -->

See [the preparation contract](renderer-preparation.md) for API behavior.

Baseline commit: `c81411806025a692091637d030fdaccef3e804d8`. The working tree also contains earlier async coverage changes. The diff spans below describe each whole file against that baseline; they are not exclusive attribution to this feature. Hashes identify this exact source snapshot. No API/SONAME version bump was made (374).

## Source locations

### `src/include/libplacebo/renderer.h`

SHA-256: `3a46654ffea76d94d748797abfcfaf9ee143cdf0dda9ea428883b4d052412748`

- `enum pl_renderer_prepare_mode {`: line 932.
- `struct pl_renderer_snapshot {`: line 948.
- `pl_renderer_prepare_mode(pl_renderer_preparation`: line 984.
- `pl_renderer_describe_image_mix(`: line 1009.

Changed spans (`-baseline +current`): `-264,4 +264,10`; `-924,2 +930,14`; `-934,0 +953`; `-949 +968,2`; `-953,0 +974,4`; `-955,0 +980,5`; `-958 +987`; `-962 +991,2`; `-970,9 +1000,9`.

### `src/renderer.c`

SHA-256: `8449a0d216abed1b274fb9193cadda34f24f744f5f1917e263f664788f19fed7`

- `static struct pl_filter_config mix_filter`: line 4527.
- `static bool mix_frame_weight`: line 4549.
- `static bool preparation_admit`: line 5114.
- `static enum pl_renderer_prepare_result preparation_traverse`: line 5148.
- `static bool preparation_snapshot_valid`: line 5310.
- `static enum pl_renderer_prepare_result describe_synchronous`: line 5317.
- `static enum pl_renderer_prepare_result temporal_output`: line 5375.
- `static enum pl_renderer_prepare_result describe_temporal_mix`: line 5436.
- `enum pl_renderer_prepare_result pl_renderer_describe_image_mix`: line 5481.
- `enum pl_renderer_prepare_result pl_renderer_prepare_submit`: line 5507.
- `enum pl_pass_prepare_state pl_renderer_prepare_poll`: line 5551.
- `enum pl_renderer_prepare_mode pl_renderer_prepare_mode`: line 5595.
- `void pl_renderer_prepare_destroy`: line 5606.
- `static enum pl_renderer_prepare_result render_temporal_mix`: line 5685.

Changed spans (`-baseline +current`): `-119,0 +120,6`; `-129,0 +136`; `-135,3 +142,6`; `-139 +149`; `-546 +556,2`; `-550 +561`; `-557 +568`; `-561 +572,2`; `-565 +577,2`; `-589 +602`; `-595 +608`; `-981 +994`; `-1033 +1046`; `-1144,0 +1158`; `-1151,0 +1166,2`; `-1155 +1171`; `-1238,2 +1254,2`; `-1328 +1344`; `-1337 +1353`; `-1352 +1368,4`; `-1361,0 +1381,28`; `-1488,2 +1534,0`; `-1491,0 +1537,2`; `-1790 +1836,0`; `-1793,0 +1840,2`; `-1803,0 +1852`; `-1870,2 +1919,2`; `-1872,0 +1922,2`; `-1881,0 +1933`; `-2686 +2738`; `-2693 +2745`; `-2695 +2747`; `-2697 +2749`; `-2699 +2751`; `-2701 +2753`; `-2703 +2755`; `-2706,2 +2758,3`; `-2712,4 +2765,20`; `-2718 +2787`; `-2720,4 +2789,4`; `-2728 +2797,2`; `-2737,5 +2807,5`; `-2745 +2815`; `-2750 +2820`; `-2791,19 +2860,0`; `-2870,2 +2921,5`; `-2873,3 +2927,18`; `-2877 +2946`; `-2881,0 +2951`; `-2883 +2953,2`; `-2925,5 +2996,25`; `-2938,4 +3029,4`; `-2944 +3035`; `-2947 +3038,4`; `-2952 +3046,2`; `-2961 +3056,2`; `-2965 +3061`; `-2969 +3065`; `-2982,2 +3078,2`; `-2985,3 +3081,13`; `-2996,3 +3102,3`; `-3005 +3111,2`; `-3023 +3130`; `-3025,2 +3132,5`; `-3031,2 +3141,2`; `-3043 +3153,2`; `-3064 +3175`; `-3066,2 +3177,5`; `-3072,2 +3186,2`; `-3080 +3194`; `-3105 +3219,5`; `-3119 +3237,2`; `-3122 +3241`; `-3127 +3246`; `-3145 +3264,2`; `-3158 +3278`; `-3162 +3282,5`; `-3217 +3341`; `-3221,2 +3345,2`; `-3227,0 +3352,4`; `-3229 +3357`; `-3282,2 +3410,2`; `-3318,0 +3447`; `-3320 +3449,2`; `-3432 +3562`; `-3469 +3598,0`; `-3481 +3610`; `-3494 +3623`; `-3599 +3728`; `-3605 +3734`; `-4108 +4237`; `-4397 +4526,76`; `-4447,15 +4651`; `-4480,47 +4670,2`; `-4643,22 +4788`; `-4821,2 +4945,3`; `-4853,2 +4978,2`; `-4867 +4992`; `-4870 +4995,5`; `-4873,0 +5003,6`; `-4983 +5118`; `-4985,9 +5120,3`; `-5026,2 +5155,3`; `-5040,12 +5169,0`; `-5053 +5171`; `-5055 +5173,4`; `-5058 +5179,2`; `-5062,0 +5185,7`; `-5089,0 +5219,3`; `-5119 +5251,3`; `-5124,0 +5259,2`; `-5150 +5286`; `-5162 +5298`; `-5163,0 +5300`; `-5172,0 +5310,23`; `-5177 +5337,11`; `-5204,0 +5375,106`; `-5210,0 +5487,2`; `-5214 +5492`; `-5216 +5494,11`; `-5225,0 +5514,7`; `-5242 +5537,7`; `-5256,0 +5558,8`; `-5260,0 +5570,10`; `-5275,0 +5595,6`; `-5285,0 +5611,2`; `-5294 +5621,2`; `-5321,0 +5650,5`; `-5341,0 +5675,7`; `-5344,0 +5685,122`; `-5349,0 +5812,9`; `-5360,0 +5832,9`.

### `src/tests/renderer_prepare.c`

SHA-256: `3724dcbad3d7e69db68f0593b92522c2e21e6a66acc052e3c856bb8ab4791874`

- `static void test_preparation_policy`: line 672.
- `static void test_temporal_mix`: line 777.

Changed spans (`-baseline +current`): `-4,0 +5`; `-44,0 +46,12`; `-215,0 +229`; `-227,0 +242,4`; `-228,0 +247,85`; `-332,0 +436,2`; `-333,0 +439,22`; `-397,0 +525`; `-426,0 +555,11`; `-526,0 +666,304`; `-532,0 +976,6`; `-541 +990`; `-554 +1003`; `-607,0 +1057,29`; `-700,0 +1179,282`; `-718,0 +1479,2`; `-726,0 +1489,34`; `-730,0 +1527,17`; `-746,0 +1560,6`; `-747,0 +1567,5`; `-750,0 +1575`; `-788,0 +1614,4`.

## Ownership and implementation decisions

`describe_temporal_mix` owns one renderer shared by its source preparations. Child destruction releases child resources without destroying that renderer; parent destruction releases the renderer after all children. This preserves shared peak-detection history while giving each source an independently prepared graph and cache texture.

`render_temporal_mix` selects native weights, reserves cached signatures, preflights every contributor and the chosen output variant, then executes. Runtime frame counters advance for cache population and output exactly as in native mixing. Eligible signatures determine eviction even when their weights fall below the cutoff.

`describe_synchronous` retains a separate ordinary renderer without invoking it. AUTO selects this path only when asynchronous description returns UNSUPPORTED. The effective-mode query makes the loss of strict execution guarantees inspectable. ASYNC remains zero for existing callers. SYNC submission and polling create no worker requests.

## Executed evidence — October 6, 2026

- Full library suite: 21/21 passed (`/tmp/temporal-policy-full-tests.log`).
- Additional policy edge tests: renderer suite passed (`/tmp/policy-edge-tests.log`), including invalid mode, failed description without fallback, and empty mix fallback.
- Temporal dithering regression: failed before cache eviction/counter correction (`/tmp/temporal-dither-debug.log`); passed afterward (`/tmp/temporal-eviction-after.log`).
- HDR regression: failed with separate peak histories (`/tmp/temporal-hdr-before.log`); passed with shared renderer state (`/tmp/temporal-hdr-after.log`).

These are local evidence files, not committed CI artifacts. The source tests reproduce the checks. Delayed-peak temporal mixing, heterogeneous source variants, temporal grain combinations, and backend-wide parity are not established by this run.

## Live mastering-peak regression — October 6, 2026

Live HDR playback with delayed peak detection repeatedly passed preflight, then rejected execution because the measured peak enabled a previously absent mastering-peak soft-clip block. The application rebuilt candidates and cleared pending output repeatedly. AUTO did not apply because this was a runtime variant mismatch rather than unsupported description.

`src/shaders/colorspace.c:2185` keeps the block present when prepared adaptive peak detection and mastering clipping are enabled. Its strength remains a dynamic uniform, including zero before a valid measurement. The disabled-strength path uses a valid normalization denominator when mastering metadata is absent. Ordinary rendering retains its existing branch optimization. Source SHA-256: `f1983f000ea7e1fcd23cc7faedb74ccac4d3bb62d0d595bdd18552a4f80c5613`.

`src/tests/renderer_prepare.c:1287` now supplies explicit 1000-nit mastering metadata and enables mastering clipping for the native-color tests. This reproduces the initial measurement transition with immediate and delayed detection. The test failed before the fix (`/tmp/mastering-clip-before.log`) and passed afterward (`/tmp/mastering-clip-after.log`).

A 35-second live run using the existing `/tmp/voidvideo-jz-release/config.msgpack` and DeckLink input completed without the recurring preparation loop after startup (`/tmp/void-auto-live-fixed.log`). This is log-based playback verification; it does not assert inspection of every displayed frame. No saved settings were changed to avoid the failure.

## Mutable prepared gamut LUT content — October 6, 2026

The gamut LUT was treated as immutable after staging, so changing color metadata rejected it even when the texture layout was unchanged. `sh_lut` now accepts content-signature changes only for explicitly mutable prepared LUTs; format, dimensions, component type/count, and interpolation checks remain enforced. Description updates CPU data, and execution uploads it into the existing texture.

`pl_shader_color_map_ex` retains a prepared-gamut flag, requests mutable LUT staging on this path, and emits gamut-domain scaling as dynamic uniforms. Ordinary gamut LUT allocation remains unchanged. The consumer's `RenderPipeline::request_generation` no longer overrides the requested `dynamic_constants` setting to false; normal renderer settings already enable it. Thus metadata-dependent coefficients can remain uniforms rather than forcing a specialized shader.

`test_lut` verifies content-signature refresh, unchanged texture identity, no GPU upload during preflight, and continued rejection of layout changes. `test_native_color` adds perceptual gamut mapping with varying mastering primaries and destination peak metadata, compares every output to ordinary rendering, and forbids runtime shader compilation. Logs: `/tmp/gamut-update-before.log`, `/tmp/gamut-dynamic-after.log`, `/tmp/gamut-library-tests.log` (21/21 passed).

Updated source identities:

- `src/shaders/lut.c`: SHA-256 `23f42e1c5adf3274688ff0bb025caaa255384703d3958040048be66af40f42d1`.
- `src/shaders.h`: SHA-256 `9b02f68287ef6f8d07a404b23ba464a107aa2c616955b6155ad655170df89b42`.
- `src/shaders/colorspace.c`: SHA-256 `a51db96d46110f97039dbeb4789e2246f0c920aac7d94afce9c1c1296099af54`.
- `src/tests/lut_prepare.c`: SHA-256 `569198b2f2b06d244f0c4db973183c536a62458ab5f7928792fda1f79d7bd8f6`.
- `src/tests/renderer_prepare.c`: SHA-256 `3724dcbad3d7e69db68f0593b92522c2e21e6a66acc052e3c856bb8ab4791874`.

## Ordered async coverage follow-up (2026-10-06)

This record supersedes earlier source hashes for the files below. Source windows
and concurrent contributors are distinct: source storage is allocated for the
supplied window, while output variants retain the native 16-contributor limit.

| Priority | Verified library behavior | Remaining boundary |
| --- | --- | --- |
| 1 — resources and adoption | LUT staging-upload failure preserves the active resource and retries; superseded CPU LUT contents publish only the latest owned data. A compiler barrier proves active rendering survives pending, cancelled and failed replacements; a ready replacement changes pixels without mutating the old handle. | Host service deadlines and atomic publication across application resources are consumer contracts, not libplacebo service APIs. |
| 2 — lifecycle transitions | Same-layout source rebinding works; absent source and incompatible target shape cannot execute an existing populated graph or mutate analysis. Restoring valid input recovers. Snapshot references return to the caller after cancellation/destruction. | Live device replacement, capture signal recovery and display recreation remain integration tests. |
| 3 — acquire/release | Shared `pass_acquire_frames` preserves native ownership. Execution revalidates acquired bindings; success, failed acquisition and invalid geometry release target/source/neighbor frames. Description and preflight have no acquisition callbacks. | Callback-provided metadata must have a valid upfront description. A later source failure cannot roll back earlier GPU submissions. |
| 4 — temporal mixing | All exposed non-null mixer presets plus Mitchell match ordinary pixels. Sources acquire/release sequentially; a 17-frame window is accepted with fewer simultaneous contributors. Existing changing-weight/cache/dither/HDR tests remain. | Every heterogeneous-layout/feature combination is not proven by these fixtures. |
| 5 — absent/empty rendering | Shared `pass_empty_overlays` handles null source and clipped-empty target. All clearing modes, with/without target overlays, match ordinary pixels. Empty mixes prepare asynchronously. | These GPU fixtures do not substitute for live no-signal UI tests. |
| 6 — hooks | Existing policy tests verify ASYNC/AUTO/SYNC for explicit preparation hooks and legacy hooks, including failure reporting. Existing MPV corpus covers prepared shader constructs. | Arbitrary hooks still need explicit describe/execute-prepared callbacks for strict async; AUTO may use synchronous rendering. |
| 7 — Dolby Vision | Existing metadata fixture compares base-layer decode and NLQ enhancement-layer composition against native rendering. Disabling the enhancement layer must change pixels. Enhancement sampling uses the shared prepared builder and texture metadata transport. | Real content/profile combinations and enhancement acquisition failure need wider coverage. |

Positive frame-access, empty-rendering, Dolby Vision and wide-window cases were
run before lifting their admission gates and failed. The final configured
libplacebo suite passed 21 tests, zero failures (`/tmp/lib-async-ordered-full.log`).
Tests are in `src/tests/renderer_prepare.c` and `src/tests/lut_prepare.c`; no
frozen production images or plugin-specific radial fixtures were introduced.

### Current source snapshot

Diff counts below include all existing changes against HEAD, not only this
follow-up. Function anchors identify the relevant implementation.

| File | Current anchors | SHA-256 | Added / removed against HEAD |
| --- | --- | --- | --- |
| `src/renderer.c` | `sample_el`:2069; `pass_acquire_frames`:4112; `pass_empty_overlays`:4214; `describe_temporal_mix`:5452; `render_image_prepared`:5678 | `bc8e9ae825d5b919622ed1c45693a127ad183135090abfd314878e37cd5fdb81` | +846 / -303 |
| `src/include/libplacebo/renderer.h` | `pl_renderer_preflight_image`:1005 | `51f9b8acb3af2d3e40ebd2d94768d6caef96e94fc8413fe27bd1cade88492065` | +61 / -18 |
| `src/tests/renderer_prepare.c` | `test_replacement_lifecycle`:688; `test_empty_render`:825; `test_frame_access`:919; `test_temporal_mix`:1115 | `5ae15bb8647be71e5a1eccf34ab27a5ea0e89d3ea6e7a56d49220d7239cfc63e` | +1214 / -2 |
| `src/tests/lut_prepare.c` | `test_lut`:73 | `efa7b572bf9be6df327d834f9ab9dcee65d83f99d8f23d4ad2445e72e148b9e9` | +52 / -2 |

The shared library was rebuilt after the final renderer tests. The configured
application suite then passed **114/114** against it
(`/tmp/app-async-ordered-full.log`). This is integration-suite evidence, not a
live capture/device-transition run. API/SONAME remains 374.

## Retained implementation fix; removed coverage additions

`render_temporal_mix_inner` now checks source compatibility before assigning a
free prepared slot when a frame window advances. Existing signature cache hits
remain reserved first. This implementation fix is retained.

At the user's request, the subsequent mixed-layout, enhancement-acquisition,
HDR/LUT matrix, pending-transition and temporal-grain test additions were removed.
The renderer fixture is restored to its state before those coverage rounds.
Their previous logs are historical evidence, not maintained acceptance cases.
Further work targets differences from ordinary libplacebo rendering, using the
existing libplacebo tests rather than adding bespoke coverage matrices.

## Native frame-mix fallback parity (2026-10-06)

`describe_temporal_mix` now resolves the reference frame and target geometry
through the existing pass setup before describing caches. As in ordinary
`pl_render_image_mix`, an empty destination or missing intermediate framebuffer
format selects the nearest-image path. That path remains asynchronously prepared.
`preparation_traverse` permits the native no-FBO fallback when the shared builders
succeed; other graph errors remain failures.

`pl_renderer_describe_image_mix` also sends an empty input window through the
normal AUTO policy handling, so an unavailable async backend can select the
ordinary renderer instead of returning unsupported prematurely.

No new tests were added. The existing 21 library tests passed
(`/tmp/lib-mix-parity-final.log`) and the shared library rebuilt
(`/tmp/lib-mix-parity-build.log`). API/SONAME remains 374.

Delayed peak detection and ICC use the same implementations as ordinary rendering;
missing combination coverage is not an unimplemented feature. Capture/display
hardware verification is separate from renderer feature parity. Strict async
still requires a backend with preparation support and hooks with explicit
preparation callbacks; AUTO retains the ordinary rendering fallback.

## Sanitizer and Vulkan validation fixes (2026-10-06)

Existing library suites exposed and drove fixes for three defects:

- `pass_uninit` now clears the dispatch callback borrowing the pass stack frame.
  ASan caught a stack-use-after-return when a later empty-frame clear invoked
  the previous pass's callback (`/tmp/lib-asan-tests.log`).
- Vulkan memory slabs now own their diagnostic allocation tag, replacing it
  only when the string changes. ASan caught a heap-use-after-free during final
  allocator statistics after the triggering object was destroyed
  (`/tmp/lib-asan-renderer-fixed.log`).
- Debug-utils commands are resolved through the instance, so they are not used
  when `VK_EXT_debug_utils` was not enabled. Forced validation previously reported
  `UNASSIGNED-GeneralParameterError-ExtensionNotEnabled` for debug labels
  (`/tmp/lib-validation-before-fixes-detail.log`).

These are implementation fixes; no new test cases were added. The separate
`build-asan` configuration enables `b_sanitize=address,undefined`, leak detection,
and halt-on-error. It uses the same shaderc/LittleCMS dependency paths as
`build-full`. The runtime selects `/etc/vulkan/icd.d/nvidia_icd.json` and disables
Mesa's implicit device selector to isolate the intended hardware driver.

Standalone `vulkaninfo --summary` with libasan preloaded reproduces the DBus leak
without loading libplacebo (`/tmp/vulkaninfo-nvidia-asan.log`). The final hardware
sanitizer run therefore uses one external suppression, `leak:libdbus-1.so.3`, in
`/tmp/libplacebo-external-lsan.supp`. Leak detection remains enabled; there is no
allocator-wide or libplacebo suppression. An unsuppressed run must not be called
leak-clean. This is library validation, not an ASan-instrumented application run.

Vulkan validation is forced with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`,
`VK_LOADER_DEBUG=layer`, and
`VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`.
The final log confirms layer insertion and reports zero validation errors,
zero VUID messages and zero synchronization hazards across 21 passing tests
(`/tmp/lib-validation-final.log`, `/tmp/lib-validation-final-detail.log`).

Final ASan/UBSan result: **21/21 passed**, no memory-access or undefined-behavior
reports, with the external DBus suppression described above
(`/tmp/lib-asan-final.log`, `/tmp/lib-asan-final-detail.log`).

The application integration run exposed an existing reset-test ordering
assumption: admission may win before reset, or cancellation may resolve afterward.
The existing assertion now checks exactly-once completion for either valid order;
no production queue behavior or new test cases were added. It passed 100 repeats
(`/tmp/app-queue-validation.log`).

Changed implementation locations:
- `src/renderer.c:4011` — `static void pass_uninit`.
- `src/vulkan/context.c:418` — `static const struct vk_fun vk_dev_funs[]`.
- `src/vulkan/malloc.c:62` — `struct vk_slab {`.
- `src/vulkan/malloc.c:379` — `static struct vk_slab *slab_alloc`.
- `src/vulkan/malloc.c:1022` — `bool vk_malloc_slice`.

Final application integration result: **114/114 passed** against the rebuilt
normal library (`/tmp/app-validation-final.log`).
