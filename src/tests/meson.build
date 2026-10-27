gpu_tests = [
  'vulkan.c', 'opengl_surfaceless.c', 'd3d11.c', 'dummy.c'
]

tests += ['pass_prepare.c', 'frame_queue_plan.c', 'frame_queue_eof.c']
if components.get('vulkan') and components.get('vk-proc-addr')
  tests += ['swapchain_cancel.c', 'pass_prepare_vulkan.c', 'dispatch_prepare.c', 'sampling_prepare.c', 'lut_prepare.c', 'renderer_prepare.c', 'color_reference.c']
endif

ts = []

foreach t : tests
  deps = [tdep_static]
  if t == 'opengl_surfaceless.c'
    deps += glad_dep
  endif
  if t == 'swapchain_cancel.c' or t == 'vulkan.c' or t == 'present_feedback.c' or t == 'pass_prepare_vulkan.c' or t == 'dispatch_prepare.c' or t == 'sampling_prepare.c' or t == 'lut_prepare.c' or t == 'renderer_prepare.c' or t == 'color_reference.c'
    deps += vulkan_headers
  endif
  sources = []
  if gpu_tests.contains(t)
    sources += 'gpu_tests.c'
  endif
  ts += { 'test': t,
          'sources': sources,
          'deps': deps }
endforeach

if components.get('vulkan') and components.get('vk-proc-addr')
  x11_cancel_dep = dependency('x11', required: false)
  if x11_cancel_dep.found()
    ts += { 'test': 'swapchain_cancel_native.c',
            'deps': [tdep_static, vulkan_headers, x11_cancel_dep] }
  endif
endif

dav1d = dependency('dav1d', required: false)
if dav1d.found()
  ts += { 'test': 'dav1d.c', 'deps': [dav1d, tdep_shared] }
endif

lavu = dependency('libavutil', version: '>=55.74.100', required: false)
lavc = dependency('libavcodec', required: false)
lavf = dependency('libavformat', required: false)
libav_found = lavu.found() and lavc.found() and lavf.found()
if libav_found
  ts += { 'test': 'libav.c', 'deps': [lavu, lavc, lavf, tdep_shared] }
endif

foreach t : ts
  e = executable('test.' + t['test'], [t['test']] + t.get('sources', []),
      objects: t.get('objects', []),
      c_args: [ '-Wno-unused-function' ],
      dependencies: t.get('deps', []),
      link_args: link_args,
      link_depends: link_depends,
  )

  test(t['test'], e, timeout: 120)
endforeach

subdir('include')

# Exercise the public discovery ABI across a real dynamic-library boundary.
mixer_module = shared_module('test_frame_mixer_module', 'frame_mixer_module.c',
    include_directories: [inc],
    c_args: ['-Wno-unused-parameter'],
    gnu_symbol_visibility: 'hidden',
)
mixer_loader = executable('test.frame_mixer_loader', 'frame_mixer_loader.c',
    dependencies: [tdep_shared, cc.find_library('dl', required: false)],
)
test('frame_mixer_loader', mixer_loader, args: [mixer_module.full_path()],
    depends: [mixer_module])

# Compile only this production unit with deterministic allocation faults. This
# target does not link the library, so substitutions cannot affect other code.
mixer_stress = executable('test.frame_mixer_stress',
    ['frame_mixer_stress.c', '../frame_mixer.c'],
    include_directories: [inc],
    c_args: ['-Dcalloc=mixer_test_calloc', '-Dfree=mixer_test_free',
             '-Wno-unused-parameter'],
)
test('frame_mixer_stress', mixer_stress)

if components.get('vulkan') and components.get('vk-proc-addr')
  blend_module = shared_module('test_frame_mixer_blend', 'frame_mixer_blend.c',
    dependencies: [tdep_shared, vulkan_headers],
    c_args: ['-Wno-unused-parameter'],
    gnu_symbol_visibility: 'hidden',
  )
  renderer_mixer = executable('test.renderer_mixer', 'renderer_mixer.c',
    dependencies: [tdep_shared, vulkan_headers, cc.find_library('dl', required: false)],
  )
  test('renderer_mixer', renderer_mixer, args: [blend_module.full_path()],
       depends: [blend_module], timeout: 120)
endif
