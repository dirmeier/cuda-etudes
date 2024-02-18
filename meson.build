project(
  'cuda-etudes',
  'cpp',
  'cuda',
  version : '0.0.1',
  default_options : [
    'buildtype=debug',
    'warning_level=2',
    'cpp_std=c++23',
    'b_ndebug=if-release'],
  license : 'BSD-3',
  meson_version : '>= 0.52.0')

cpp = meson.get_compiler('cpp')

cuda_etudes_warning_flags = []
foreach cflag: [
  '-Wduplicated-cond',
  '-Wduplicated-branches',
  '-Wlogical-op',
  '-Wrestrict',
  '-Wnull-dereference',
  '-Wuseless-cast',
  '-Wdouble-promotion',
  '-Wshadow',
  '-Wformat=1']
    if cpp.has_argument(cflag)
      cuda_etudes_warning_flags += cflag
    endif
endforeach

cuda_dep = dependency('cuda', modules: ['cublas', 'curand'])

inc = include_directories('etudes')

subdir('etudes')

aarch_cpp_sources = files([
  'main.cpp',
  'etudes/rmh.cu'
])

executable(
  'cuda-etudes',
  'main.cpp',
   include_directories: inc,
   dependencies: [cuda_dep],
   cpp_args: cuda_etudes_warning_flags,
   link_with: lib
)
