project(
  'cuda-etudes',
  'cpp',
  version : '0.0.1',
  default_options : [
    'buildtype=debug',
    'warning_level=3',
    'cpp_std=c++14',
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

inc = include_directories('include')
subdir('include')
subdir('src')

executable(
  'cuda-etudes',
  'main.cpp',
   include_directories : inc,
   link_with : lib)
