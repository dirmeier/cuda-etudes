header_files = [
  'add.h',
  'matmul.h',
  'gauss_jordan.h',
  'rmh.h'
]

lib_sources = [
  'add.cu',
  'matmul.cu',
  'gauss_jordan.cu',
  'rmh.cu'
]

install_headers(files(header_files))

lib = library(
  'etudes',
  lib_sources,
  include_directories: inc,
  gnu_symbol_visibility: 'hidden',
  cpp_args: cuda_etudes_warning_flags
)
