lib_sources = [
  '01-add.cu',
  '02-matmul.cu',
  '03-gauss_jordan.cu'
]

lib = library(
  'etudes',
  lib_sources,
  include_directories: inc,
  gnu_symbol_visibility: 'hidden',
  cpp_args: cuda_etudes_warning_flags
)
