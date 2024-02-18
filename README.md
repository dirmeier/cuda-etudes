# CUDA études

[![Project Status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)

:notes: A collection of CUDA recipes

## About

This repository implements some recipes in to learn CUDA.

## Dependencies

- `meson` and `ninja`
- CUDA >=10 + `cuBLAS`/`cuRAND`
- `gtest`

On Arch, you can install using `pacman`

```bash
pacman -S meson ninja
pacman -S cuda cuda-tools
pacman -S gtest
```

## Usage

Clone/download the project and run:

```bash
make run
```

This builds an executable and runs all CUDA examples.

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
