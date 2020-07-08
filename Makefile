
.PHONY: clean
.PHONY: all
.PHONY: test

all: build

build:
	meson build . && ninja -C build

clean:
	rm -rf build

test: build
	./build/cuda-etudes

memcheck: build
	cuda-memcheck build/cuda-etudes

format:
	clang-format -i src/*.cu
	clang-format -i include/*.h include/etudes/*.h