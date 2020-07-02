
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
