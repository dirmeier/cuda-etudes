
.PHONY: run

all:
	meson build . && ninja -C build

clean:
	rm -rf build

run: all
	./build/cuda-etudes
