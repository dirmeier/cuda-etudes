.PHONY: clean
.PHONY: run
.PHONY: lint
.PHONY: format
.PHONY: memcheck

all: build

build:
	meson setup build . && ninja -C build

clean:
	rm -rf build

run: build
	./build/cuda-etudes

memcheck: build
	cuda-memcheck build/cuda-etudes

format:
	clang-format -i etudes/*.cu  etudes/*.h

lint:
	cd etudes && cpplint --filter=-legal/copyright,-readability/casting,-whitespace/braces,-whitespace/indent,-build/include_subdir,-whitespace/line_length \
	*.cu *.h
