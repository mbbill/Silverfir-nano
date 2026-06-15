commit: ecf26a68

On Apple M4 the native backend posts CoreMark 16,053 (3.59x wasm3, 1.85x
Winch, ~1.03x Cranelift), LZ4 compress 563 MB/s, lua/sunfish 2,941,
lua/json_bench 8,942. Across the suite it runs roughly 2.3x-5.5x wasm3 and is
competitive with or ahead of Winch, trailing Cranelift on float-heavy kernels
(mandelbrot/c-ray) and STREAM bandwidth.
