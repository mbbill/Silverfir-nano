# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark | Silverfir (micro-jit) | wasm3 | Cranelift | SF/wasm3 | SF/CL |
|-----------|----------------------:|------:|----------:|---------:|------:|
| CoreMark (score) | **14,619** | 4,224 | 14,669 | 3.46x | 99.7% |
| SHA-256 (MB/s) | **93.33** | 28.51 | 249.26 | 3.27x | 37.4% |
| bzip2 (MB/s) | **7.40** | 3.13 | 19.41 | 2.36x | 38.1% |
| LZ4 compress (MB/s) | **414.96** | 193.93 | 736.45 | 2.14x | 56.3% |
| LZ4 decompress (MB/s) | **816.13** | 379.04 | 3,455.15 | 2.15x | 23.6% |
| mandelbrot (ms) | 4,480 | 4,234 | 855 | 0.95x | 19.1% |
| c-ray (ms) | **2,922** | 5,163 | 412 | 1.77x | 14.1% |
| STREAM Copy (MB/s) | **8,841** | 3,165 | 44,124 | 2.79x | 20.0% |
| STREAM Scale (MB/s) | **12,850** | 4,122 | 49,692 | 3.12x | 25.9% |
| STREAM Add (MB/s) | **15,652** | 4,403 | 48,398 | 3.56x | 32.3% |
| STREAM Triad (MB/s) | **13,565** | 4,215 | 47,864 | 3.22x | 28.3% |
| lua/fib (s) | **3.20** | 10.54 | 4.60 | 3.29x | 143.7% |
| lua/sunfish (score) | **1,904** | 772 | 2,896 | 2.47x | 65.7% |
| lua/json_bench (score) | **5,342** | 1,600 | 9,616 | 3.34x | 55.6% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- wasm3: `build-release/wasm3` 79d412ea5fcf92f0efe658d52827a0e0a96ff442
- Cranelift: wasmtime (`-C compiler=cranelift`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- SF beats wasm3 on all benchmarks except mandelbrot (0.95x, essentially tied)
- SF matches Cranelift on CoreMark (99.7%) and beats it on Lua fib (143.7%)
