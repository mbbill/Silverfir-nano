# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark | Silverfir (micro-jit) | wasm3 | Cranelift | SF/wasm3 | SF/CL |
|-----------|----------------------:|------:|----------:|---------:|------:|
| CoreMark (score) | **14,692** | 4,224 | 14,669 | 3.48x | 100.2% |
| SHA-256 (MB/s) | **117.89** | 28.51 | 249.26 | 4.13x | 47.3% |
| bzip2 (MB/s) | **8.86** | 3.13 | 19.41 | 2.83x | 45.6% |
| LZ4 compress (MB/s) | **399.36** | 193.93 | 736.45 | 2.06x | 54.2% |
| LZ4 decompress (MB/s) | **1,099.45** | 379.04 | 3,455.15 | 2.90x | 31.8% |
| mandelbrot (ms) | **2,849** | 4,234 | 855 | 1.49x | 30.0% |
| c-ray (ms) | **2,300** | 5,163 | 412 | 2.24x | 17.9% |
| STREAM Copy (MB/s) | **10,581** | 3,165 | 44,124 | 3.34x | 24.0% |
| STREAM Scale (MB/s) | **14,243** | 4,122 | 49,692 | 3.46x | 28.7% |
| STREAM Add (MB/s) | **19,706** | 4,403 | 48,398 | 4.48x | 40.7% |
| STREAM Triad (MB/s) | **18,696** | 4,215 | 47,864 | 4.43x | 39.1% |
| lua/fib (s) | **2.98** | 10.54 | 4.60 | 3.54x | 154.4% |
| lua/sunfish (score) | **1,914** | 772 | 2,896 | 2.48x | 66.1% |
| lua/json_bench (score) | **5,474** | 1,600 | 9,616 | 3.42x | 56.9% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- wasm3: `build-release/wasm3` 79d412ea5fcf92f0efe658d52827a0e0a96ff442
- Cranelift: wasmtime (`-C compiler=cranelift`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- SF beats wasm3 on all 14 benchmarks (2.1–4.5x)
- SF matches Cranelift on CoreMark (100.2%) and beats it on Lua fib (154.4%)
