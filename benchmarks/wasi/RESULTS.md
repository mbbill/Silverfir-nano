# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (micro-jit) |    wasm3 |    Winch | Cranelift | SF/wasm3 | SF/Winch |  SF/CL |
|-----------------------|----------------------:|---------:|---------:|----------:|---------:|---------:|-------:|
| CoreMark (score)      |            **14,898** |    4,224 |    8,660 |    14,669 |    3.53x |   172.0% | 101.6% |
| SHA-256 (MB/s)        |            **126.80** |    28.51 |   112.61 |    249.26 |    4.45x |   112.6% |  50.9% |
| bzip2 (MB/s)          |             **10.51** |     3.13 |     8.46 |     19.41 |    3.36x |   124.2% |  54.1% |
| LZ4 compress (MB/s)   |            **419.71** |   193.93 |   445.66 |    736.45 |    2.16x |    94.2% |  57.0% |
| LZ4 decompress (MB/s) |          **1,293.81** |   379.04 | 1,074.52 |  3,455.15 |    3.41x |   120.4% |  37.4% |
| mandelbrot (ms)       |            **2,853**  |    4,234 |    2,166 |       855 |    1.48x |    75.9% |  30.0% |
| c-ray (ms)            |            **1,866**  |    5,163 |    1,543 |       412 |    2.77x |    82.6% |  22.1% |
| STREAM Copy (MB/s)    |           **15,575**  |    3,165 |   14,536 |    44,124 |    4.92x |   107.1% |  35.3% |
| STREAM Scale (MB/s)   |           **18,450**  |    4,122 |   26,958 |    49,692 |    4.48x |    68.4% |  37.1% |
| STREAM Add (MB/s)     |           **22,537**  |    4,403 |   26,236 |    48,398 |    5.12x |    85.9% |  46.6% |
| STREAM Triad (MB/s)   |           **20,617**  |    4,215 |   28,149 |    47,864 |    4.89x |    73.2% |  43.1% |
| lua/fib38 (s)         |              **7.47** |    26.63 |    17.46 |     12.18 |    3.57x |   233.7% | 163.1% |
| lua/sunfish (score)   |            **2,224**  |      772 |    1,657 |     2,896 |    2.88x |   134.2% |  76.8% |
| lua/json_bench (score)|            **6,252**  |    1,600 |    5,506 |     9,616 |    3.91x |   113.5% |  65.0% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- wasm3: `build-release/wasm3` 79d412ea5fcf92f0efe658d52827a0e0a96ff442
- Winch: wasmtime (`-C compiler=winch`, single-pass baseline JIT)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- SF beats wasm3 on all 14 benchmarks (1.5–5.1x)
- SF beats Winch on CoreMark (172%), SHA-256 (113%), bzip2 (124%), LZ4 decompress (120%), STREAM Copy (107%), Lua fib (234%), Lua sunfish (134%), Lua json (114%)
- SF matches/beats Cranelift on CoreMark (101.6%) and Lua fib (163.1%)
