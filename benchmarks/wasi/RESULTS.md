# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (micro-jit) |    wasm3 |    Winch | Cranelift | SF/wasm3 | SF/Winch |  SF/CL |
|-----------------------|----------------------:|---------:|---------:|----------:|---------:|---------:|-------:|
| CoreMark (score)      |            **14,854** |    4,224 |    8,660 |    14,669 |    3.52x |   171.5% | 101.3% |
| SHA-256 (MB/s)        |            **115.21** |    28.51 |   112.61 |    249.26 |    4.04x |   102.3% |  46.2% |
| bzip2 (MB/s)          |              **9.35** |     3.13 |     8.46 |     19.41 |    2.99x |   110.5% |  48.2% |
| LZ4 compress (MB/s)   |            **447.30** |   193.93 |   445.66 |    736.45 |    2.31x |   100.4% |  60.7% |
| LZ4 decompress (MB/s) |          **1,068.77** |   379.04 | 1,074.52 |  3,455.15 |    2.82x |    99.5% |  30.9% |
| mandelbrot (ms)       |            **2,851**  |    4,234 |    2,166 |       855 |    1.49x |    76.0% |  30.0% |
| c-ray (ms)            |            **1,966**  |    5,163 |    1,543 |       412 |    2.63x |    78.4% |  20.9% |
| STREAM Copy (MB/s)    |           **10,297**  |    3,165 |   14,536 |    44,124 |    3.25x |    70.8% |  23.3% |
| STREAM Scale (MB/s)   |           **15,546**  |    4,122 |   26,958 |    49,692 |    3.77x |    57.7% |  31.3% |
| STREAM Add (MB/s)     |           **17,830**  |    4,403 |   26,236 |    48,398 |    4.05x |    67.9% |  36.8% |
| STREAM Triad (MB/s)   |           **18,613**  |    4,215 |   28,149 |    47,864 |    4.42x |    66.1% |  38.9% |
| lua/fib (s)           |              **2.75** |    10.54 |     6.63 |      4.60 |    3.83x |   241.1% | 167.3% |
| lua/sunfish (score)   |            **2,262**  |      772 |    1,657 |     2,896 |    2.93x |   136.5% |  78.1% |
| lua/json_bench (score)|            **6,329**  |    1,600 |    5,506 |     9,616 |    3.96x |   114.9% |  65.8% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- wasm3: `build-release/wasm3` 79d412ea5fcf92f0efe658d52827a0e0a96ff442
- Winch: wasmtime (`-C compiler=winch`, single-pass baseline JIT)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- SF beats wasm3 on all 14 benchmarks (1.5–4.4x)
- SF beats Winch on CoreMark (172%), SHA-256 (102%), bzip2 (111%), LZ4 compress (100%), Lua fib (241%), Lua sunfish (137%), Lua json (115%)
- SF matches/beats Cranelift on CoreMark (101.3%) and Lua fib (167.3%)
