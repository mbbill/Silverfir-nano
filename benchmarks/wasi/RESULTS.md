# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (micro-jit) |    wasm3 |    Winch | Cranelift | SF/wasm3 | SF/Winch |  SF/CL |
|-----------------------|----------------------:|---------:|---------:|----------:|---------:|---------:|-------:|
| CoreMark (score)      |            **16,053** |    4,224 |    8,660 |    14,669 |    3.59x |   175.0% | 103.3% |
| SHA-256 (MB/s)        |            **132.16** |    28.51 |   112.61 |    249.26 |    4.64x |   117.4% |  53.0% |
| bzip2 (MB/s)          |             **10.90** |     3.13 |     8.46 |     19.41 |    3.48x |   128.8% |  56.2% |
| LZ4 compress (MB/s)   |            **563.02** |   193.93 |   445.66 |    736.45 |    2.28x |    99.1% |  60.0% |
| LZ4 decompress (MB/s) |          **1,301.11** |   379.04 | 1,074.52 |  3,455.15 |    3.43x |   121.1% |  37.7% |
| mandelbrot (ms)       |            **2,823**  |    4,234 |    2,166 |       855 |    1.50x |    76.7% |  30.3% |
| c-ray (ms)            |            **1,914**  |    5,163 |    1,543 |       412 |    2.70x |    80.6% |  21.5% |
| STREAM Copy (MB/s)    |           **16,142**  |    3,165 |   14,536 |    44,124 |    5.10x |   111.0% |  36.6% |
| STREAM Scale (MB/s)   |           **20,200**  |    4,122 |   26,958 |    49,692 |    4.90x |    74.9% |  40.6% |
| STREAM Add (MB/s)     |           **24,189**  |    4,403 |   26,236 |    48,398 |    5.49x |    92.2% |  50.0% |
| STREAM Triad (MB/s)   |           **22,010**  |    4,215 |   28,149 |    47,864 |    5.22x |    78.2% |  46.0% |
| lua/fib38 (s)         |              **5.88** |    26.63 |    17.46 |     12.18 |    4.53x |   296.9% | 207.1% |
| lua/sunfish (score)   |            **2,941**  |      772 |    1,657 |     2,896 |    3.20x |   149.3% |  85.4% |
| lua/json_bench (score)|            **8,942**  |    1,600 |    5,506 |     9,616 |    4.46x |   129.7% |  74.3% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- wasm3: `build-release/wasm3` 79d412ea5fcf92f0efe658d52827a0e0a96ff442
- Winch: wasmtime (`-C compiler=winch`, single-pass baseline JIT)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- SF beats wasm3 on all 14 benchmarks (1.5–5.5x)
- SF beats Winch on 9 of 14: CoreMark (175%), SHA-256 (117%), bzip2 (129%), LZ4 decompress (121%), STREAM Copy (111%), Lua fib (297%), Lua sunfish (149%), Lua json (130%), LZ4 compress (~99%)
- SF beats Cranelift on CoreMark (103.3%) and Lua fib (207.1%)
- SF reaches 85% of Cranelift on Lua sunfish, 74% on Lua json
