# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (micro-jit) |    wasm3 |    Winch | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------------:|---------:|---------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |            **20,269** |    4,224 |    8,660 |    14,669 |   38,996 | 138.2% |  52.0% |
| SHA-256 (MB/s)        |            **170.82** |    28.51 |   112.61 |    249.26 |   205.02 |  68.5% |  83.3% |
| bzip2 (MB/s)          |             **11.11** |     3.13 |     8.46 |     19.41 |    19.78 |  57.2% |  56.2% |
| LZ4 compress (MB/s)   |            **563.02** |   193.93 |   445.66 |    736.45 |   684.00 |  76.4% |  82.3% |
| LZ4 decompress (MB/s) |          **1,609.56** |   379.04 | 1,074.52 |  3,455.15 | 2,889.57 |  46.6% |  55.7% |
| mandelbrot (ms)       |             **2,823** |    4,234 |    2,166 |       855 |    2,073 |  30.3% |  73.4% |
| c-ray (ms)            |             **8,685** |   25,058 |    7,747 |     2,055 |    1,999 |  23.7% |  23.0% |
| STREAM Copy (MB/s)    |            **17,168** |    3,165 |   14,536 |    44,124 |   42,105 |  38.9% |  40.8% |
| STREAM Scale (MB/s)   |            **24,867** |    4,122 |   26,958 |    49,692 |   19,048 |  50.0% | 130.5% |
| STREAM Add (MB/s)     |            **28,514** |    4,403 |   26,236 |    48,398 |   30,000 |  58.9% |  95.0% |
| STREAM Triad (MB/s)   |            **24,855** |    4,215 |   28,149 |    47,864 |   29,630 |  51.9% |  83.9% |
| lua/fib38 (s)         |             **3.877** |    26.63 |    17.46 |     12.18 |     3.20 | 314.2% |  82.5% |
| lua/sunfish (score)   |             **3,422** |      772 |    1,657 |     2,896 |    9,795 | 118.2% |  34.9% |
| lua/json_bench (score)|            **10,828** |    1,600 |    5,506 |     9,616 |   24,536 | 112.6% |  44.1% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- wasm3: `build-release/wasm3` 79d412ea5fcf92f0efe658d52827a0e0a96ff442
- Winch: wasmtime (`-C compiler=winch`, single-pass baseline JIT)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Chrome browser via `bench.html` (V8 TurboFan optimizing JIT)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
- SF beats Cranelift on CoreMark (138%), Lua fib (314%), Lua sunfish (118%), Lua json (113%)
- SF beats V8 on STREAM Scale (131%)
- SF reaches 82–83% of V8 on SHA-256, LZ4 compress, Lua fib, STREAM Triad
