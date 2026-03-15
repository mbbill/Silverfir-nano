# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (JIT) | Cranelift |         V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|-----------:|-------:|-------:|
| CoreMark (score)      |      **20,768** |    14,669 |     37,869 | 141.6% |  54.8% |
| SHA-256 (MB/s)        |      **170.82** |    249.26 |     201.20 |  68.5% |  84.9% |
| bzip2 (MB/s)          |       **11.40** |     19.41 |      19.88 |  58.7% |  57.3% |
| LZ4 compress (MB/s)   |      **563.02** |    736.45 |     704.01 |  76.4% |  80.0% |
| LZ4 decompress (MB/s) |    **1,609.56** |  3,455.15 |   2,908.07 |  46.6% |  55.3% |
| mandelbrot (ms)       |       **1,706** |       855 |      2,035 |  50.1% | 119.3% |
| c-ray (ms)            |       **8,350** |     2,055 |      1,947 |  24.6% |  23.3% |
| STREAM Copy (MB/s)    |      **17,168** |    44,124 |     39,714 |  38.9% |  43.2% |
| STREAM Scale (MB/s)   |      **26,878** |    49,692 |     18,332 |  54.1% | 146.6% |
| STREAM Add (MB/s)     |      **31,242** |    48,398 |     29,989 |  64.6% | 104.2% |
| STREAM Triad (MB/s)   |      **26,184** |    47,864 |     30,869 |  54.7% |  84.8% |
| lua/fib38 (s)         |       **3.877** |     12.18 |       3.14 | 314.2% |  81.0% |
| lua/sunfish (score)   |       **3,422** |     2,896 |     11,101 | 118.2% |  30.8% |
| lua/json_bench (score)|      **10,828** |     9,616 |     30,179 | 112.6% |  35.9% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
- SF beats Cranelift on CoreMark (138%), Lua fib (314%), Lua sunfish (118%), Lua json (113%)
- SF beats V8 on STREAM Scale (136%)
- SF reaches 80–85% of V8 on SHA-256, LZ4 compress, Lua fib, STREAM Triad
