# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (JIT) | Cranelift |         V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|-----------:|-------:|-------:|
| CoreMark (score)      |      **20,768** |    14,669 |     37,869 | 141.6% |  54.8% |
| SHA-256 (MB/s)        |      **170.82** |    249.26 |     201.20 |  68.5% |  84.9% |
| bzip2 (MB/s)          |       **11.40** |     19.41 |      19.88 |  58.7% |  57.3% |
| LZ4 compress (MB/s)   |      **563.02** |    736.45 |     704.01 |  76.4% |  80.0% |
| LZ4 decompress (MB/s) |    **1,759.18** |  3,455.15 |   2,908.07 |  50.9% |  60.5% |
| mandelbrot (ms)       |       **1,546** |       855 |      2,035 |  55.3% | 131.6% |
| c-ray (ms)            |       **6,073** |     2,055 |      1,947 |  33.8% |  32.1% |
| STREAM Copy (MB/s)    |      **17,168** |    44,124 |     39,714 |  38.9% |  43.2% |
| STREAM Scale (MB/s)   |      **26,914** |    49,692 |     18,332 |  54.2% | 146.8% |
| STREAM Add (MB/s)     |      **31,266** |    48,398 |     29,989 |  64.6% | 104.3% |
| STREAM Triad (MB/s)   |      **28,504** |    47,864 |     30,869 |  59.5% |  92.3% |
| lua/fib38 (s)         |       **3.849** |     12.18 |       3.14 | 316.4% |  81.6% |
| lua/sunfish (score)   |       **3,563** |     2,896 |     11,101 | 123.0% |  32.1% |
| lua/json_bench (score)|      **11,187** |     9,616 |     30,179 | 116.3% |  37.1% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
- SF beats Cranelift on CoreMark (142%), Lua fib (316%), Lua sunfish (123%), Lua json (116%)
- SF beats V8 on STREAM Scale (147%), mandelbrot (132%), STREAM Add (104%)
- SF reaches 80–85% of V8 on SHA-256, LZ4 compress, Lua fib, STREAM Triad
