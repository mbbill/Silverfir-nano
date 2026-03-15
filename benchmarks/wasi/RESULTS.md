# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Results

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **22,405** |    14,669 |   37,869 | 152.7% |  59.2% |
| SHA-256 (MB/s)        |      **205.06** |    249.26 |   201.20 |  82.3% | 101.9% |
| bzip2 (MB/s)          |       **12.56** |     19.41 |    19.88 |  64.7% |  63.2% |
| LZ4 compress (MB/s)   |      **563.02** |    736.45 |   704.01 |  76.4% |  80.0% |
| LZ4 decompress (MB/s) |    **1,976.52** |  3,455.15 | 2,908.07 |  57.2% |  68.0% |
| mandelbrot (ms)       |       **1,178** |       855 |    2,035 |  72.6% | 172.7% |
| c-ray (ms)            |       **5,615** |     2,055 |    1,947 |  36.6% |  34.7% |
| STREAM Copy (MB/s)    |      **20,484** |    44,124 |   39,714 |  46.4% |  51.6% |
| STREAM Scale (MB/s)   |      **26,914** |    49,692 |   18,332 |  54.2% | 146.8% |
| STREAM Add (MB/s)     |      **31,275** |    48,398 |   29,989 |  64.6% | 104.3% |
| STREAM Triad (MB/s)   |      **28,504** |    47,864 |   30,869 |  59.5% |  92.3% |
| lua/fib38 (s)         |       **3.472** |     12.18 |     3.14 | 350.8% |  90.4% |
| lua/sunfish (score)   |       **3,836** |     2,896 |   11,101 | 132.5% |  34.6% |
| lua/json_bench (score)|      **11,895** |     9,616 |   30,179 | 123.7% |  39.4% |

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
- SF beats Cranelift on CoreMark (153%), Lua fib (351%), Lua sunfish (133%), Lua json (124%)
- SF beats V8 on mandelbrot (173%), STREAM Scale (147%), STREAM Add (104%), SHA-256 (102%)
- SF reaches 90% of V8 on Lua fib, 92% on STREAM Triad
