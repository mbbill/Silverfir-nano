# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **33,770** |    14,669 |   37,869 | 230.2% |  89.2% |
| SHA-256 (MB/s)        |      **269.10** |    249.26 |   201.20 | 108.0% | 133.7% |
| bzip2 (MB/s)          |       **16.84** |     19.41 |    19.88 |  86.8% |  84.7% |
| LZ4 compress (MB/s)   |      **783.14** |    736.45 |   704.01 | 106.3% | 111.2% |
| LZ4 decompress (MB/s) |    **3,136.42** |  3,455.15 | 2,908.07 |  90.8% | 107.9% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.543** |     12.18 |     3.14 | 478.9% | 123.5% |
| lua/sunfish (score)   |       **5,372** |     2,896 |   11,101 | 185.5% |  48.4% |
| lua/json_bench (score)|      **15,249** |     9,616 |   30,179 | 158.6% |  50.5% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **808** |       855 |    2,035 | 105.8% | 251.9% |
| c-ray (ms)            |       **3,629** |     2,055 |    1,947 |  56.6% |  53.6% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,113** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,644** |    49,692 |   18,332 | 100.0% | 270.8% |
| STREAM Add (MB/s)     |      **64,223** |    48,398 |   29,989 | 132.7% | 214.2% |
| STREAM Triad (MB/s)   |      **48,408** |    47,864 |   30,869 | 101.1% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 3, tied 1
- SF wins: CoreMark (230%), Lua fib (479%), Lua sunfish (186%), Lua json (159%), STREAM Add (133%), SHA-256 (108%), LZ4 compress (106%), mandelbrot (106%), STREAM Triad (101%), STREAM Copy (100.0%)
- Ties: STREAM Scale (100.0%)
- Closest losses: LZ4 decompress (91%), bzip2 (87%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (252%), STREAM Add (214%), STREAM Triad (157%), SHA-256 (134%), Lua fib (124%), STREAM Copy (111%), LZ4 compress (111%), LZ4 decompress (108%)
- Closest losses: CoreMark (89%), bzip2 (85%)

**Overall best** (absolute winner per benchmark): SF wins 7, V8 wins 5, CL wins 2
- SF wins: SHA-256, mandelbrot, LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
