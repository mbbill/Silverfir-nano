# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **37,032** |    14,669 |   37,869 | 252.4% |  97.8% |
| SHA-256 (MB/s)        |      **271.04** |    249.26 |   201.20 | 108.7% | 134.7% |
| bzip2 (MB/s)          |       **20.23** |     19.41 |    19.88 | 104.2% | 101.8% |
| LZ4 compress (MB/s)   |      **768.21** |    736.45 |   704.01 | 104.3% | 109.1% |
| LZ4 decompress (MB/s) |    **3,214.90** |  3,455.15 | 2,908.07 |  93.0% | 110.6% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.507** |     12.18 |     3.14 | 485.8% | 125.2% |
| lua/sunfish (score)   |      **10,924** |     2,896 |   11,101 | 377.2% |  98.4% |
| lua/json_bench (score)|      **27,884** |     9,616 |   30,179 | 290.0% |  92.4% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **854** |       855 |    2,035 | 100.2% | 238.4% |
| c-ray (ms)            |       **2,140** |     2,055 |    1,947 |  96.0% |  91.0% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,124** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,629** |    49,692 |   18,332 |  99.9% | 270.7% |
| STREAM Add (MB/s)     |      **64,256** |    48,398 |   29,989 | 132.8% | 214.3% |
| STREAM Triad (MB/s)   |      **48,387** |    47,864 |   30,869 | 101.1% | 156.7% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 2, tied 2
- SF wins: Lua fib (486%), Lua sunfish (377%), Lua json (290%), CoreMark (252%), STREAM Add (133%), SHA-256 (109%), LZ4 compress (104%), bzip2 (104%), STREAM Triad (101%), mandelbrot (100.2%)
- Ties: STREAM Copy (100.0%), STREAM Scale (99.9%)
- Closest losses: c-ray (96%), LZ4 decompress (93%)

**SF vs V8** (TurboFan JIT): SF wins 10, V8 wins 4
- SF wins: STREAM Scale (271%), mandelbrot (238%), STREAM Add (214%), STREAM Triad (157%), SHA-256 (134%), Lua fib (125%), STREAM Copy (111%), LZ4 decompress (111%), LZ4 compress (109%), bzip2 (102%)
- Closest losses: Lua sunfish (98%), CoreMark (98%), Lua json (92%), c-ray (91%)

**Overall best** (absolute winner per benchmark): SF wins 8, V8 wins 4, CL wins 2
- SF wins: SHA-256, LZ4 compress, bzip2, Lua fib, mandelbrot, STREAM Copy, STREAM Add, STREAM Triad
- V8 wins: CoreMark, Lua sunfish, Lua json, c-ray
- CL wins: LZ4 decompress, STREAM Scale

## Notes

- Silverfir: `sf-nano-cli` (release build, jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- Internal tracking dashboard: https://mbbill.github.io/Silverfir-nano/dev/bench/
