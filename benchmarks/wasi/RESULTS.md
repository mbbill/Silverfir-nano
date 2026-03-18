# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **32,010** |    14,669 |   37,869 | 218.1% |  84.5% |
| SHA-256 (MB/s)        |      **238.12** |    249.26 |   201.20 |  95.5% | 118.3% |
| bzip2 (MB/s)          |       **15.96** |     19.41 |    19.88 |  82.2% |  80.3% |
| LZ4 compress (MB/s)   |      **762.89** |    736.45 |   704.01 | 103.6% | 108.4% |
| LZ4 decompress (MB/s) |    **3,214.31** |  3,455.15 | 2,908.07 |  93.0% | 110.5% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.351** |     12.18 |     3.14 | 517.9% | 133.6% |
| lua/sunfish (score)   |       **5,373** |     2,896 |   11,101 | 185.5% |  48.4% |
| lua/json_bench (score)|      **15,794** |     9,616 |   30,179 | 164.2% |  52.3% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **834** |       855 |    2,035 | 102.5% | 244.0% |
| c-ray (ms)            |       **3,688** |     2,055 |    1,947 |  55.7% |  52.8% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **51,314** |    44,124 |   39,714 | 116.3% | 129.2% |
| STREAM Scale (MB/s)   |      **49,688** |    49,692 |   18,332 | 100.0% | 271.0% |
| STREAM Add (MB/s)     |      **64,412** |    48,398 |   29,989 | 133.1% | 214.8% |
| STREAM Triad (MB/s)   |      **48,426** |    47,864 |   30,869 | 101.2% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 9, CL wins 4, tied 1
- SF wins: CoreMark (218%), Lua fib (518%), Lua sunfish (186%), Lua json (164%), STREAM Add (133%), STREAM Copy (116%), LZ4 compress (104%), mandelbrot (103%), STREAM Triad (101%)
- Ties: STREAM Scale (100.0%)
- Closest losses: SHA-256 (96%), LZ4 decompress (93%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (244%), STREAM Add (215%), STREAM Triad (157%), Lua fib (134%), STREAM Copy (129%), SHA-256 (118%), LZ4 decompress (111%), LZ4 compress (108%)
- Closest losses: CoreMark (85%), bzip2 (80%)

**Overall best** (absolute winner per benchmark): SF wins 6, V8 wins 5, CL wins 3
- SF wins: mandelbrot, LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
