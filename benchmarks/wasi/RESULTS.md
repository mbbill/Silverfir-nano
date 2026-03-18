# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **32,139** |    14,669 |   37,869 | 219.0% |  84.9% |
| SHA-256 (MB/s)        |      **259.42** |    249.26 |   201.20 | 104.1% | 128.9% |
| bzip2 (MB/s)          |       **16.51** |     19.41 |    19.88 |  85.1% |  83.0% |
| LZ4 compress (MB/s)   |      **762.89** |    736.45 |   704.01 | 103.6% | 108.4% |
| LZ4 decompress (MB/s) |    **3,214.31** |  3,455.15 | 2,908.07 |  93.0% | 110.5% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.351** |     12.18 |     3.14 | 517.9% | 133.6% |
| lua/sunfish (score)   |       **5,423** |     2,896 |   11,101 | 187.3% |  48.9% |
| lua/json_bench (score)|      **16,039** |     9,616 |   30,179 | 166.8% |  53.1% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **834** |       855 |    2,035 | 102.5% | 244.0% |
| c-ray (ms)            |       **3,657** |     2,055 |    1,947 |  56.2% |  53.2% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **51,314** |    44,124 |   39,714 | 116.3% | 129.2% |
| STREAM Scale (MB/s)   |      **49,688** |    49,692 |   18,332 | 100.0% | 271.0% |
| STREAM Add (MB/s)     |      **64,412** |    48,398 |   29,989 | 133.1% | 214.8% |
| STREAM Triad (MB/s)   |      **48,426** |    47,864 |   30,869 | 101.2% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 3, tied 1
- SF wins: CoreMark (219%), Lua fib (518%), Lua sunfish (187%), Lua json (167%), STREAM Add (133%), STREAM Copy (116%), SHA-256 (104%), LZ4 compress (104%), mandelbrot (103%), STREAM Triad (101%)
- Ties: STREAM Scale (100.0%)
- Closest losses: LZ4 decompress (93%), bzip2 (85%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (244%), STREAM Add (215%), STREAM Triad (157%), Lua fib (134%), STREAM Copy (129%), SHA-256 (129%), LZ4 decompress (111%), LZ4 compress (108%)
- Closest losses: CoreMark (85%), bzip2 (83%)

**Overall best** (absolute winner per benchmark): SF wins 7, V8 wins 5, CL wins 2
- SF wins: SHA-256, mandelbrot, LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
