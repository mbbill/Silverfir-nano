# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **32,790** |    14,669 |   37,869 | 223.5% |  86.6% |
| SHA-256 (MB/s)        |      **268.39** |    249.26 |   201.20 | 107.7% | 133.4% |
| bzip2 (MB/s)          |       **16.26** |     19.41 |    19.88 |  83.8% |  81.8% |
| LZ4 compress (MB/s)   |      **769.02** |    736.45 |   704.01 | 104.4% | 109.2% |
| LZ4 decompress (MB/s) |    **3,129.87** |  3,455.15 | 2,908.07 |  90.6% | 107.6% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.522** |     12.18 |     3.14 | 483.0% | 124.5% |
| lua/sunfish (score)   |       **5,434** |     2,896 |   11,101 | 187.6% |  48.9% |
| lua/json_bench (score)|      **15,923** |     9,616 |   30,179 | 165.6% |  52.7% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **827** |       855 |    2,035 | 103.4% | 246.1% |
| c-ray (ms)            |       **3,655** |     2,055 |    1,947 |  56.2% |  53.3% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,139** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,659** |    49,692 |   18,332 | 100.0% | 270.9% |
| STREAM Add (MB/s)     |      **64,342** |    48,398 |   29,989 | 133.0% | 214.5% |
| STREAM Triad (MB/s)   |      **48,417** |    47,864 |   30,869 | 101.2% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 3, tied 1
- SF wins: CoreMark (224%), Lua fib (483%), Lua sunfish (188%), Lua json (166%), STREAM Add (133%), SHA-256 (108%), LZ4 compress (104%), mandelbrot (103%), STREAM Triad (101%), STREAM Copy (100.0%)
- Ties: STREAM Scale (100.0%)
- Closest losses: LZ4 decompress (91%), bzip2 (84%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (246%), STREAM Add (215%), STREAM Triad (157%), SHA-256 (133%), STREAM Copy (111%), Lua fib (125%), LZ4 decompress (108%), LZ4 compress (109%)
- Closest losses: CoreMark (87%), bzip2 (82%)

**Overall best** (absolute winner per benchmark): SF wins 7, V8 wins 5, CL wins 2
- SF wins: SHA-256, mandelbrot, LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
