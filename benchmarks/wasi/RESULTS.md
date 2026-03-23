# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **34,118** |    14,669 |   37,869 | 232.6% |  90.1% |
| SHA-256 (MB/s)        |      **262.14** |    249.26 |   201.20 | 105.2% | 130.3% |
| bzip2 (MB/s)          |       **16.83** |     19.41 |    19.88 |  86.7% |  84.7% |
| LZ4 compress (MB/s)   |      **777.51** |    736.45 |   704.01 | 105.6% | 110.4% |
| LZ4 decompress (MB/s) |    **3,133.44** |  3,455.15 | 2,908.07 |  90.7% | 107.7% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.535** |     12.18 |     3.14 | 480.5% | 123.9% |
| lua/sunfish (score)   |       **5,542** |     2,896 |   11,101 | 191.4% |  49.9% |
| lua/json_bench (score)|      **15,646** |     9,616 |   30,179 | 162.7% |  51.8% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **811** |       855 |    2,035 | 105.4% | 250.9% |
| c-ray (ms)            |       **3,635** |     2,055 |    1,947 |  56.5% |  53.6% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,139** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,659** |    49,692 |   18,332 | 100.0% | 270.9% |
| STREAM Add (MB/s)     |      **64,342** |    48,398 |   29,989 | 132.9% | 214.5% |
| STREAM Triad (MB/s)   |      **48,408** |    47,864 |   30,869 | 101.1% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 3, tied 1
- SF wins: CoreMark (233%), Lua fib (481%), Lua sunfish (191%), Lua json (163%), STREAM Add (133%), SHA-256 (105%), LZ4 compress (106%), mandelbrot (105%), STREAM Triad (101%), STREAM Copy (100.0%)
- Ties: STREAM Scale (100.0%)
- Closest losses: LZ4 decompress (91%), bzip2 (87%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (251%), STREAM Add (215%), STREAM Triad (157%), SHA-256 (130%), Lua fib (124%), STREAM Copy (111%), LZ4 compress (110%), LZ4 decompress (108%)
- Closest losses: CoreMark (90%), bzip2 (85%)

**Overall best** (absolute winner per benchmark): SF wins 7, V8 wins 5, CL wins 2
- SF wins: SHA-256, mandelbrot, LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
