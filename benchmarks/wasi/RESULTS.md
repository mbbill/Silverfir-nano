# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **31,668** |    14,669 |   37,869 | 215.9% |  83.6% |
| SHA-256 (MB/s)        |      **238.12** |    249.26 |   201.20 |  95.5% | 118.3% |
| bzip2 (MB/s)          |       **15.45** |     19.41 |    19.88 |  79.6% |  77.7% |
| LZ4 compress (MB/s)   |      **752.44** |    736.45 |   704.01 | 102.2% | 106.9% |
| LZ4 decompress (MB/s) |    **3,183.87** |  3,455.15 | 2,908.07 |  92.1% | 109.5% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.398** |     12.18 |     3.14 | 507.9% | 130.9% |
| lua/sunfish (score)   |       **4,259** |     2,896 |   11,101 | 147.1% |  38.4% |
| lua/json_bench (score)|      **12,996** |     9,616 |   30,179 | 135.2% |  43.1% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |       **1,178** |       855 |    2,035 |  72.6% | 172.7% |
| c-ray (ms)            |       **5,219** |     2,055 |    1,947 |  39.4% |  37.3% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,127** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **46,148** |    49,692 |   18,332 |  92.9% | 251.7% |
| STREAM Add (MB/s)     |      **60,408** |    48,398 |   29,989 | 124.8% | 201.4% |
| STREAM Triad (MB/s)   |      **48,408** |    47,864 |   30,869 | 101.1% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 7, CL wins 7
- SF wins: CoreMark (216%), Lua fib (508%), Lua sunfish (147%), Lua json (135%), STREAM Add (125%), LZ4 compress (102%), STREAM Triad (101%)
- Ties: STREAM Copy (100.0%)
- Closest losses: SHA-256 (96%), STREAM Scale (93%), LZ4 decompress (92%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (252%), STREAM Add (201%), mandelbrot (173%), STREAM Triad (157%), Lua fib (131%), SHA-256 (118%), STREAM Copy (111%), LZ4 decompress (110%), LZ4 compress (107%)
- Closest losses: CoreMark (84%), bzip2 (78%)

**Overall best** (absolute winner per benchmark): SF wins 5, V8 wins 5, CL wins 4
- SF wins: LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
