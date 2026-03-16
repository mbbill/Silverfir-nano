# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **31,546** |    14,669 |   37,869 | 215.0% |  83.3% |
| SHA-256 (MB/s)        |      **238.12** |    249.26 |   201.20 |  95.5% | 118.3% |
| bzip2 (MB/s)          |       **15.13** |     19.41 |    19.88 |  77.9% |  76.1% |
| LZ4 compress (MB/s)   |      **736.64** |    736.45 |   704.01 | 100.0% | 104.6% |
| LZ4 decompress (MB/s) |    **3,183.87** |  3,455.15 | 2,908.07 |  92.1% | 109.5% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.476** |     12.18 |     3.14 | 491.9% | 126.8% |
| lua/sunfish (score)   |       **4,207** |     2,896 |   11,101 | 145.3% |  37.9% |
| lua/json_bench (score)|      **12,837** |     9,616 |   30,179 | 133.5% |  42.5% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |       **1,178** |       855 |    2,035 |  72.6% | 172.7% |
| c-ray (ms)            |       **5,274** |     2,055 |    1,947 |  39.0% |  36.9% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,116** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **46,136** |    49,692 |   18,332 |  92.8% | 251.7% |
| STREAM Add (MB/s)     |      **60,408** |    48,398 |   29,989 | 124.8% | 201.4% |
| STREAM Triad (MB/s)   |      **48,408** |    47,864 |   30,869 | 101.1% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 6, CL wins 8
- SF wins: CoreMark (215%), Lua fib (492%), Lua sunfish (145%), Lua json (134%), STREAM Add (125%), STREAM Triad (101%)
- Ties: LZ4 compress (100.0%), STREAM Copy (100.0%)
- Closest losses: SHA-256 (96%), STREAM Scale (93%), LZ4 decompress (92%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (252%), STREAM Add (201%), mandelbrot (173%), STREAM Triad (157%), Lua fib (127%), SHA-256 (118%), STREAM Copy (111%), LZ4 decompress (110%), LZ4 compress (105%)
- Closest losses: CoreMark (83%), bzip2 (76%)

**Overall best** (absolute winner per benchmark): CL wins 5, V8 wins 5, SF wins 4
- SF wins: CoreMark, Lua fib, STREAM Add, STREAM Triad
- SF ties CL: LZ4 compress, STREAM Copy

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
