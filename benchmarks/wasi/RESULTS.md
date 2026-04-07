# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **34,325** |    14,669 |   37,869 | 234.0% |  90.6% |
| SHA-256 (MB/s)        |      **275.32** |    249.26 |   201.20 | 110.5% | 136.8% |
| bzip2 (MB/s)          |       **18.13** |     19.41 |    19.88 |  93.4% |  91.2% |
| LZ4 compress (MB/s)   |      **765.71** |    736.45 |   704.01 | 104.0% | 108.8% |
| LZ4 decompress (MB/s) |    **3,199.12** |  3,455.15 | 2,908.07 |  92.6% | 110.0% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.631** |     12.18 |     3.14 | 462.9% | 119.3% |
| lua/sunfish (score)   |       **7,440** |     2,896 |   11,101 | 256.9% |  67.0% |
| lua/json_bench (score)|      **21,292** |     9,616 |   30,179 | 221.4% |  70.6% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **852** |       855 |    2,035 | 100.4% | 238.9% |
| c-ray (ms)            |       **2,785** |     2,055 |    1,947 |  73.8% |  69.9% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,162** |    44,124 |   39,714 | 100.1% | 111.2% |
| STREAM Scale (MB/s)   |      **49,674** |    49,692 |   18,332 | 100.0% | 271.0% |
| STREAM Add (MB/s)     |      **64,433** |    48,398 |   29,989 | 133.1% | 214.9% |
| STREAM Triad (MB/s)   |      **48,435** |    47,864 |   30,869 | 101.2% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 9, CL wins 4, tied 1
- SF wins: Lua fib (463%), Lua sunfish (257%), CoreMark (234%), Lua json (221%), STREAM Add (133%), SHA-256 (111%), LZ4 compress (104%), STREAM Triad (101%), mandelbrot (100.4%), STREAM Copy (100.1%)
- Ties: STREAM Scale (100.0%)
- Closest losses: LZ4 decompress (93%), bzip2 (93%), c-ray (74%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (239%), STREAM Add (215%), STREAM Triad (157%), SHA-256 (137%), Lua fib (119%), STREAM Copy (111%), LZ4 decompress (110%), LZ4 compress (109%)
- Closest losses: bzip2 (91%), CoreMark (91%), Lua json (71%), c-ray (70%), Lua sunfish (67%)

**Overall best** (absolute winner per benchmark): SF wins 7, V8 wins 4, CL wins 3
- SF wins: SHA-256, LZ4 compress, mandelbrot, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
