# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **33,546** |    14,669 |   37,869 | 228.7% |  88.6% |
| SHA-256 (MB/s)        |      **262.54** |    249.26 |   201.20 | 105.3% | 130.5% |
| bzip2 (MB/s)          |       **17.61** |     19.41 |    19.88 |  90.7% |  88.6% |
| LZ4 compress (MB/s)   |      **734.50** |    736.45 |   704.01 |  99.7% | 104.3% |
| LZ4 decompress (MB/s) |    **3,187.24** |  3,455.15 | 2,908.07 |  92.2% | 109.6% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.627** |     12.18 |     3.14 | 463.6% | 119.5% |
| lua/sunfish (score)   |       **7,014** |     2,896 |   11,101 | 242.2% |  63.2% |
| lua/json_bench (score)|      **20,402** |     9,616 |   30,179 | 212.2% |  67.6% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **863** |       855 |    2,035 |  99.1% | 235.8% |
| c-ray (ms)            |       **2,893** |     2,055 |    1,947 |  71.0% |  67.3% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,151** |    44,124 |   39,714 | 100.1% | 111.2% |
| STREAM Scale (MB/s)   |      **49,659** |    49,692 |   18,332 |  99.9% | 270.9% |
| STREAM Add (MB/s)     |      **64,379** |    48,398 |   29,989 | 133.0% | 214.7% |
| STREAM Triad (MB/s)   |      **48,398** |    47,864 |   30,869 | 101.1% | 156.8% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 8, CL wins 5, tied 1
- SF wins: Lua fib (464%), Lua sunfish (242%), CoreMark (229%), Lua json (212%), STREAM Add (133%), SHA-256 (105%), STREAM Triad (101%), STREAM Copy (100.1%)
- Ties: STREAM Scale (99.9%)
- Closest losses: mandelbrot (99%), LZ4 compress (99.7%), LZ4 decompress (92%), bzip2 (91%), c-ray (71%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (236%), STREAM Add (215%), STREAM Triad (157%), SHA-256 (130%), Lua fib (120%), STREAM Copy (111%), LZ4 decompress (110%), LZ4 compress (104%)
- Closest losses: bzip2 (89%), CoreMark (89%), c-ray (67%), Lua json (68%), Lua sunfish (63%)

**Overall best** (absolute winner per benchmark): SF wins 5, V8 wins 5, CL wins 4
- SF wins: SHA-256, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
