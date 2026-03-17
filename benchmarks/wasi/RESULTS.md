# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **31,829** |    14,669 |   37,869 | 217.0% |  84.1% |
| SHA-256 (MB/s)        |      **238.12** |    249.26 |   201.20 |  95.5% | 118.3% |
| bzip2 (MB/s)          |       **15.77** |     19.41 |    19.88 |  81.2% |  79.3% |
| LZ4 compress (MB/s)   |      **762.89** |    736.45 |   704.01 | 103.6% | 108.4% |
| LZ4 decompress (MB/s) |    **3,199.18** |  3,455.15 | 2,908.07 |  92.6% | 110.0% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.355** |     12.18 |     3.14 | 517.0% | 133.3% |
| lua/sunfish (score)   |       **4,645** |     2,896 |   11,101 | 160.4% |  41.8% |
| lua/json_bench (score)|      **14,699** |     9,616 |   30,179 | 152.9% |  48.7% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **846** |       855 |    2,035 | 101.1% | 240.6% |
| c-ray (ms)            |       **3,949** |     2,055 |    1,947 |  52.0% |  49.3% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,127** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,644** |    49,692 |   18,332 |  99.9% | 270.8% |
| STREAM Add (MB/s)     |      **64,330** |    48,398 |   29,989 | 132.9% | 214.5% |
| STREAM Triad (MB/s)   |      **48,426** |    47,864 |   30,869 | 101.2% | 156.9% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 8, CL wins 5, tied 1
- SF wins: CoreMark (217%), Lua fib (517%), Lua sunfish (160%), Lua json (153%), STREAM Add (133%), LZ4 compress (104%), mandelbrot (101%), STREAM Triad (101%)
- Ties: STREAM Copy (100.0%), STREAM Scale (99.9%)
- Closest losses: SHA-256 (96%), LZ4 decompress (93%)

**SF vs V8** (TurboFan JIT): SF wins 9, V8 wins 5
- SF wins: STREAM Scale (271%), mandelbrot (241%), STREAM Add (215%), STREAM Triad (157%), Lua fib (133%), SHA-256 (118%), STREAM Copy (111%), LZ4 decompress (110%), LZ4 compress (108%)
- Closest losses: CoreMark (84%), bzip2 (79%)

**Overall best** (absolute winner per benchmark): SF wins 6, V8 wins 4, CL wins 4
- SF wins: mandelbrot, LZ4 compress, STREAM Copy, STREAM Add, STREAM Triad, Lua fib

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
