# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **26,998** |    14,669 |   37,869 | 184.0% |  71.3% |
| SHA-256 (MB/s)        |      **238.12** |    249.26 |   201.20 |  95.5% | 118.3% |
| bzip2 (MB/s)          |       **14.66** |     19.41 |    19.88 |  75.5% |  73.7% |
| LZ4 compress (MB/s)   |      **670.86** |    736.45 |   704.01 |  91.1% |  95.3% |
| LZ4 decompress (MB/s) |    **2,663.12** |  3,455.15 | 2,908.07 |  77.1% |  91.6% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.587** |     12.18 |     3.14 | 470.7% | 121.4% |
| lua/sunfish (score)   |       **4,207** |     2,896 |   11,101 | 145.3% |  37.9% |
| lua/json_bench (score)|      **12,837** |     9,616 |   30,179 | 133.5% |  42.5% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |       **1,178** |       855 |    2,035 |  72.6% | 172.7% |
| c-ray (ms)            |       **5,305** |     2,055 |    1,947 |  38.7% |  36.7% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **41,471** |    44,124 |   39,714 |  94.0% | 104.4% |
| STREAM Scale (MB/s)   |      **40,291** |    49,692 |   18,332 |  81.1% | 219.8% |
| STREAM Add (MB/s)     |      **48,116** |    48,398 |   29,989 |  99.4% | 160.4% |
| STREAM Triad (MB/s)   |      **43,916** |    47,864 |   30,869 |  91.8% | 142.3% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 4, CL wins 10
- SF wins: CoreMark (184%), Lua fib (471%), Lua sunfish (145%), Lua json (134%)
- Closest losses: STREAM Add (99.4%), SHA-256 (96%), STREAM Copy (94%), LZ4 compress (91%)

**SF vs V8** (TurboFan JIT): SF wins 7, V8 wins 7
- SF wins: STREAM Scale (220%), mandelbrot (173%), STREAM Add (160%), STREAM Triad (142%), Lua fib (121%), SHA-256 (118%), STREAM Copy (104%)
- Closest losses: LZ4 compress (95%), LZ4 decompress (92%)

**Overall best** (absolute winner per benchmark): CL wins 7, V8 wins 6, SF wins 1
- SF wins: Lua fib (2.587s < V8 3.14s < CL 12.18s)
- SF nearly matches CL on STREAM Add (99.4%) and STREAM Copy (94%)

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
