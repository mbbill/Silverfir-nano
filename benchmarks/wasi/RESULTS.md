# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **26,031** |    14,669 |   37,869 | 177.4% |  68.7% |
| SHA-256 (MB/s)        |      **213.75** |    249.26 |   201.20 |  85.7% | 106.2% |
| bzip2 (MB/s)          |       **13.41** |     19.41 |    19.88 |  69.1% |  67.5% |
| LZ4 compress (MB/s)   |      **670.86** |    736.45 |   704.01 |  91.1% |  95.3% |
| LZ4 decompress (MB/s) |    **2,226.68** |  3,455.15 | 2,908.07 |  64.4% |  76.6% |

### Lua

| lua/fib38 (s)         |       **3.119** |     12.18 |     3.14 | 390.5% | 100.7% |
| lua/sunfish (score)   |       **3,906** |     2,896 |   11,101 | 134.9% |  35.2% |
| lua/json_bench (score)|      **12,220** |     9,616 |   30,179 | 127.1% |  40.5% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |       **1,178** |       855 |    2,035 |  72.6% | 172.7% |
| c-ray (ms)            |       **5,516** |     2,055 |    1,947 |  37.3% |  35.3% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **23,509** |    44,124 |   39,714 |  53.3% |  59.2% |
| STREAM Scale (MB/s)   |      **29,331** |    49,692 |   18,332 |  59.0% | 160.0% |
| STREAM Add (MB/s)     |      **34,623** |    48,398 |   29,989 |  71.5% | 115.5% |
| STREAM Triad (MB/s)   |      **31,266** |    47,864 |   30,869 |  65.3% | 101.3% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 4, CL wins 10
- SF wins: CoreMark (177%), Lua fib (391%), Lua sunfish (135%), Lua json (127%)
- Closest losses: LZ4 compress (91%), SHA-256 (86%), mandelbrot (73%)

**SF vs V8** (TurboFan JIT): SF wins 6, V8 wins 8
- SF wins: mandelbrot (173%), STREAM Scale (160%), STREAM Add (116%), SHA-256 (106%), STREAM Triad (101%), Lua fib (101%)
- Closest losses: LZ4 compress (95%), CoreMark (69%)

**Overall best** (absolute winner per benchmark): CL wins 8, V8 wins 5, SF wins 1
- SF wins: Lua fib (3.119s < V8 3.14s < CL 12.18s)
- CL dominates compute-heavy and memory-bound workloads
- V8 dominates interpreter-like workloads (Lua sunfish/json) and CoreMark

## Notes

- Silverfir: `sf-nano-cli` (release build, micro-jit, `microjit` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- **Bold** = best interpreter result
- c-ray: 4000x4000 resolution
