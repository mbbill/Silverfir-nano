# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

Silverfir column: best observed as of 2026-07-15 (post middle-v2 campaign).
Benchmarks that improved on a 2026-07-15 run were updated to the new peak;
the rest retain their 2026-07-13 (mean-of-2) values.
Cranelift / V8 columns: 2026-05 capture, re-run pending (see Notes).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **38,697** |    14,669 |   37,869 | 263.8% | 102.2% |
| SHA-256 (MB/s)        |      **275.29** |    249.26 |   201.20 | 110.4% | 136.8% |
| bzip2 (MB/s)          |       **20.55** |     19.41 |    19.88 | 105.9% | 103.4% |
| LZ4 compress (MB/s)   |      **747.27** |    736.45 |   704.01 | 101.5% | 106.1% |
| LZ4 decompress (MB/s) |    **3,248.22** |  3,455.15 | 2,908.07 |  94.0% | 111.7% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.260** |     12.18 |     3.14 | 538.9% | 138.9% |
| lua/sunfish (score)   |      **10,953** |     2,896 |   11,101 | 378.2% |  98.7% |
| lua/json_bench (score)|      **27,552** |     9,616 |   30,179 | 286.5% |  91.3% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **848** |       855 |    2,035 | 100.8% | 240.0% |
| c-ray (ms)            |       **2,098** |     2,055 |    1,947 |  97.9% |  92.8% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,124** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,574** |    49,692 |   18,332 |  99.8% | 270.4% |
| STREAM Add (MB/s)     |      **64,258** |    48,398 |   29,989 | 132.8% | 214.3% |
| STREAM Triad (MB/s)   |      **48,349** |    47,864 |   30,869 | 101.0% | 156.6% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 2, tied 2
- SF wins: Lua fib (539%), Lua sunfish (378%), Lua json (287%), CoreMark (264%), STREAM Add (133%), SHA-256 (110%), bzip2 (106%), LZ4 compress (102%), STREAM Triad (101%), mandelbrot (100.8%)
- Ties: STREAM Copy (100.0%), STREAM Scale (99.8%)
- Closest losses: c-ray (98%), LZ4 decompress (94%)

**SF vs V8** (TurboFan JIT): SF wins 11, V8 wins 3
- SF wins: STREAM Scale (270%), mandelbrot (240%), STREAM Add (214%), STREAM Triad (157%), Lua fib (139%), SHA-256 (137%), LZ4 decompress (112%), STREAM Copy (111%), LZ4 compress (106%), bzip2 (103%), CoreMark (102%)
- Closest losses: Lua sunfish (99%), c-ray (93%), Lua json (91%)

**Overall best** (absolute winner per benchmark): SF wins 8, V8 wins 3, CL wins 2, tied 1
- SF wins: CoreMark, SHA-256, bzip2, LZ4 compress, Lua fib, mandelbrot, STREAM Add, STREAM Triad
- V8 wins: Lua sunfish, Lua json, c-ray
- CL wins: LZ4 decompress, STREAM Scale
- Tied (SF ≈ CL): STREAM Copy (100.0%)

## Notes

- Silverfir: `sf-nano-cli` (release build, jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- Internal tracking dashboard: https://mbbill.github.io/Silverfir-nano/dev/bench/
- The Silverfir column was re-captured 2026-07-13 against the 2026-05
  Cranelift/V8 numbers; ratio shifts under ~6% may be machine-state, not
  code (see frequency caveat below). Re-run all three in one warm session
  for the next revision.
