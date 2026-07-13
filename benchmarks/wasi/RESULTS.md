# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

Silverfir column: 2026-07-13 (post middle-v2 campaign, mean of 2 runs).
Cranelift / V8 columns: 2026-05 capture, re-run pending (see Notes).

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **36,628** |    14,669 |   37,869 | 249.7% |  96.7% |
| SHA-256 (MB/s)        |      **275.06** |    249.26 |   201.20 | 110.4% | 136.7% |
| bzip2 (MB/s)          |       **20.48** |     19.41 |    19.88 | 105.5% | 103.0% |
| LZ4 compress (MB/s)   |      **747.27** |    736.45 |   704.01 | 101.5% | 106.1% |
| LZ4 decompress (MB/s) |    **3,200.68** |  3,455.15 | 2,908.07 |  92.6% | 110.1% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.260** |     12.18 |     3.14 | 538.9% | 138.9% |
| lua/sunfish (score)   |      **10,953** |     2,896 |   11,101 | 378.2% |  98.7% |
| lua/json_bench (score)|      **27,552** |     9,616 |   30,179 | 286.5% |  91.3% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **849** |       855 |    2,035 | 100.7% | 239.7% |
| c-ray (ms)            |       **2,101** |     2,055 |    1,947 |  97.8% |  92.7% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,071** |    44,124 |   39,714 |  99.9% | 111.0% |
| STREAM Scale (MB/s)   |      **49,574** |    49,692 |   18,332 |  99.8% | 270.4% |
| STREAM Add (MB/s)     |      **64,258** |    48,398 |   29,989 | 132.8% | 214.3% |
| STREAM Triad (MB/s)   |      **48,310** |    47,864 |   30,869 | 100.9% | 156.5% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 2, tied 2
- SF wins: Lua fib (539%), Lua sunfish (378%), Lua json (287%), CoreMark (250%), STREAM Add (133%), SHA-256 (110%), bzip2 (106%), LZ4 compress (102%), STREAM Triad (101%), mandelbrot (100.7%)
- Ties: STREAM Copy (99.9%), STREAM Scale (99.8%)
- Closest losses: c-ray (98%), LZ4 decompress (93%)

**SF vs V8** (TurboFan JIT): SF wins 10, V8 wins 4
- SF wins: STREAM Scale (270%), mandelbrot (240%), STREAM Add (214%), STREAM Triad (157%), Lua fib (139%), SHA-256 (137%), STREAM Copy (111%), LZ4 decompress (110%), LZ4 compress (106%), bzip2 (103%)
- Closest losses: Lua sunfish (99%), CoreMark (97%), c-ray (93%), Lua json (91%)

**Overall best** (absolute winner per benchmark): SF wins 7, V8 wins 4, CL wins 3
- SF wins: SHA-256, bzip2, LZ4 compress, Lua fib, mandelbrot, STREAM Add, STREAM Triad
- V8 wins: CoreMark, Lua sunfish, Lua json, c-ray
- CL wins: LZ4 decompress, STREAM Scale, STREAM Copy (by 0.1%)

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

## Addendum (2026-07-13)

- **lua/fib38 2.507 → 2.260 s (−9.9%)** is the one shift from the 2026-05
  capture clearly outside the frequency band — consistent with the
  preserved-class residency contract's lua win (−3.4k native instructions).
  Everything else moved less than ±3%.
- **SHA-256** same-session interleaved record: **277.3 ± 0.8 MB/s** (old
  code 267.9 ± 1.3, +2.5% cycles-normalized). The counter-forwarding pass
  removes clang's in-memory `ctx->datalen` store→reload chain, an Apple-M4
  pipeline hazard (dispatch stalls + exit-branch mispredicts) exposed
  whenever surrounding loads are optimized away.
- **sqlite speedtest1** newly tracked in the suite: 29.852 / 29.962 s
  (TOTAL, two runs); no Cranelift/V8 comparison yet.
- **STREAM numbers are cold-machine captures.** Sustained-load runs ramp
  ~8-10% higher over the first ~4 minutes (memory-subsystem warm-up; no
  thermal cap involved): warm steady-state ≈ Copy 48.3k, Scale 52.5k,
  Add 67.5k, Triad 51.0k MB/s — which would take all four STREAM rows
  outright.
- **Frequency caveat (applies to every absolute number in this file):** this
  M4's sustained P-core clock drifts 3.9-4.4 GHz with chip temperature (no
  throttle flag). sha256 tracks it at ~66-69 MB/s per GHz, so any
  cross-session delta under ~6% may be pure frequency. Record
  `scripts/freqprobe.c` output alongside future captures and compare
  per-GHz, or A/B interleaved within one session.
