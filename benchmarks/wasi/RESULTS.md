# WASI Benchmark Results

Run with `run_tests.py` on macOS (Apple M4).

Three-way comparison table captured 2026-05 (pre middle-v2 campaign). For the
current Silverfir numbers see the 2026-07-13 section at the bottom; Cranelift
and V8 have not been re-run since 2026-05, so the ratio columns below are
historical.

## Integer / Control Flow

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| CoreMark (score)      |      **37,032** |    14,669 |   37,869 | 252.4% |  97.8% |
| SHA-256 (MB/s)        |      **271.04** |    249.26 |   201.20 | 108.7% | 134.7% |
| bzip2 (MB/s)          |       **20.23** |     19.41 |    19.88 | 104.2% | 101.8% |
| LZ4 compress (MB/s)   |      **768.21** |    736.45 |   704.01 | 104.3% | 109.1% |
| LZ4 decompress (MB/s) |    **3,214.90** |  3,455.15 | 2,908.07 |  93.0% | 110.6% |

### Lua

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| lua/fib38 (s)         |       **2.507** |     12.18 |     3.14 | 485.8% | 125.2% |
| lua/sunfish (score)   |      **10,924** |     2,896 |   11,101 | 377.2% |  98.4% |
| lua/json_bench (score)|      **27,884** |     9,616 |   30,179 | 290.0% |  92.4% |

## Floating Point

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| mandelbrot (ms)       |         **854** |       855 |    2,035 | 100.2% | 238.4% |
| c-ray (ms)            |       **2,140** |     2,055 |    1,947 |  96.0% |  91.0% |

## Memory Bound

| Benchmark             | Silverfir (JIT) | Cranelift |       V8 |  SF/CL |  SF/V8 |
|-----------------------|----------------:|----------:|---------:|-------:|-------:|
| STREAM Copy (MB/s)    |      **44,124** |    44,124 |   39,714 | 100.0% | 111.1% |
| STREAM Scale (MB/s)   |      **49,629** |    49,692 |   18,332 |  99.9% | 270.7% |
| STREAM Add (MB/s)     |      **64,256** |    48,398 |   29,989 | 132.8% | 214.3% |
| STREAM Triad (MB/s)   |      **48,387** |    47,864 |   30,869 | 101.1% | 156.7% |

## Summary

**SF vs Cranelift** (optimizing JIT): SF wins 10, CL wins 2, tied 2
- SF wins: Lua fib (486%), Lua sunfish (377%), Lua json (290%), CoreMark (252%), STREAM Add (133%), SHA-256 (109%), LZ4 compress (104%), bzip2 (104%), STREAM Triad (101%), mandelbrot (100.2%)
- Ties: STREAM Copy (100.0%), STREAM Scale (99.9%)
- Closest losses: c-ray (96%), LZ4 decompress (93%)

**SF vs V8** (TurboFan JIT): SF wins 10, V8 wins 4
- SF wins: STREAM Scale (271%), mandelbrot (238%), STREAM Add (214%), STREAM Triad (157%), SHA-256 (134%), Lua fib (125%), STREAM Copy (111%), LZ4 decompress (111%), LZ4 compress (109%), bzip2 (102%)
- Closest losses: Lua sunfish (98%), CoreMark (98%), Lua json (92%), c-ray (91%)

**Overall best** (absolute winner per benchmark): SF wins 8, V8 wins 4, CL wins 2
- SF wins: SHA-256, LZ4 compress, bzip2, Lua fib, mandelbrot, STREAM Copy, STREAM Add, STREAM Triad
- V8 wins: CoreMark, Lua sunfish, Lua json, c-ray
- CL wins: LZ4 decompress, STREAM Scale

## Notes

- Silverfir: `sf-nano-cli` (release build, jit, `main` branch)
- Cranelift: wasmtime (`-C compiler=cranelift`, optimizing JIT)
- V8: Node.js 25.4.0, V8 14.1.146.11 (`run_v8.mjs`)
- Higher is better for score/MB/s metrics; lower is better for ms/s metrics
- Internal tracking dashboard: https://mbbill.github.io/Silverfir-nano/dev/bench/

## Current HEAD (2026-07-13, post middle-v2 campaign)

Full-suite capture after the middle-end v2 refactor + preserved-class
residency contract + bounded-counter forwarding pass. Two back-to-back runs
(run-to-run spread <1% everywhere); Silverfir only — Cranelift/V8 not re-run.

| Benchmark             |   Run 1 |   Run 2 | vs 2026-05 SF |
|-----------------------|--------:|--------:|--------------:|
| CoreMark (score)      |  36,810 |  36,445 |         −1.1% |
| SHA-256 (MB/s)        |  274.40 |  275.71 |         +1.5% |
| bzip2 (MB/s)          |   20.47 |   20.48 |         +1.2% |
| LZ4 compress (MB/s)   |  745.58 |  748.96 |         −2.7% |
| LZ4 decompress (MB/s) | 3,216.8 | 3,184.6 |         −0.4% |
| lua/fib38 (s)         |   2.270 |   2.250 |     **−9.9%** |
| lua/sunfish (score)   |  10,959 |  10,947 |         +0.3% |
| lua/json_bench (score)|  27,548 |  27,556 |         −1.2% |
| mandelbrot (ms)       |  849.14 |  849.35 |         −0.6% |
| c-ray (ms)            |   2,099 |   2,103 |         −1.8% |
| STREAM C/S/A/T (MB/s) | 44,017 / 49,534 / 64,256 / 48,213 | 44,124 / 49,615 / 64,260 / 48,408 | level |
| sqlite speedtest1 (s) |  29.852 |  29.962 |     new entry |

Notes:

- **lua/fib38 −9.9% wall clock** is the one shift clearly outside the
  frequency band — consistent with the preserved-class contract's lua win
  (−3.4k native instructions). Everything else is within ±3%.
- **SHA-256** same-session interleaved record is **277.3 ± 0.8 MB/s** (vs
  old code 267.9 ± 1.3, +2.5% cycles-normalized); the 274-276 here is the
  same code in a different frequency session. The counter-forwarding pass
  removes clang's in-memory `ctx->datalen` store→reload chain, an Apple-M4
  pipeline hazard (dispatch stalls + exit-branch mispredicts) exposed
  whenever surrounding loads are optimized away.
- **sqlite speedtest1** is newly in the suite; no earlier baseline row.
- **STREAM numbers are cold-machine captures.** Sustained-load runs ramp
  ~8-10% higher over the first ~4 minutes (memory-subsystem warm-up; no
  thermal cap involved): warm steady-state ≈ Copy 48.3k, Scale 52.5k,
  Add 67.5k, Triad 51.0k MB/s — which would take all four STREAM rows
  outright against the 2026-05 CL/V8 columns.
- **Frequency caveat (applies to every absolute number in this file):** this
  M4's sustained P-core clock drifts 3.9-4.4 GHz with chip temperature (no
  throttle flag). sha256 tracks it at ~66-69 MB/s per GHz, so any
  cross-session delta under ~6% may be pure frequency. Future captures:
  record `scripts/freqprobe.c` output alongside and compare per-GHz, or A/B
  interleaved within one session.
- Next revision: re-run Cranelift and V8 in the same session (warm) to
  refresh the ratio columns above.
