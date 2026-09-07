# ARM64 release startup follow-up

Date: 2026-09-07. Baseline main: `0983d9e487db6a9c50a34be6f596d6b66bf3e942`. Candidate: uncommitted release branch after tag/diagnostic cleanup and both validator optimizations, before subsequent `cargo fmt` only.

This repeats all seven actual startup benchmarks with both Nano engines against the untouched main binary. Same upstream commit, Cargo features, bench profile, ten Criterion samples, one-second warmup and two-second target measurement as [the full workload pilot](arm64-api-workload-pilot.md). Large modules require longer measurement to complete ten samples. Per-process ASLR was disabled. Adjacent process order alternates by workload. All Rust compilation finished before timing. Full samples, binary digests and candidate source digest are in [arm64-release-startup.json](arm64-release-startup.json).

A CUA state query attempted to access the locked Mac during the first FFmpeg JIT candidate measurement. That pair had large outliers and is retained in the JSON as confounded. A separate reversed-order FFmpeg JIT pair was run without other tool activity that could access the UI or compile code. The table and aggregate use this repeat; the first pair was -1.06%, and the repeat is +2.66%. This is a reason to avoid interpreting small single-pair effects as established improvements.

| Engine | Module | Main (ms) | Candidate (ms) | Elapsed change |
| --- | --- | ---: | ---: | ---: |
| jit | bz2 | 43.816263 | 45.523569 | +3.90% |
| interp | bz2 | 1.117059 | 1.989411 | +78.09% |
| jit | pulldown-cmark | 99.086868 | 101.606594 | +2.54% |
| interp | pulldown-cmark | 2.954236 | 4.980803 | +68.60% |
| jit | spidermonkey | 2111.610813 | 2172.675062 | +2.89% |
| interp | spidermonkey | 62.744581 | 106.441083 | +69.64% |
| jit | ffmpeg | 7518.357541 | 7718.519541 | +2.66% |
| interp | ffmpeg | 226.338209 | 395.152896 | +74.59% |
| jit | coremark | 4.551593 | 4.642978 | +2.01% |
| interp | coremark | 0.088885 | 0.183634 | +106.60% |
| jit | argon2 | 14.199578 | 14.587682 | +2.73% |
| interp | argon2 | 0.437516 | 0.798460 | +82.50% |
| jit | erc20 | 3.189793 | 3.236955 | +1.48% |
| interp | erc20 | 0.086928 | 0.162222 | +86.62% |

Geometric-mean elapsed changes: jit **+2.60%**, interp **+80.56%**.

The interpreter aggregate was +94.36% before the validator optimizations and is now +80.56%. These separate local pilots show partial recovery, not a passing performance gate. The large interpreter increase remains unresolved; its CoreMark increment is about 0.095 ms and ERC20 about 0.075 ms. Small JIT effects need normal differential CI confirmation. No thresholds, exclusions, validation checks or benchmark timing boundaries were relaxed.

Original local artifacts: `/tmp/sf-release-startup-final` and `/tmp/sf-release-startup-ffmpeg-repeat`. This follow-up measures startup only; execution results in the earlier 54-pair pilot describe that earlier candidate.
