# ARM64 public API release workload pilot

Date: 2026-09-07. Baseline: main `0983d9e487db6a9c50a34be6f596d6b66bf3e942`.
Candidate: uncommitted `codex/public-api-release`, after temporary value buffers and before the diagnostic/limits API cleanup.

Pinned wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`. Both Nano engine features were compiled in both binaries, using the same Cargo bench profile. No competitor runtime was enabled. Each workload/engine received one adjacent A/B pair, alternating process order between workloads. macOS ASLR was disabled for each process. All compilation finished before measurement. The original 10-sample, 1-second warmup, 2-second measurement Criterion settings were retained.

This is a local pilot, not the single-engine CI gate or a confirmed regression verdict. Values below are medians of the ten per-iteration elapsed-time samples. Positive changes mean slower. Execution oracles and invalid-input contracts are covered by separate correctness tests; timing success alone is not their proof.

## Aggregate elapsed-time changes

| Engine | Execute, 20 cases | Startup, 7 cases |
| --- | ---: | ---: |
| jit | +0.21% | +2.78% |
| interp | +0.03% | +94.36% |

## Per-case medians

| Engine | Workload | Main (ms) | Candidate (ms) | Elapsed change |
| --- | --- | ---: | ---: | ---: |
| jit | execute/counter-local | 0.311489 | 0.315616 | +1.33% |
| interp | execute/counter-local | 1.029014 | 1.037035 | +0.78% |
| jit | execute/counter-param | 0.313207 | 0.317223 | +1.28% |
| interp | execute/counter-param | 1.032220 | 1.037543 | +0.52% |
| jit | execute/counter-global | 0.157742 | 0.156754 | -0.63% |
| interp | execute/counter-global | 1.326227 | 1.331341 | +0.39% |
| jit | execute/fibonacci-rec | 2.139893 | 2.154191 | +0.67% |
| interp | execute/fibonacci-rec | 13.260747 | 13.209192 | -0.39% |
| jit | execute/fibonacci-iter | 0.635867 | 0.632583 | -0.52% |
| interp | execute/fibonacci-iter | 4.147633 | 4.182686 | +0.85% |
| jit | execute/fibonacci-tail | 0.316483 | 0.317257 | +0.24% |
| interp | execute/fibonacci-tail | 5.059442 | 5.045352 | -0.28% |
| jit | execute/sort | 15.071392 | 15.114755 | +0.29% |
| interp | execute/sort | 156.834334 | 157.018063 | +0.12% |
| jit | execute/prime_sieve | 21.369904 | 21.409495 | +0.19% |
| interp | execute/prime_sieve | 89.318958 | 89.245896 | -0.08% |
| jit | execute/matrix_mul | 48.407484 | 48.556971 | +0.31% |
| interp | execute/matrix_mul | 228.909230 | 227.909416 | -0.44% |
| jit | execute/nbody | 6.576057 | 6.562681 | -0.20% |
| interp | execute/nbody | 56.235330 | 55.936129 | -0.53% |
| jit | execute/argon2 | 23.878368 | 23.809309 | -0.29% |
| interp | execute/argon2 | 529.686126 | 527.714480 | -0.37% |
| jit | execute/tiny_keccak | 0.009044 | 0.009075 | +0.34% |
| interp | execute/tiny_keccak | 0.101679 | 0.101431 | -0.24% |
| jit | execute/mandelbrot | 18.353506 | 18.198617 | -0.84% |
| interp | execute/mandelbrot | 42.462912 | 42.545211 | +0.19% |
| jit | execute/spectralnorm | 9.186338 | 9.177255 | -0.10% |
| interp | execute/spectralnorm | 71.124941 | 71.147137 | +0.03% |
| jit | execute/compression | 6.366016 | 6.364518 | -0.02% |
| interp | execute/compression | 24.212196 | 24.268686 | +0.23% |
| jit | execute/word_count | 0.707605 | 0.708693 | +0.15% |
| interp | execute/word_count | 3.922845 | 3.894964 | -0.71% |
| jit | execute/json_parse | 3.394396 | 3.383944 | -0.31% |
| interp | execute/json_parse | 28.212199 | 28.254397 | +0.15% |
| jit | execute/reverse_complement | 0.007626 | 0.007675 | +0.65% |
| interp | execute/reverse_complement | 0.131554 | 0.132014 | +0.35% |
| jit | execute/regex_redux | 0.022871 | 0.023044 | +0.76% |
| interp | execute/regex_redux | 0.077191 | 0.077736 | +0.71% |
| jit | execute/bulk-ops | 0.590090 | 0.595058 | +0.84% |
| interp | execute/bulk-ops | 0.507761 | 0.504958 | -0.55% |
| jit | startup/bz2 | 44.459928 | 45.957773 | +3.37% |
| interp | startup/bz2 | 1.117294 | 2.133983 | +91.00% |
| jit | startup/pulldown-cmark | 99.851493 | 103.129094 | +3.28% |
| interp | startup/pulldown-cmark | 2.955070 | 5.379343 | +82.04% |
| jit | startup/spidermonkey | 2129.750812 | 2185.512187 | +2.62% |
| interp | startup/spidermonkey | 62.837488 | 114.716792 | +82.56% |
| jit | startup/ffmpeg | 7597.926271 | 7847.222750 | +3.28% |
| interp | startup/ffmpeg | 224.896104 | 423.464250 | +88.29% |
| jit | startup/coremark | 4.585753 | 4.658224 | +1.58% |
| interp | startup/coremark | 0.089959 | 0.199550 | +121.82% |
| jit | startup/argon2 | 14.151400 | 14.715665 | +3.99% |
| interp | startup/argon2 | 0.439418 | 0.857682 | +95.19% |
| jit | startup/erc20 | 3.195560 | 3.240399 | +1.40% |
| interp | startup/erc20 | 0.086809 | 0.175781 | +102.49% |

## Evidence

Candidate source digest recorded before compilation/measurement: `457cfeb9521704d8ad1aa12a43efe47757f54ecd63ecbf72a51e8f697b4ad147`.
Baseline binary SHA-256: `7efe1234b72a1ddc7eac893d3eb858686787b74596ef29b43be0f19d83cb5d18`.
Candidate binary SHA-256: `73c398776d89dc85ac48ba80eefed35ecd8a3874ee0680debac2d454b9cb257a`.

Local commands, run logs, Criterion raw files and complete 54-pair sample metadata: `/tmp/sf-release-workload-pilot/`. The repository JSON companion preserves the normalized per-iteration samples for each case. These artifacts do not certify later source edits.

Interpreter startup increased substantially across all seven cases. A separate parse/validation diagnostic attributes approximately 0.085–193 ms of added module-loading cost to semantic validation, depending on the module. The unchecked diagnostic first validates the exact bytes outside timing and never changes the checked upstream benchmark path. Further optimization or an explicit release decision remains necessary; no thresholds or exclusions were relaxed.
