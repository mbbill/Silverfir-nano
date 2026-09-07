# Borrowed validator immediates: ARM64 diagnostic

Measured 2026-09-07 against the immediately preceding uncommitted release candidate. Only `sf-nano-core/src/module/validator/functions.rs` differs between the two builds. The validator borrows decoded immediates, branch labels, catch clauses and typed-select types; it checks the same operands and labels in the same order. No decoder, runtime, public API or feature-boundary changes are involved.

The seven real Wasm inputs come from the CI-pinned wasmi-benchmarks checkout `16a3d7c8fdb05506c116a9451175732d1ac77099`. Native ARM64 macOS, rustc 1.98.1, release LTO, one codegen unit, debug info level 1. Both binaries finished building before timing; each process uses the repository no-ASLR wrapper. Six alternating before/after process pairs per module each produce nine checked construction samples. Reported values are medians of process medians. Full samples and binary/input/source digests are in [arm64-validator-borrow.json](arm64-validator-borrow.json).

| Module | Before (ms) | Borrowed (ms) | Elapsed change |
| --- | ---: | ---: | ---: |
| bz2 | 1.0019 | 0.8886 | -11.30% |
| pulldown-cmark | 2.3930 | 2.1903 | -8.47% |
| spidermonkey | 54.1716 | 49.5696 | -8.50% |
| ffmpeg | 201.3475 | 181.5683 | -9.82% |
| coremark | 0.1033 | 0.0956 | -7.46% |
| erc20 | 0.0905 | 0.0817 | -9.73% |
| argon2 | 0.4005 | 0.3606 | -9.95% |

This measures safe `Module::new`, including parsing, full validation, input ownership and dropping the resulting module. It does not measure complete engine instantiation or execution, does not establish x64 results, and does not replace the performance CI gate. In particular, it only reduces part of the unresolved interpreter startup regression documented in the full workload pilot.

The diagnostic establishes validity with `Module::new` before its separate unsafe parse-only samples; those parse-only samples are not used in this comparison. Each checked timing still performs complete validation. The local harness, build logs and original CSV files are under `/tmp/sf-release-validation-profile` and `/tmp/sf-release-validator-borrow-*`.
