# Avoid discarded validator operand collections: ARM64 diagnostic

Measured 2026-09-07 against the preceding borrowed-immediate candidate. The only runtime source change is in `module/validator/functions.rs`: `pop_vals` checks and pops operands without collecting their discarded types; `br_table` retains the original collection-and-restore path, including polymorphic `Unknown` values. No validation rules, engine boundaries or public APIs change.

Uses the same seven CI-pinned upstream Wasm inputs and local protocol as [the borrowed-immediate diagnostic](arm64-validator-borrow.md): six alternating process pairs, nine checked `Module::new` samples per process, release LTO on ARM64, per-process ASLR disabled and no concurrent builds. Each displayed time is the median of process medians. Raw samples and binary/input/source digests are in [arm64-validator-pop.json](arm64-validator-pop.json).

| Module | Before (ms) | After (ms) | Elapsed change |
| --- | ---: | ---: | ---: |
| bz2 | 0.8841 | 0.8672 | -1.92% |
| pulldown-cmark | 2.1384 | 2.0194 | -5.57% |
| spidermonkey | 47.9489 | 44.5112 | -7.17% |
| ffmpeg | 179.8689 | 171.6359 | -4.58% |
| coremark | 0.0948 | 0.0900 | -5.06% |
| erc20 | 0.0806 | 0.0756 | -6.20% |
| argon2 | 0.3614 | 0.3482 | -3.66% |

These results concern checked module construction, not full startup or execution. They do not establish x64 or CI results. The earlier full-workload interpreter startup regression remains a separate release concern. Workspace tests and the unchanged release spec suites pass without warnings (JIT 260/260, interpreter 175/175). Local logs and harnesses are `/tmp/sf-release-validator-pop-*` and `/tmp/sf-release-validation-profile/compare-pop.py`.
