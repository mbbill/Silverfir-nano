# CI implementation

This directory owns exhaustive validation. Local development should normally
use `cargo build` and focused `cargo test` commands.

## Correctness jobs

`.github/workflows/correctness.yml` starts one independent runner per logical
platform:

| Job kind | Platforms | Coverage |
|---|---|---|
| Native host | x64 Linux, ARM64 Linux, ARM64 macOS, x64 Windows | workspace debug/release build and tests, complete core/CLI feature matrix, native JIT and interpreter spec/WASI |
| Linux QEMU-user | ARMv7/Thumb-2, RV64 Linux, RV32 Linux | target builds plus JIT and pure-interpreter spec/WASI execution |
| Bare-metal compile | ARMv8-M `thumbv8m`, RV32IMAC `none-elf` | real `cargo build` and JIT/interpreter assembler coverage; no fake runtime claim |

All cross-runtime jobs intentionally use x64 Linux. macOS Colima, WSL
re-execution, platform skipping, and reduced local modes are not part of the
CI implementation.

Every command runs to completion where its dependencies permit. A compiler
warning is recorded as a correctness failure, but a successful build artifact
may still be used by later runtime tests so one warning does not hide the rest
of the diagnostics. The final report groups duplicate diagnostics and lists
every distinct issue.

Entry points:

```text
python -m ci.correctness host x64-linux
python -m ci.correctness cross riscv64
python -m ci.correctness bare thumbv8m
python -m ci.lint_policy
```

## Performance jobs

`ci.performance` performs the staged alternating comparison:

- Initial ABAB/BABA block for every metric.
- Regression candidates below -1% receive up to three target-only
  confirmation rounds and fail only when all four rounds remain below -1%.
- Improvement candidates above +3% are reported only when all four rounds
  remain above +3%; improvements never fail CI.
- Metrics that were not initial candidates cannot enter the gate later.

`dev/**` uses the native performance subset and soft-fails only to suppress
failure email. The warning annotation and job summary remain action-required.
Pull requests and `main` use the full native and cross-target performance
matrix.

## Policy

`ci.lint_policy` rejects unaudited Rust `allow`/`expect` attributes for
warnings, dead code, and unused items, lint-lowering compiler flags, and stale
exceptions. `ci/lint_suppressions.toml` is a human-reviewed record, not a list
agents may expand to make CI green.
