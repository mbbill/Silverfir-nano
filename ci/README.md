# CI implementation

This directory owns CI validation. Local development should normally
use `cargo build` and focused `cargo test` commands.

## Correctness jobs

`.github/workflows/correctness.yml` starts one independent runner per logical
platform:

| Job kind | Platforms | Coverage |
|---|---|---|
| Native host | x64 Linux, ARM64 Linux, ARM64 macOS, x64 Windows MSVC | explicit debug/release workspace builds, debug unit tests, and release JIT/interpreter spec/WASI; x64 Linux also checks the engine feature boundaries and one combined diagnostic-feature smoke |
| Linux QEMU-user | ARMv7/Thumb-2, RV64 Linux, RV32 Linux | target builds plus JIT and interpreter-tier spec/WASI execution; pure-interpreter compilation is checked separately |
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
  confirmation rounds and fail only when all four rounds remain below -1%
  and the pooled adjacent pairs pass a one-sided exact sign test (`p <= 0.05`;
  normally at least 7 of 8 pairs below -1%).
- Improvement candidates above +3% are reported only when all four rounds
  remain above +3% and pass the same pair-consistency test; improvements never
  fail CI.
- Metrics that were not initial candidates cannot enter the gate later.

A regression candidate whose four round geomeans all cross -1% but whose
individual adjacent pairs are contradictory is reported as `UNSTABLE` and
does not fail CI. This keeps noisy runners visible without treating drift as
a source regression.

If `ci.performance_build` records byte-identical baseline and candidate
executables, the run is an implicit drift calibration: measurements still
run and remain visible, but they cannot fail the gate or claim an improvement.

`ci.performance_build` builds the two CLI executables with both checkout
roots remapped to the same virtual source path. It records executable hashes,
sizes, revisions, and whether the binaries are byte-identical in
`build-metadata.json`, which is uploaded with every performance artifact.
This separates source changes from build-path/code-layout noise.

`dev/**` uses the native performance subset and soft-fails only to suppress
failure email. The warning annotation and job summary remain action-required.
Pull requests and `main` use the full native and cross-target performance
matrix. STREAM remains enabled for native and cross-JIT jobs, but is excluded
from cross-interpreter jobs: nested QEMU/interpreter execution measures
emulator memory-loop overhead (over two minutes per Armv7 sample) rather than
a useful target signal.

## Policy

`ci.lint_policy` rejects unaudited Rust `allow`/`expect` attributes for
warnings, dead code, and unused items, lint-lowering compiler flags, and stale
exceptions. `ci/lint_suppressions.toml` is a human-reviewed record, not a list
agents may expand to make CI green.
