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

`ci.performance` compares adjacent baseline/candidate samples as paired log
ratios:

- Initial blocks alternate ABAB/BABA and measure every metric. Each block
  contains two adjacent A/B pairs.
- A direction with at least 80% pilot probability enters a new, independent
  confirmation sample. The pilot chooses only the direction and is never
  reused by the final gate.
- Confirmation starts with six pairs. If the observed effect and volatility
  can still resolve within the budget, it adaptively adds pairs up to a
  maximum of 24.
- A one-sided Student-t probability classifies the confirmation sample.
  Regressions fail and improvements are reported only at 99.99% requested
  family-wide confidence. The per-look threshold applies a Bonferroni
  correction across every metric, native performance job, and possible
  adaptive confirmation look.
- Only directions selected by the initial pilot can affect the gate. New
  signals observed incidentally while rerunning a multi-metric benchmark are
  ignored.

The requested duration applies to every benchmark. In CI, CoreMark uses its
explicit `--target-seconds` regression mode, which calibrates separately from
the reported sample and labels the result non-standard. A bare CoreMark
invocation remains unchanged and keeps the official EEMBC
10-second-minimum measured interval.

If `ci.performance_build` records byte-identical baseline and candidate
executables, the run is an implicit drift calibration: measurements still
run and remain visible, but they cannot fail the gate or claim an improvement.
An apparent confirmed regression is labeled `UNSTABLE`.

`ci.performance_build` builds the two CLI executables with both checkout
roots remapped to the same virtual source path. It records executable hashes,
sizes, revisions, and whether the binaries are byte-identical in
`build-metadata.json`, which is uploaded with every performance artifact.
Before measuring, `ci.performance` verifies that metadata against the requested
revisions and the actual executable bytes. This separates source changes from
build-path/code-layout noise without trusting a stale artifact.

`dev/**` uses the native performance subset and soft-fails only to suppress
failure email. The warning annotation and job summary remain action-required.
Pull requests and `main` run the same eight native performance jobs plus JIT
and interpreter correctness jobs for ARMv7-A, RV64, and RV32 under QEMU. QEMU
executes every benchmark once on both revisions and validates its fixed oracle;
it does not calculate or gate performance deltas, and no benchmark receives a
platform-specific exclusion.

## Policy

`ci.lint_policy` rejects unaudited Rust `allow`/`expect` attributes for
warnings, dead code, and unused items, lint-lowering compiler flags, and stale
exceptions. `ci/lint_suppressions.toml` is a human-reviewed record, not a list
agents may expand to make CI green.
