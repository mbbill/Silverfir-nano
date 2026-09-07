- Safe module loading validates the complete input before instantiation.
- Only the unsafe constructor accepts input whose semantic validity the caller
  has already established; instantiation does not repeat validation.

## Facts

- 2026-09-07 (f67436c1) measurement: against the earlier unchecked-default main,
  x64/ARM64 interpreter startup elapsed increased 62–89% (CoreMark about
  0.095 ms extra); four confirmed JIT cases increased 1.6–2.9%. Execution
  regressions did not survive independent confirmation. These are release-wide
  measurements, not an isolated estimate of validator overhead (code).

- 2026-09-07 statement: the compared V8 and Cranelift loaders validate input;
  wasmi eager.checked does too. Lazy or unchecked startup modes defer or omit
  work and must not be used as equivalent full-validation baselines (sourced).

- 2026-09-07 (82ac86d1) measurement: unbounded between-function validator scratch
  reuse added 196608 peak bytes on a two-function stress case. Dropping combined
  capacities above 4 KiB removed that increase; seven real modules added only
  0–655 peak bytes, with unchanged final retention. Local ARM64 parse/validate
  medians improved 0.65–5.38%; reset all function-local typing facts on reuse (code).

## Moves

- 2026-09-07 (7d1f0c8c) replaced [[optional-input-validation]]: a safe public loader must not rely on the embedder enabling semantic validation (sourced).
