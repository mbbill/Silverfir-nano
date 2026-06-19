- Compare-and-branch fusion is a shared MachineIR peephole that runs last and
  across the whole program: an IntCompare last-op whose result is consumed only
  by the block's Branch terminator and proven dead at both successors' entries
  is folded into a Branch carrying the compare condition; every backend
  eliminates the boolean materialization (CSET on ARM64, SETCC+MOVZX on x86_64)
  and emits a hardware compare-and-branch (`fuse_compare_branch`).

- The fusion is limited to integer/test conditions and never fuses float
  compares; float-compare-and-branch is not fused anywhere.

- The fusion fires only when the compare-result register is transient.

## Facts

- 2026-03-18 (769c3794) statement: the fusion is limited to integer/test
  conditions and never fuses float compares; backends where
  float-compare-and-branch is single-instruction safe (ARM64 FCMP+B.cond) must
  do that fusion locally (code).

- 2026-03-18 (769c3794) statement: the fusion is deliberately limited to
  integer/test conditions because x86_64 Wasm float comparison requires
  multi-instruction NaN handling (SETCC+SETNP+AND) that cannot collapse into one
  conditional branch (sourced).

- 2026-03-23 (6b4ba56e) rationale: the shared peephole deliberately does not
  fuse float compares because x86_64 requires multi-instruction NaN handling
  that cannot be expressed as a single conditional branch; ARM64 instead fuses
  FCMP+B.cond locally in its own backend codegen rather than in the shared
  peephole (sourced).

- 2026-03-26 (da4a5eaa) pitfall: fusion proves the compare-result register
  dead via a single-successor liveness check, but that check is valid only for
  transient registers — cached-local and fixed registers are implicitly live
  across every block boundary, so fusing when the result lands in such a
  register drops a still-needed value and miscompiles; fusion must be gated on
  the compare-result register being transient (surfaced as a SQLite failure)
  (code).

- 2026-03-26 (0d429c9e) rationale: the transient guard holds because
  successor-block dead-at-entry liveness is sufficient to prove the result dead
  only for a transient; ARM64's arch-backend float-compare-branch fusion
  carried the same transient guard while it existed (code).

## Moves

- 2026-03-18 (769c3794) replaced [[shared-int-float-fusion]]: a shared
  FloatCompare+Branch fold is unsafe for x86_64, whose Wasm float comparison
  needs multi-instruction NaN handling (SETCC+SETNP+AND) that cannot be
  expressed as a single conditional branch, so the shared peephole is
  restricted to integer compares and float-compare-and-branch fusion is
  reinstated only in the ARM64 backend, where FCMP condition codes give the
  correct NaN behavior with a single B.cond (code).

- 2026-03-29 (072ef2b5) dropped: ARM64-backend-private float compare-branch
  fusion: a backend performing its own MachineIR-level compare-branch fusion
  that the shared peephole deliberately declines is a layering violation
  (structural choices belong in the shared stages, only small late rewrites in
  the backend); float compares are now simply not fused anywhere rather than
  fused privately in one backend (code).
