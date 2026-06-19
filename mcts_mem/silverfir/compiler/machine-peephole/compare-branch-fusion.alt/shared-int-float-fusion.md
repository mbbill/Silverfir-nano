- The shared MachineIR compare-and-branch peephole folds both IntCompare and
  FloatCompare last-ops into a Branch carrying a MachineBranchCond::IntCompare
  or MachineBranchCond::FloatCompare, applied uniformly to every backend.

## Moves

- 2026-03-18 (767099b9) replaced [[arm64-local-compare-branch-fusion]]: keeping
  the fusion inside the ARM64 backend benefited no other target; hoisting it
  into the shared MachineIR peephole as a cross-block pass that folds
  IntCompare/FloatCompare+Branch(Reg) into a Branch carrying the compare
  condition lets every backend eliminate the boolean materialization (CSET on
  ARM64, SETCC+MOVZX on x86_64) and emit a hardware compare-and-branch (code).

- 2026-03-18 (769c3794) replaced by [[compare-branch-fusion]]: a shared
  FloatCompare+Branch fold is unsafe for x86_64, whose Wasm float comparison
  needs multi-instruction NaN handling (SETCC+SETNP+AND) that cannot be
  expressed as a single conditional branch, so the shared peephole is
  restricted to integer compares and float-compare-and-branch fusion is
  reinstated only in the ARM64 backend, where FCMP condition codes give the
  correct NaN behavior with a single B.cond (code).
