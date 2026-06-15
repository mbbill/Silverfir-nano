- Compare-and-branch fusion is an ARM64 backend-local rewrite that runs while
  emitting each block: an IntCompare last op whose dst is consumed only by the
  Branch terminator and proven dead at both successors' entries emits CMP
  without CSET, and the terminator emits B.cond directly off the set flags.

- Successor liveness is established by scanning each target block for whether
  the compare's result register is defined-before-used, an edge parameter, or
  never touched.

## Moves

- 2026-03-18 (767099b9) replaced by [[shared-int-float-fusion]]: keeping the
  fusion inside the ARM64 backend benefited no other target; hoisting it into
  the shared MachineIR peephole as a cross-block pass that folds
  IntCompare/FloatCompare+Branch(Reg) into a Branch carrying the compare
  condition lets every backend eliminate the boolean materialization (CSET on
  ARM64, SETCC+MOVZX on x86_64) and emit a hardware compare-and-branch (diff).
