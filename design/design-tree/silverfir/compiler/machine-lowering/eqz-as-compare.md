- Wasm `i32.eqz`/`i64.eqz` is lowered as `IntCompare { Eq, rhs: 0 }` rather than
  a dedicated MachineIR op, leaving an eqz feeding a br_if reachable by the
  compare-and-branch fusion peephole that collapses the pair into one conditional
  branch (`lower_leaf_arith`).

## Moves

- 2026-04-06 (10cfbc1b) replaced [[dedicated-eqz-op]]: a standalone Eqz opcode
  could not be reached by the compare-and-branch fusion peephole, so
  i32.eqz/i64.eqz followed by br_if stayed two instructions; reframing eqz as
  IntCompare{Eq, rhs:0} (bit-for-bit identical) lets fuse_compare_branch
  collapse the eqz+br_if pattern into a single conditional branch (arm64
  cbz/cbnz) (diff)
