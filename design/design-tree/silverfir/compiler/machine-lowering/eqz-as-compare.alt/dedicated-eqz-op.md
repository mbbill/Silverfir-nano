- MachineIR carries a dedicated Eqz integer-unary op; each backend (x86_64,
  arm32, arm64, emulator) lowers wasm i32.eqz/i64.eqz as that Eqz op producing a
  0/1 boolean register.

## Moves

- 2026-04-06 (10cfbc1b) replaced by [[eqz-as-compare]]: a standalone Eqz opcode
  could not be reached by the compare-and-branch fusion peephole, so
  i32.eqz/i64.eqz followed by br_if stayed two instructions; reframing eqz as
  IntCompare{Eq, rhs:0} (bit-for-bit identical) lets fuse_compare_branch
  collapse the eqz+br_if pattern into a single conditional branch (arm64
  cbz/cbnz) (diff)
