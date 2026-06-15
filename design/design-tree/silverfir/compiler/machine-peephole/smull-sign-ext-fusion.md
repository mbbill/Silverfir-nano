- A MachineIR peephole fuses an `i64.mul` whose two operands both originate
  from `i64.extend_i32_s` into a single signed 32x32->64 multiply op
  (`Int64MulFromSignExt32`), emitted only on 32-bit GP backends (where the
  i64-pair source pattern exists) and lowered to one SMULL on ARM32.

- The fusion propagates sign-extended pairs across CFG edges by whole-program
  value-id dataflow with multi-predecessor and self-loop safety; it fires
  even when one operand's sign-extension happens in a different block than the
  multiply (`fuse_smull_sign_ext_across_edges`).

## Facts

- 2026-04-27 (1ea01e67) statement: the signed-widening-multiply recovery runs
  at MachineIR rather than in a backend because the fact being recovered (both
  operands are sign-extended-from-i32, so the 64-bit product is a signed
  32x32->64 multiply) is target-independent; each backend separately decides
  whether it has a native widening multiply or lowers the generic pair form
  (diff).

## Moves

- 2026-04-27 (1ea01e67) replaced [[block-local-smull-fusion]]: the single-block
  forward sweep could not see a sign-extension produced in a different block, so
  the Mandelbrot hot loop (operands sign-extended in the loop header, multiplied
  in the body) never collapsed to a single signed widening multiply;
  whole-program value-id dataflow propagates sign-extended pairs across CFG
  edges with multi-predecessor and self-loop safety so the fusion fires across
  block boundaries (diff).
