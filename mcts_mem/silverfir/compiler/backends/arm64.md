- The ARM64 backend fuses a base-plus-scaled-index address feeding an adjacent
  load or store into a single indexed-addressing instruction, and selects
  immediate-operand forms of add/sub/mul/logical/compare when the constant fits
  the encoding rather than materializing it into a register first
  (`lower_indexed_load`).

- A block edge whose argument registers already equal the target block's
  parameter registers is an identity edge that branches straight to the target;
  an edge stub (a synthetic block of copies plus a jump) is emitted only when
  the edge actually moves values (`emit_edge`, `is_identity_edge`).

## Facts

- 2026-04-08 (21d6f6bf) rationale: the arm64 FP scratch pool was cut from 3
  slots to 2 by restructuring FP-binary lowering so peak FP scratch never
  exceeds 2 — fmin/fmax fold into one shared NaN-patch helper (fcmp-first with a
  branch-skipped cold FADD so the result may safely alias an operand), copysign
  is split into its own helper so the binary path no longer eagerly prepares an
  unused rhs_fp, and the result reuses the consumed lhs's slot when dst is not
  FP-mapped — returning fp(2) to the dynamic FP bank (code).

- 2026-03-17 (b4ecd7ee) pitfall: the fixed-shape convert lowering
  unconditionally materialized the source into a GP scratch register before
  dispatching, forcing FP-source conversions (float->float demote/promote, FP
  reinterprets) through an avoidable GP round-trip; the source materialization is
  now pushed into only the arms whose source is genuinely GP, so float-to-float
  conversions stay in FP registers end-to-end (the same representation-driven
  GP/FP churn the typed-residency work targeted) (code).

- 2026-03-17 (1c3292fb) pitfall: an i32.reinterpret_f32 / i64.reinterpret_f64
  whose source value is resident in an FP register must move it GP<-FP directly
  with fmov (x<-d), not materialize it through a GP scratch as if it were a GP
  value; the arm64 reinterpret arm now checks the source bank and uses fmov for
  FP-resident sources, avoiding the wrong/extra GP round-trip the bank-blind path
  introduced (code).

- 2026-03-22 (5eca447e) rationale: trapping truncations stay an out-of-line
  helper call (arm64_trapping_trunc) because they must detect out-of-range/NaN
  and raise a Wasm trap, while saturating truncations are emitted inline as
  native fcvtzs/fcvtzu — the two paths diverge precisely on whether a trap is
  possible, and ARM64's hardware fcvtzs/fcvtzu saturation matches the Wasm
  saturating result so no software clamping is needed (code).

- 2026-03-22 (5eca447e) pitfall: a GpWord select must always emit the 64-bit
  csel_64, never csel_32: GpWord covers both i32 and reference values, and a
  reference (e.g. the null sentinel) needs its full 64 bits preserved, so a
  32-bit conditional select would truncate a live ref operand (code).

- 2026-04-06 (10cfbc1b) pitfall: on arm64 the compare-against-zero branch is
  emitted as the 64-bit cbz/cbnz form even for i32 operands; this is correct
  because AArch64 32-bit ops zero the upper 32 bits of the X register, so a
  64-bit zero test on an i32 result is equivalent to a 32-bit one, and both
  lhs==0 and rhs==0 operand orders are matched because the wasm operand may be
  hoisted to either side (code).
