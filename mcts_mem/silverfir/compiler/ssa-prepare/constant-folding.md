- Single-use constant absorption happens at the SSA level: the fold pass
  evaluates pure all-constant ops, then absorbs a single-use constant definition
  into its consumer as an inline-constant operand (`SsaOperand::Const`) that the
  backend lowers straight to a native immediate, with no later peephole needed to
  re-detect a move-imm-then-use shape (`fold_constants_into_operands`).

## Facts

- 2026-03-22 (f9e742ed) rationale: a deleted plan (docs/WASM_CONSTANT_FOLDING_PLAN.md,
  commit f9e742ed, removed cb040a2c) proposed real numeric folding as a wasm/
  semantic-IR pass after decode+inline that would add one new optimized semantic
  binop form carrying an embedded constant (op kind, constant payload,
  constant-side flag, pop1/push1 stack effect) plus centralized stack-effect
  computation over full semantic op kinds, capped so optimized IR never holds two
  embedded constants; that semantic-op shape was not adopted — folding instead
  landed in middle/ at the SSA level (fold_constants_into_operands), keeping the
  canonical Wasm op vocabulary unextended and expressing an absorbed constant as
  an inline SsaOperand::Const operand the backend lowers to a native immediate,
  because SSA's linear single-use structure makes the absorption a cheap operand
  rewrite and post-cleanup block-locality already exposes the profitable chains,
  so no new immediate-bearing semantic opcode or earlier wasm/ pass was needed
  (sourced).

- 2026-03-23 (ba35b941) rationale: float rounding ops (ceil/floor/trunc/nearest)
  are constant-folded with hand-written software rounding (bit-exponent
  manipulation, round-half-to-even) rather than libm, because the engine is
  no_std with no libm; trapping truncations fold only when the truncated value is
  in range (else None preserves the trap) while saturating truncations always
  fold to a defined clamped result (code).

- 2026-04-05 (2cf59fff) rationale: folding keeps immediates explicit in SSA
  operands so the backend emits native immediate forms directly instead of
  materializing each constant into a transient register; it is kept block-local
  because cleanup has already merged trivial CFG structure so the profitable
  constant chains are visible within one prepared block (code).

## Moves

- 2026-03-23 (7d2ca7fb) replaced [[machineir-fold-constants-peephole]]: absorbing
  a single-use constant into its consumer is more naturally an SSA-IR rewrite
  than a post-lowering pattern match: linear SSA structurally guarantees the
  single use, so the SSA pass folds the const into the operand slot and the
  backend lowers SsaOperand::Const straight to MachineValue::Imm64, removing the
  need for the machine peephole to re-detect the move-imm-then-use shape (code).
