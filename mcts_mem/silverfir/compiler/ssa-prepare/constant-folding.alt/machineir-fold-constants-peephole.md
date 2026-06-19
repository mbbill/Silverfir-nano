- An SSA leaf op's arguments are a flat vector of transient value references; an
  inline immediate cannot appear in an operand slot.

- A MachineIR peephole (`fold_constants`) scans each block for
  `move rX <- imm; op ... rX ...` (and the FloatConst form), and when rX is a
  single-use transient whose last use is the next op, it rewrites the use to
  MachineValue::Imm64(imm) and deletes the move.

- The peephole only folds transient registers; cached-local and fixed registers,
  which persist across the block, are never folded.

## Facts

- 2026-03-17 (5afdd2a1) rationale: constant folding into operands is extended
  to FP transient constants (FloatConst rX <- bits folds into a following float
  compare/op as Imm64(bits)), which lets both backends special-case
  compare-against-zero without materializing a zero into an FP register: arm64
  emits FCMP Dn,#0.0 and x86_64 zeroes a scratch XMM with xorpd before ucomis,
  removing the zero-constant load on the very common float==0 test (code).

- 2026-03-22 (1e66fdb2) pitfall: folding a single-use transient constant into a
  later use by scanning forward across the whole block is unsafe — a constant may
  be folded past intervening instructions whose effects make the substitution
  wrong; the fold is restricted to the immediately-following instruction only,
  dropping the def and replacing its immediate into op[i+1] only when that single
  adjacent op is the last use before any redefinition (code).

- 2026-03-22 (87e309c3) pitfall: even when the adjacent instruction reads the
  constant exactly once, the fold must also check that that use is replaceable
  (`count_replaceable_value_uses == 1`): some operand positions cannot legally
  take an embedded immediate, so a single textual use is not sufficient to
  authorize substituting the constant in place (code).

- 2026-03-22 (f9e742ed) statement: the author plan frames the adjacent-only
  MachineIR fold as a deliberate temporary safety baseline, not the final
  architecture — MachineIR peephole folding stays conservative (only fold a
  transient constant into the immediately following instruction, never long-range
  whole-block folding) while the intended home for real numeric constant folding
  is an earlier semantic-IR pass after decode and inlining, so folding can cut
  transient pressure before spill/fill planning (sourced).

## Moves

- 2026-03-23 (7d2ca7fb) replaced by [[constant-folding]]: absorbing a single-use
  constant into its consumer is more naturally an SSA-IR rewrite than a
  post-lowering pattern match: linear SSA structurally guarantees the single use,
  so the SSA pass folds the const into the operand slot and the backend lowers
  SsaOperand::Const straight to MachineValue::Imm64, removing the need for the
  machine peephole to re-detect the move-imm-then-use shape (code).
