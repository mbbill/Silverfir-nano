- SMULL sign-extension fusion is a single-block forward sweep: it tracks
  per-register alias and sign-of relationships within one block and rewrites an
  i64 pair multiply of two sign-extended-from-i32 operands into
  Int64MulFromSignExt32 only when both operands' sign-extension is visible
  inside that same block.

## Facts

- 2026-04-23 (2e7d330e) rationale: on a 32-bit GP backend a generic
  `Int64PairBinary{Mul}` lowers to UMULL + two MLA correction terms + two movs,
  but when both i64 operands are i32 sign-extended (their hi halves are
  `lo >> 31`) the whole 64-bit product collapses to a single signed SMULL; the
  pass runs after copy_propagate (so spill/reload and Move aliasing are already
  settled) and tracks per-register `sign_ext_of` and `value_alias` plus a
  recent-spill table in one forward sweep, rewriting at each mul against the
  state live at that point — a 'build final-state map then scan muls' approach
  would report the wrong state for earlier muls when intervening writes
  invalidate the relationships (code).

## Moves

- 2026-04-27 (1ea01e67) replaced by [[smull-sign-ext-fusion]]: the single-block
  forward sweep could not see a sign-extension produced in a different block, so
  the Mandelbrot hot loop (operands sign-extended in the loop header, multiplied
  in the body) never collapsed to a single signed widening multiply;
  whole-program value-id dataflow propagates sign-extended pairs across CFG
  edges with multi-predecessor and self-loop safety so the fusion fires across
  block boundaries (code).
