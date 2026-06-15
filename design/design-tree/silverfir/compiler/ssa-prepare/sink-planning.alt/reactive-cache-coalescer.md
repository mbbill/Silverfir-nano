- A machine-lowering peephole reactively coalesces a transient-to-cache-local
  move by patching the immediately-preceding instruction's destination register
  in place, rewriting `op r_transient; move r_cache <- r_transient` to
  `op r_cache` directly, but only when the source is a same-bank, same-storage
  transient defined by that preceding instruction, with zero remaining uses, and
  that instruction does not also read the cache register; otherwise it emits the
  explicit move.

## Moves

- 2026-03-26 (98de6d7b) replaced by [[sink-planning]]: the reactive coalescer
  could only patch the single immediately-preceding instruction and could not
  reason about semantic local versions or cross-instruction legality, so
  sink-legality analysis was lifted into the middle-end where the version is
  known and a producer can be proactively pre-mapped into the local's cache home
  (diff).
