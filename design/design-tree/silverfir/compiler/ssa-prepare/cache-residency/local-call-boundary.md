- At a compiled local call the lowerer publishes or drops dirty and non-selected
  cached locals, but carries selected clean non-ref cached locals that already sit
  in preserved dynamic lanes across the `Call.success` edge as cached-local block
  params; those values survive the call in their registers with no save/reload
  while ref-typed and dirty caches still publish for root visibility and frame
  authority (`prepare_cached_locals_for_local_call`).

- A cached local earns a preserved-lane preference only after it crosses a
  backend-configured number of direct local calls (default 7) — a preserved
  dynamic lane carries a callee-saved register's prologue/epilogue save/restore
  cost; one isolated call does not pay for it, and inherited block-entry layouts
  are not forced to switch banks merely to satisfy the preference.

## Facts

- 2026-03-30 (3babe5ff) rationale: every call/helper boundary previously published
  ALL cached locals to their frame slots, re-storing registers unchanged since the
  last boundary; a cross-block forward dataflow (join=OR, fixpoint, entry
  all-clean, non-entry conservatively all-dirty) now computes which cached locals
  are dirty at each block entry, and the boundary publishes only the dirty ones
  (diff).

- 2026-05-14 (b4c75b62) rationale: identical cached-local boundary repair blocks
  are interned by (target, predecessor exit cached slots, repair actions), so the
  multiple identical edges of a `br_table` that need the same cache reconciliation
  are all retargeted to one shared repair block instead of emitting a duplicate
  store/jump block per edge, which otherwise multiplied repair-block code by the
  `br_table` fan-out (diff).

- 2026-03-17 (1f6c4731) rationale: the continuation reload after a call is elided
  per cached-local by a conservative forward straight-line scan from each call
  site — walking until the first branch, loop, other call, or end-of-function, any
  cached local seen written before it is read will be overwritten before use, so
  its reload is skipped; the per-call-site skip masks are computed in prepare
  alongside the cache prefs and carried on each boundary op rather than recomputed
  at lowering (diff).

## Moves

- 2026-05-14 (9cdd924a) replaced [[publish-all-boundary]]: requiring every
  cached local to be frame-published before a call forces a save before and a
  reload after each compiled local call even when the value is unchanged and
  lives in a callee-saved register; a cache-layout analysis can instead select
  clean non-ref caches needed after the call that already sit in preserved
  dynamic lanes and carry them as explicit Call.success edge arguments, so the
  value survives the call in its register with no save/reload, while ref-typed
  and dirty caches still publish for root visibility and frame authority (diff)
