- Before every compiled local call, all cached locals are saved to their
  canonical frame slots and the cache state is cleared; no cached local survives a
  call in a register (`emit_save_dirty_cached_locals`).

- Registers are only execution caches over frame state: if a backend ever relies
  on a cached local or transient being the only copy of a value at a boundary, the
  design has been violated.

## Moves

- 2026-05-14 (9cdd924a) replaced by [[local-call-boundary]]: requiring every
  cached local to be frame-published before a call forces a save before and a
  reload after each compiled local call even when the value is unchanged and
  lives in a callee-saved register; a cache-layout analysis can instead select
  clean non-ref caches needed after the call that already sit in preserved
  dynamic lanes and carry them as explicit Call.success edge arguments, so the
  value survives the call in its register with no save/reload, while ref-typed
  and dirty caches still publish for root visibility and frame authority (diff)
