- The runtime page cap is enforced against a memory's initial page count at
  instantiation, and growth is re-checked against the requested new size only
  when `memory.grow` actually asks for growth — the declared maximum is treated
  as a type-level growth ceiling, not an instantiation-time gate
  (`check_memory_quota`).

## Moves

- 2026-04-22 (219b5e56) replaced [[declared-max-at-instantiation]]: checking a
  module's declared max against the runtime page cap rejected at instantiation
  modules that declare a large type-level ceiling but never grow into it; the
  declared maximum is a type-level growth ceiling, so the runtime cap is applied
  to the initial page count at instantiation and re-applied to the requested new
  size only when `memory.grow` actually asks for growth (diff).
