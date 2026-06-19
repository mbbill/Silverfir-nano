- `check_memory_quota` enforces `wasm_memory_max_pages` against the memory's
  declared maximum (or its minimum when no maximum is declared) at every
  `MemInst` construction; a module declaring a maximum above the cap is
  rejected at instantiation even if it never grows.

## Facts

- 2026-04-21 (b206d2aa) rationale: the configured `wasm_memory_max_pages`
  ceiling is enforced at every MemInst construction (plain and guarded) against
  the module's declared max, and exceeding it there is reported as Unlinkable
  (the module is valid, it just cannot be instantiated in this configuration),
  not a validation error or trap (code).

## Moves

- 2026-04-22 (219b5e56) replaced by [[page-quota]]: checking a module's declared
  max against the runtime page cap rejected at instantiation modules that declare
  a large type-level ceiling but never grow into it; the declared maximum is a
  type-level growth ceiling, so the runtime cap is applied to the initial page
  count at instantiation and re-applied to the requested new size only when
  `memory.grow` actually asks for growth (code).
