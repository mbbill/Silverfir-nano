- Each function is lowered to XIR at most once and the result is cached on the
  function; compilation is one entry point that checks the cache, compiles on a
  miss, stores, and returns the cached function (`compile`).

- The cached lowered function is handed out as a refcount clone, not a borrow of
  the cache slot: it holds self-referential raw pointers (cannot be
  cloned by value) and a held borrow of the cache slot cannot survive the
  recursive on-demand compilation of a callee that re-enters the same cache.

- The parameter-to-slot and result-to-slot maps are computed once during lowering
  and stored on the cached function; a call recovers them without rebuilding the
  function.

- A function compiles lazily on its first call by default; when module-level eager
  precompilation runs, every same-module function is compiled up front in one pass
  (a no-op once cached).

## Facts

- 2025-10-12 (cf2feafe) rationale: the parameter-to-slot and result-to-slot maps
  are computed once during lowering and stored on the cached function; before this,
  every call rebuilt the whole SSA function just to recover params/results
  metadata, so caching the slot maps removes a per-call rebuild and makes the
  'compiled at most once' property actually hold for the call path (diff).

## Moves

- 2025-10-23 (2986784e) replaced [[ref-projection-prepare-run]]: the cached lowered
  function is handed out as an Rc clone rather than a projected RefCell borrow,
  because the lowered function holds self-referential raw pointers and cannot be
  cloned, and a held Ref borrow of the cache slot cannot survive the recursive
  on-demand compilation of a callee that re-enters the same cache, so a cheap
  ref-count clone is the only shape that both shares without copying and releases
  the borrow before recursing (diff).
