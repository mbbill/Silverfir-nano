- A ScratchGuard can be consumed by `release()`, which frees the pool slot and
  returns the raw register value for use after the lexical guard ends (e.g. to
  patch a previously-emitted instruction).

- `PreparedGp`/`PreparedFp` expose `reg()` to read the physical register and
  `release()` to surrender the guard while keeping the register value.

## Moves

- 2026-04-01 (db81af27) replaced by [[scratch-pool]]: release() freed the pool
  slot immediately while the caller kept using the register, so a later
  scoped_alloc could hand the same physical register out again; detach() returns
  an owned DetachedScratch token that keeps the slot reserved with RAII until
  dropped, so a temp that must survive later &mut self emission calls stays
  protected (code).
