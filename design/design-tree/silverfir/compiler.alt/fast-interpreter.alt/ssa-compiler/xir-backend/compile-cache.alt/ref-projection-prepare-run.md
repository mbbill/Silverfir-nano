- The lowered code is stored as the concrete LoweredFunction type and reached
  through a RefCell Ref projection, not a type-erased wrapper.

- Function execution is split into a prepare phase and a run phase
  (`PreparedExecution`): the heavyweight setup that touches the cache is done while
  holding the cache borrow, then copied out and the trampoline runs only after the
  borrow is dropped, since the trampoline executes the whole re-entrant recursive
  call chain and a recursive re-entry into the same function would re-borrow that
  same cache cell.

## Moves

- 2025-10-13 (710b985e) replaced [[box-any-closure-cache]]: the Any wrapper, used
  only to dodge a module-structure import cycle, forced a runtime downcast and a
  closure-passing access API on every call; importing LoweredFunction directly
  removes the type-erasure and closure overhead (diff).

- 2025-10-12 (1216946b) replaced [[closure-held-borrow-execution]]: the trampoline
  executes the whole re-entrant recursive call chain, so it must not run while the
  with_ssa_code RefCell borrow on a function's SSA-code cache cell is held: a
  recursive re-entry into the same function would re-borrow that same cell, so
  heavyweight setup that touches the cache is done inside the closure and copied
  out into a PreparedExecution whose run() fires the trampoline only after the
  borrow is dropped (diff).

- 2025-10-23 (2986784e) replaced by [[compile-cache]]: the cached lowered function
  is handed out as an Rc clone rather than a projected RefCell borrow, because the
  lowered function holds self-referential raw pointers and cannot be cloned, and a
  held Ref borrow of the cache slot cannot survive the recursive on-demand
  compilation of a callee that re-enters the same cache, so a cheap ref-count clone
  is the only shape that both shares without copying and releases the borrow before
  recursing (diff).
