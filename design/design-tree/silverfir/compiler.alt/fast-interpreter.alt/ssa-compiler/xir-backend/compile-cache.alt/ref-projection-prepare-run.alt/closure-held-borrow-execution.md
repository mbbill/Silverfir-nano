- Function execution allocates the register file, marshals params into their
  register slots, builds the context, and runs the trampoline all inside the
  cache-access closure; the cached-lowered-code RefCell borrow is held for the
  entire call including the re-entrant recursive call chain.

- The module instance is borrowed from the function-instance cell rather than
  cloned, since the trampoline runs within the scope of that borrow.

## Moves

- 2025-10-12 (1216946b) replaced by [[ref-projection-prepare-run]]: the trampoline
  executes the whole re-entrant recursive call chain, so it must not run while the
  with_ssa_code RefCell borrow on a function's SSA-code cache cell is held: a
  recursive re-entry into the same function would re-borrow that same cell, so
  heavyweight setup that touches the cache is done inside the closure and copied
  out into a PreparedExecution whose run() fires the trampoline only after the
  borrow is dropped (diff).
