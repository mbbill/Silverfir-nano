- An arm64 direct local call emits an `ldr_lit_64` placeholder that loads the
  callee's internal entry address from a per-function deferred literal pool,
  then a BLR through that scratch register; the literal is flushed after edge
  stubs by `lower_function_literal_pool` and patched with the resolved callee
  address at module link time.

- The direct-call patch record carries only a single literal_offset to
  overwrite with the callee address.

## Moves

- 2026-05-13 (d7d328e7) replaced by [[direct-call]]: loading the callee address
  from a deferred literal and doing an indirect BLR on every direct call pays
  the cost of an indirect branch even for the common in-range case; emitting a
  real bl (or b for tail calls) that the linker patches directly to the callee
  keeps the hot path a single predicted branch, falling back to a literal-loading
  veneer only when the target lies outside arm64's +/-128MiB branch range
  (code).
