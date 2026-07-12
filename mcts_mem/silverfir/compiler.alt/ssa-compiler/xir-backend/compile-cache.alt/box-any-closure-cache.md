- The lowered function is stored on the function spec type-erased as a Box<dyn Any
  + Send + Sync> and recovered by downcast.

- Callers reach the lowered code through a closure-taking accessor that downcasts
  and invokes the closure under the borrow.

## Facts

- 2025-10-11 (a44db9ef) rationale: the lowered code is cached type-erased rather
  than as the concrete lowered-function type because the function spec lives in the
  module layer which cannot name the interpreter-internal lowered type; the
  attachment helpers downcast back on use (code).

## Moves

- 2025-10-13 (710b985e) replaced by [[ref-projection-prepare-run]]: the Any
  wrapper, used only to dodge a module-structure import cycle, forced a runtime
  downcast and a closure-passing access API on every call; importing LoweredFunction
  directly removes the type-erasure and closure overhead (code).
