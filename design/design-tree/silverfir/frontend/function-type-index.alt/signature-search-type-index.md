- A function's type index is not stored; it is recovered when needed by
  scanning the module's type section for the first composite type whose function
  signature (params and results) equals the function's type.

- ref.func in a constant expression and the instantiation-time
  function-index-to-type-index map both perform this signature search, panicking
  or erroring if no structurally-matching type is found.

## Moves

- 2025-10-07 (f7febf40) replaced by [[function-type-index]]: recovering a
  function's type index by linear-searching the type section for a
  structurally-matching signature returns the first match, which is the wrong
  index once the GC type system allows several distinct type indices to share
  one signature (different recursion groups), so each function now records the
  type index it was declared with at parse time (diff).
