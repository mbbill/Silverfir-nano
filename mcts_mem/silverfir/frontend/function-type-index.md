- Each function records the type index it was declared with at parse time
  (`type_index` on the function entity and on a function import), rather than
  recovering it later from its signature.

## Moves

- 2025-10-07 (f7febf40) replaced [[signature-search-type-index]]: recovering a
  function's type index by linear-searching the type section for a
  structurally-matching signature returns the first match, which is the wrong
  index once the GC type system allows several distinct type indices to share
  one signature (different recursion groups), so each function now records the
  type index it was declared with at parse time (code).
