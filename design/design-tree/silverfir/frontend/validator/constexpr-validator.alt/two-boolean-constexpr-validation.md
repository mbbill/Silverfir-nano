- Constant-expression validation is parameterized by two booleans: is_passive,
  which restricts passive element/data segments to ref.null/ref.func/end, and
  only_imported_globals, which restricts table initializers to referencing
  imported globals; there is no way to carry the validating global's own index,
  leaving the earlier-than-self rule on a global's init expression
  unenforceable.

## Moves

- 2025-10-05 (439fd90d) replaced by [[constexpr-validator]]: the two-boolean
  parameter pair (is_passive, only_imported_globals) could not carry the
  validating global's own index needed for the earlier-than-self rule (a global
  init expression may reference only globals defined earlier than itself;
  forward/self reference is 'unknown global'), which the context struct adds as
  validating_global_index: Option<usize> (diff).
