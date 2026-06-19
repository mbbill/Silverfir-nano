- The validator pops a value against an expectation expressed as an
  `Option<ValueType>`: `None` accepts any value, `Some(t)` accepts the popped
  type only when it is a subtype of `t`, checked directionally (`actual`
  must be a subtype of `expected`, never the reverse) (`pop_val`).

- `Unknown` is the bottom type for this check: a value popped in unreachable
  code is `Unknown`, which is a subtype of every expectation and satisfies any
  `Some(t)`.

## Moves

- 2025-10-05 (38eb8a20) replaced [[stack-pop-expectation.alt/popvalexpect-matcher]]:
  the matcher trait checked expectations with a symmetric matches relation and
  treated Unknown as a wildcard matching in both directions, which cannot
  express Wasm validation's directional rule (actual must be a subtype of
  expected) nor a true bottom type; Option<ValueType> with is_subtype_of and
  Unknown as bot encodes both (code).
