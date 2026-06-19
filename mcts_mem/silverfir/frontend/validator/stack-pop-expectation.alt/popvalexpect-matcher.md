- The validator pops a value by passing a PopValExpect (implemented for
  ValueType, &ValueType, and predicate closures); the trait's matches(actual)
  decides acceptance, with a None-less API that always names an expectation.

- Acceptance is symmetric: Unknown matches any expected type and any actual type
  matches an Unknown expectation; the wildcard is not directional.

## Moves

- 2025-10-05 (38eb8a20) replaced by [[stack-pop-expectation]]: the matcher trait
  checked expectations with a symmetric matches relation and treated Unknown as
  a wildcard matching in both directions, which cannot express Wasm validation's
  directional rule (actual must be a subtype of expected) nor a true bottom
  type; Option<ValueType> with is_subtype_of and Unknown as bot encodes both
  (code).
