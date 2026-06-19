- The backend lowers and interprets the GC reference and struct opcodes:
  ref.cast/ref.test (and their null variants), ref.eq, ref.is_null,
  ref.as_non_null, call_ref, and struct.get/get_s/get_u/set, with packed struct
  fields sign/zero-extended per the field's storage type from the module type
  definition.

- A concrete GC cast/test matches the object type by a bespoke depth-limited
  structural type-equivalence walk that compares composite shape (struct fields,
  array element, supertypes) directly along the supertype chain, with a
  recursion-depth cap to avoid infinite descent on cyclic types.

- ref.cast traps on a null non-nullable reference or on type mismatch and
  otherwise passes the reference through; the nullable variant lets null pass and
  traps only on mismatch. ref.as_non_null traps on null and passes through
  otherwise.

- A reference operation's target heap type, too large for an inline immediate, is
  passed to its handler as a pointer to a type held in the instruction's side-table
  metadata.

## Facts

- 2025-10-14 (5fb7ed2d) statement: the reference-vs-heap-type match logic is
  reimplemented in the SSA backend, duplicating the equivalent logic already
  present in the classic in-place interpreter, marked as a candidate for extraction
  into a shared module — two consumers now compute GC subtype matches independently
  (code).

- 2025-10-14 (6f3e5855) rationale: call_ref's signature check tries exact
  structural equality of the callee's function type against the expected type first
  as a fast path, and only on mismatch falls back to a TypeContext canonicalization,
  so identical signatures skip building a TypeContext at all (code).

- 2025-10-25 (10d34744) statement: the GC reference type-matching rules
  (concrete subtype-chain walk, structural type equivalence, and the abstract
  heap-type rules for i31/struct/array/eq/any/extern/func) live in one shared
  module (`check_ref_type_match`) with two real consumers — the classic in-place
  interpreter [[classic-inplace]] and the XIR backend's ref.cast/ref.test
  handlers [[xir-backend]] — hoisted out of the in-place interpreter so both
  execution paths share one implementation instead of duplicating the rules
  (code).

## Moves

- 2025-10-14 (f2489323) replaced [[cast-by-typecontext-canon]]: the cast match
  abandons TypeContext canonicalization for a hand-rolled structural-equivalence
  recursion that compares composite shape (struct fields, array element,
  supertypes) directly, guarded by a recursion-depth cap to avoid infinite descent
  on cyclic types (code).
