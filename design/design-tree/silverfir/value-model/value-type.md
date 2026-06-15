- A value type is a structured descriptor whose reference case is
  `ValueType::Ref(RefType)`, where `RefType` is a nullability flag plus a
  `HeapType` that is either an abstract heap type (any/eq/i31/struct/array/exn
  and their bottoms) or a concrete module type index.

- Nullability is an orthogonal boolean flag on `RefType`, not a separate
  enumerated variant; the abstract heap types and arbitrary concrete type
  indices need no combinatorial enum of reference shapes.

- A non-nullable reference type is parsed and validated as distinct from its
  nullable form but carries no separate runtime representation: it is
  materialized as the same null-capable reference word, and non-nullability is
  enforced only as a validation subtype relation.

- A `RefType` target for ref.cast / ref.test encodes into a single u64 — its
  nullability, abstract-vs-concrete bit, and either the abstract-type
  discriminant or the concrete type index — and the target rides in the
  instruction's immediate rather than through a heap-allocated type array.

## Facts

- 2025-11-12 (64d36bb5) rationale: a RefType's nullability, its
  abstract-vs-concrete bit, and either the abstract-type discriminant or the
  concrete type index all fit in a single u64, so the ref.cast/ref.test target
  rides directly in the instruction's immediate (encode_to_u64 at codegen,
  decode_from_u64 in the handler) instead of being stored in a separately
  heap-allocated type array and reached through a per-instruction pointer
  indirection (diff).

## Moves

- 2025-10-04 (b76cdd46) replaced [[flat-enum-value-type]]: the flat enum
  encoded each reference shape as its own variant (funcref, externref, their
  non-null twins, a single TypeRef catch-all) and could not express nullability
  as a flag, the abstract heap-type hierarchy (any/eq/i31/struct/array/exn and
  their bottoms), or concrete type indices (diff).
