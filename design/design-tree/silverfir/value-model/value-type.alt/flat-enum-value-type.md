- A value type is a flat enum whose reference cases are enumerated variants:
  FuncRef, ExternRef, their non-nullable twins FuncRefNonNull/ExternRefNonNull,
  a single TypeRef catch-all for any concrete-type-indexed reference, plus
  Unknown for the validator.

- Every enum case has a fixed u8 discriminant equal to its binary value type
  byte; the non-spec cases (TypeRef, Unknown) borrow unused bytes 0x42/0x41,
  and the non-nullable cases borrow 0x71/0x72 to avoid the 0x63/0x64 ref
  prefixes.

- Reference-type compatibility is decided by enumerated pairwise rules over
  these variants (nullable<->non-nullable, TypeRef<->func/extern), with TypeRef
  treated as compatible with both the func and extern hierarchies.

## Facts

- 2025-10-01 (cf8ff870) rationale: WebAssembly 3.0 nullable/non-nullable
  reference types are parsed and validated as distinct ValueType variants (the
  0x63 ref-null-ht and 0x64 ref-ht binary constructors over heap-type bytes
  0x70 func / 0x6F extern) but carry no separate runtime representation: a
  non-nullable funcref/externref is materialized as the same null-capable Ref
  word as its nullable form, and ref.null on a non-nullable type still produces
  a null; non-nullability is enforced only as a validation subtype relation —
  the validator accepts a non-nullable value where a nullable one is expected
  (and ref.func yields a non-nullable funcref) but never distinguishes them at
  runtime (diff).

- 2025-10-02 (b9a6c947) pitfall: the internal FuncRefNonNull/ExternRefNonNull
  ValueType discriminants were first assigned 0x64/0x63 — the very bytes that
  are the binary-format prefixes for the multi-byte ref-ht / ref-null-ht
  constructors; because single-byte value types are decoded through
  ValueType::try_from(byte), a stray 0x64/0x63 byte would decode straight into
  the non-nullable variant instead of being treated as a structured-reference
  prefix, so the discriminants were moved to the spec-unused bytes 0x71/0x72 so
  internal value-type tags never collide with binary-format prefix bytes
  (diff).

## Moves

- 2025-10-04 (b76cdd46) replaced by [[value-type]]: the flat enum encoded each
  reference shape as its own variant (funcref, externref, their non-null twins,
  a single TypeRef catch-all) and could not express nullability as a flag, the
  abstract heap-type hierarchy (any/eq/i31/struct/array/exn and their bottoms),
  or concrete type indices (diff).
