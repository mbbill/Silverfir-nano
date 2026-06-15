- The externally-facing Value enum has one reference variant per reference kind
  (FuncRef(Ref), ExternRef(Ref)); a value's type is recovered from which
  variant it is.

- default_for_type maps funcref/externref/typeref value types onto these two
  reference variants, folding TypeRef onto the externref variant; from_raw maps
  only the bare funcref/externref value types onto them and yields Unknown for
  any other value type (including typeref).

## Moves

- 2025-10-04 (b76cdd46) replaced by [[external-value]]: separate
  FuncRef/ExternRef value variants could not carry the full reference type
  (heap type + nullability) needed once references span the whole 3.0 heap-type
  hierarchy; one Ref(handle, RefType) variant carries the type alongside the
  handle (diff).
