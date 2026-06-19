- The externally-facing `Value` carries its type and is used only at API
  boundaries (arguments, results, host calls), converted to and from the raw
  word at the edge.

- Its single reference variant `Value::Ref` carries the reference handle
  alongside the full `RefType` (heap type plus nullability), rather than one
  Value variant per reference kind.

## Moves

- 2025-10-04 (b76cdd46) replaced [[per-kind-ref-variants]]: separate
  FuncRef/ExternRef value variants could not carry the full reference type
  (heap type + nullability) needed once references span the whole 3.0 heap-type
  hierarchy; one Ref(handle, RefType) variant carries the type alongside the
  handle (code).
