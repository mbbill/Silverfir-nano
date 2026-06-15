- GC struct/array constructor opcodes (struct.new, array.new_fixed) are
  lowered through preserved runtime helpers that receive their field/element
  values in a spilled payload area passed by pointer plus a field count, with
  no fixed upper bound on the number of fields (`do_struct_new`).

- struct.new and array.new_fixed share the one payload transport rather than
  each carrying its own field-passing convention.

## Moves

- 2026-04-17 (9bcf20b4) replaced [[three-field-register-transport]]: the
  [u64;3] register transport could hold at most three fields, so wide structs
  were rejected as unsupported; routing struct.new (like array.new_fixed)
  through a spilled payload area passed by pointer plus a field count lets
  arbitrary field counts work uniformly across the native backends (diff).
