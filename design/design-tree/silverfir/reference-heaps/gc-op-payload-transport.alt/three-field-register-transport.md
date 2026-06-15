- struct.new passes its fields to the preserved helper in three fixed I/O
  register slots (ARG0..ARG2); the MachineIR StructNew op stores fields in a
  fixed [(MachineValue, Option<MachineValue>); 3] array with a field_count, and
  a struct with more than three fields is rejected as not yet supported.

- The arm64 struct.new lowering and the array.new_fixed lowering are separate
  code paths even though array.new_fixed already spills its elements to a
  payload area.

## Facts

- 2026-04-16 (ccceb9a8) statement: when GC struct/array ops were first lowered
  through preserved helpers, struct.new carried its fields in a fixed [u64;3]
  in-register transport (ARG0..ARG2) and the decoder, machine IR
  (StructNew.fields: [(MachineValue, Option<MachineValue>); 3]), and helper all
  rejected any struct with more than three fields as "not yet supported";
  array.new_fixed already used a spilled payload area, so the two transports
  were inconsistent until struct.new was unified onto the payload transport two
  commits later (diff).

## Moves

- 2026-04-17 (9bcf20b4) replaced by [[gc-op-payload-transport]]: the [u64;3]
  register transport could hold at most three fields, so wide structs were
  rejected as unsupported; routing struct.new (like array.new_fixed) through a
  spilled payload area passed by pointer plus a field count lets arbitrary
  field counts work uniformly across the native backends (diff).
