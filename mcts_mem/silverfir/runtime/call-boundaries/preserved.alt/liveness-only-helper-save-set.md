- Every ARM64 C-ABI helper call, including GC, reference, table, and other
  semantic runtime operations, saves only dynamic registers reported live at
  that MachineIR instruction.

## Facts

- 2026-07-22 measurement: this design failed four of eight native array tests
  and caused 22 native spectest failures while emulator and x64 paths stayed
  green, demonstrating that the normal MIR live-after set is not a complete
  description of the semantic preserved-helper ABI (sourced).

## Moves

- 2026-07-22 replaced by [[preserved]]: ordinary MIR live-after data omits
  dynamic lanes required by semantic runtime
  helpers; full preservation fixes native GC/reference/table behavior while
  liveness-only remains scoped to raw infallible libc memory calls (code).
