---
status: abandoned
---
# Legalize i64 at the MachineIR level

A large MachineIR pass (`legalize.rs`) lowers i64 ops to 32-bit register pairs:
one Wasm value still occupies one 8-byte frame slot, but below MachineIR the
pass distinguishes word-sized GP values from true 64-bit GP values and explodes
i64 arithmetic, shifts, and compares into low/high half sequences. An `emu32`
reference backend executes the legalized MachineIR.

## In practice

While in force this entailed:

Must:
- Run i64-on-32-bit legalization as a MachineIR pass below the shared
  instruction set, emitting explicit low/high half operations.
- Carry a MachineIR-level distinction between 32-bit GP values and true 64-bit
  GP values.
- Keep one 8-byte frame slot per Wasm value regardless of the pair split.

Must not:
- Expose i64 pair structure to the SSA-layer register planner (the split is
  invisible above MachineIR, so pressure is not accounted at planning time).
