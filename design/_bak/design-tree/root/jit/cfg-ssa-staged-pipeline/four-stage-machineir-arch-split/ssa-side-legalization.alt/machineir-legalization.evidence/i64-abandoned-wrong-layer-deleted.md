---
commit: cf1c59e
---
The first 32-bit (armv7a) bring-up legalized i64-on-32-bit as a large MachineIR
pass: a `legalize.rs` that grew to ~2589 then ~2877 lines, alongside an `emu32`
reference backend, distinguishing word-sized GP values from true 64-bit GP values
below MachineIR and exploding i64 ops into low/high register-pair sequences. By
the time it was deleted the MachineIR `legalize.rs` was 4582 lines plus a
1455-line test file.

It was abandoned because MachineIR was judged the wrong layer: i64 high/low-half
register pressure must be accounted at *planning* time, and a late split caused
high-pressure failures — seen concretely as emu32 high-pressure failures that had
to be fixed once legalization moved to SSA. The pivot commit "move to ssa-based
legalization" was followed by deleting `legalize.rs` outright rather than
maintaining it. This is the seed of the abandoned `oldlegal` / early-legalization
lineage. The deciding fact is a layering fact (pressure must be visible to the
planner), corroborated by the concrete high-pressure failures the late split
produced.
