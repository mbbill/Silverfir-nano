# Legalize i64 in SSA / the middle layer (lane mapping)

i64-on-32-bit legalization happens in SSA-IR / the middle layer: an i64 pair
consumes 2 register units in the joint planner's lane mapping, so its register
pressure is accounted at planning time. The middle layer still keeps i64 work in
pair-aware MachineIR ops (`Int64PairBinary`, `Int64PairDivRem`,
`Int64PairShift`, `Int64PairCompare`) that backends lower with carry/borrow
sequences; 64-bit GP targets use the thin scalar path.

A narrow slice of i64-pair lowering is informed by a shared "low32 dead-hi"
liveness analysis computed once in MachineIR and routed through `CompilerCore`,
so RV32 and ARM32 can drop the high half of a pair result that is provably dead.
This stays within the SSA-side option: the planner still owns pressure
accounting; only the dead-half lowering detail is shared at MachineIR.

## In practice

Must:
- Perform i64 → low/high split in the middle layer (SSA-IR), before register
  residency is decided.
- Charge a resident i64 local as 2 register units (`units(L) = 2`) in
  `ALGORITHM4` capacity, transition cost, and the lane map on 32-bit GP targets.
- Keep i64 work in the pair-aware MachineIR ops (`Int64PairBinary`,
  `Int64PairDivRem`, `Int64PairShift`, `Int64PairCompare`); each 32-bit backend
  must lower these natively with carry/borrow sequences.
- On 32-bit GP targets, allocate an i64 value as two adjacent dynamic registers
  (no even-parity alignment requirement; see `lower_context.rs` /
  `lower_regalloc.rs`).
- Compute the low32 dead-hi liveness fact once in MachineIR and share it through
  `CompilerCore` rather than recomputing it per backend.

Must not:
- Reintroduce a late MachineIR `legalize.rs` that hides the pair split from the
  planner.
- Skip i64 const-folding handling on 32-bit where the SSA legalization requires
  it (folding an i64 constant on a 32-bit target was a known correctness hazard
  fixed on the SSA side).
- Run the pair-aware ops or the dead-hi analysis on 64-bit GP targets, which use
  the scalar path.
