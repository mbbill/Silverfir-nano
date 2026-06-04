---
commit: ae55756, 9b5e0e1
---
After i64 legalization settled on the SSA side, a finer twist swung one slice of
i64-pair lowering back toward the arch backends — not a return to the abandoned
MachineIR legalization, but a re-balancing driven by hot-loop codegen quality on
RV32 and ARM32. The RV32 backend first added a high-half liveness analysis for
i64-pair results: when the high half of a pair result is provably dead, the
backend lowers only the low 32 bits (low-only lowering for selected pair ops,
small const shifts, and extend32s) and lowers `Int64MulFromSignExt32` directly to
a register multiply.

That high-half liveness analysis was then lifted out of the RV32 backend into
MachineIR and routed through `CompilerCore`, so RV32 and ARM32 share the same
"low32 dead-hi" facts and ARM32 gained the corresponding low-only cases for
add/sub/and, extend32s, and small const shifts. CompilerCore register sizing was
also derived from `BackendConfig` instead of threading backend max register
counts through each constructor. The analysis is computed once in MachineIR; the
SSA layer still owns i64 pressure accounting. This is a shared-fact-at-MachineIR
optimization, motivated by Mandelbrot-class hot-loop codegen quality on 32-bit
targets.
