# Micro-JIT — runtime fusion via micro-assembly (the JIT *is* the fusion)

The fusion source is a micro-assembler that emits machine code at run time
rather than an offline-discovered table. For each JIT-able variant op it emits
one to two native instructions, reusing the interpreter's already-fixed TOS
window and L0/L1/L2 register mapping. Because that mapping already pins physical
register assignment, the emitter needs no SSA and no register allocation: it
plugs into the existing builder after variant selection and spill/fill
insertion, groups consecutive JIT-able ops into "JIT groups," emits code, and
uses the code address as the handler pointer. Ops that are not JIT-able keep
their C handlers. The fused pattern set is therefore open-ended — any run of
JIT-able ops becomes native code without an encoding budget, a finite table, or
a discovery step.

A JIT group is a leaf: no calls, no register save/restore. Each group of N Wasm
opcodes compacts to a single handler slot, which the existing finalizer treats
identically to any other handler.

## In practice

Must:
- Assemble fused native code at run time from the decoded stream; the fused
  pattern set is open-ended, with no fixed table and no offline discovery step.
- Emit only one to two native instructions per variant op, taking value homes
  directly from the already-fixed TOS window and L0/L1/L2 mapping.
- Keep each JIT group a leaf (no calls, no register save/restore) and expose it
  as a single handler slot through the existing builder/finalizer.
- Fall back to the per-op C handler for any op that is not JIT-able.
- Carry a build-time check that the fixed register ABI used by the emitter
  matches the JIT's register model.

Must not:
- Run SSA construction or register allocation inside the emitter; physical
  registers are already pinned by the TOS + L0/L1/L2 mapping.
- Manage TOS overflow in the micro-assembler; the builder's stack tracker
  inserts spill/fill before the emitter runs.
