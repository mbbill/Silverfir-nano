---
status: abandoned
---
# Linear LIR, no SSA / no register allocation

The micro-JIT's original internal shape: a single linear, interpreter-era IR
(`TempInst`, later a neutral `IrOp`) that resolves all stack-machine management
once — TOS variant selection, spill/fill, hot-local mapping — and then feeds
three emitters off the same resolved stream: 1:1 interpreter handlers, static
fusion, and the micro-assembler. There is deliberately no SSA and no register
allocation, because the TOS + L0/L1/L2 mapping already fixes every physical
register before the IR is emitted.

The IR carries no control-flow graph and no notion of value liveness across
blocks; placement is decided per instruction at decode time and never revised.

## In practice

Must:
- Resolve TOS variants, spill/fill, and hot-local mapping in one linear pass and
  share that resolved stream across all three emitters.
- Keep physical register assignment fixed by the TOS + L0/L1/L2 mapping; the LIR
  introduces no allocation step.

Must not:
- Build a control-flow graph or any SSA form; the IR is linear with no
  cross-block value model.
- Attempt to carry register residency across loop boundaries or to remove
  loop-boundary dispatch or memory-metadata re-setup; the linear shape cannot
  express either.
