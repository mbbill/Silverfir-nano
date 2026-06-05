---
commit: 90b81fa1
---
middle_design.md retires the middle IR (VIR, by then renamed MIR) in one
sentence: "MIR was just 'SSA IR without phi'. Not enough differentiation to
justify separate IR. Real IR boundary is virtual regs (SSA IR) -> physical
regs (LIR)." The same reframing retires the window manager: v0/v1/v2 stop
being runtime-managed window slots and become ordinary physical registers
of the XIR target, filled by standard register allocation with explicit
spill instructions — the interpreter becomes just another target of a real
compiler back half.
