- The middle layer is passes, not a parallel IR: φ elimination
  (critical-edge splitting + ParCopy insertion) transforms SSA IR in place;
  register allocation is the only real IR boundary — virtual registers in,
  physical registers out.
- LIR is the post-allocation form: operands are physical register numbers,
  spills are explicit slot load/stores, no SSA, no φ.
- The CFG computed on SSA IR is preserved through to LIR.
- Targets are parameterized by a descriptor of instruction signatures and
  register counts. All backends consume LIR: the interpreter backend (see
  `xir-backend`) is live; a JIT backend (LIR → native code, with deopt and
  safepoints planned) holds the sibling slot.
- Block-level peephole optimizations run on LIR after allocation.
