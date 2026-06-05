- A backend-agnostic three-address IR between SSA and the execution
  backends: instructions reference virtual registers by index; each consumer
  decides where values physically live.
- SSA lowers into it 1:1; liveness (backward dataflow over the CFG, refined
  to intra-block segments), register allocation (linear scan, Belady
  furthest-next-use spilling, live-range splitting), and ParCopy resolution
  run as passes on it.
- φ nodes become edge-attached ParCopy pseudo-instructions with parallel
  semantics, kept intact through allocation and resolved into sequential
  moves (with cycle detection) afterwards; landing pads only on critical
  edges.
- Per-vreg metadata travels with the IR — live ranges, usage counts, loop
  depth — and backend lowering heuristics read it.
- Fused forms (Madd, Shladd) are matched during lowering, not carried as
  block metadata.
- A verifier checks structural invariants after lowering; targets are
  parameterized by a descriptor.
- Two consumers by design: the window-based threaded interpreter (current)
  and a native-code JIT backend (planned).
