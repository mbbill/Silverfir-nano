- Liveness is backward dataflow to fixpoint over the SSA CFG.
- The allocator is instruction-aware: per-instruction signatures from the
  target descriptor constrain which operands need registers.
- Allocation is split into an analysis phase and an allocation phase, with
  an independent checker validating the result: structural invariants,
  canonical value tracking, and spill-slot interference.
- Register choice prefers the least costly register; call-site argument
  positions hint allocation, weighted against spill cost.
- Spill-victim strategies are pluggable (Belady furthest-next-use via
  next-use chains); spill slots are colored by direct live-range overlap —
  no interference graph is built.
- Constants rematerialize instead of spilling when cheaper.
- Commutative operations swap operands when it avoids a spill.
- Register state is reconciled explicitly across block edges; critical
  edges are split and the reconciliation validated.
