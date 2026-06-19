- SSA IR is lowered straight to the executable XIR instruction stream in a
  single pass (two sub-passes only to back-patch forward branch targets), with no
  intermediate register-level IR and no separate register-allocation stage.

- Register assignment is a naive 1:1 identity map: SSA value N occupies
  backing-regfile slot N, with the slot count equal to the value count and no
  liveness analysis, slot reuse, or coalescing.

- A fixed hot window of three slots (v0/v1/v2) sits on top of the flat regfile
  and is carried in CPU registers across the trampoline ABI; explicit load/store
  moves shuttle values between the regfile and the window around each op.

- Phi nodes are resolved inline at branch emission by emitting a move per edge
  into the phi's slot, with no separate phi-elimination pass and no critical-edge
  splitting.

## Facts

- 2025-10-11 (70085428) rationale: this generation lowers SSA straight to the XIR
  instruction stream in a single pass with a naive 1:1 identity register map
  (num_slots == value count, no liveness, no reuse, no coalescing); on top of the
  flat regfile sits a fixed hot window of three slots carried in CPU registers
  across the trampoline ABI (code).

- 2025-10-11 (70085428) statement: this generation is documented as the Phase-3
  MVP with live-range reuse and tighter packing expected in later phases
  (sourced).

## Moves

- 2025-10-18 (9ff77d1b) replaced by [[vir-two-stage-middle]]: the old allocator
  gave every SSA value its own slot (identity map, no reuse) and the lowering
  targeted an interpreter-only three-register window; coloring overlapping live
  ranges collapses 100-200 SSA values to 10-30 virtual registers and the
  virtual-register IR serves both the interpreter and a future JIT (code).

- 2025-10-20 (bb0d2820) replaced by [[vir-two-stage-middle]]: the first-generation
  lowering assigned registers by a naive identity map (SSA value N to slot N, slot
  count equal to value count, no liveness, no reuse) baked into a single
  SSA-to-bytecode pass; splitting register allocation into a middle stage that
  runs liveness analysis, builds an interference graph, and colors values onto a
  bounded physical register set lets the same values share registers and the
  code-emitting backend become a thin translation with no allocation logic of its
  own (code).
