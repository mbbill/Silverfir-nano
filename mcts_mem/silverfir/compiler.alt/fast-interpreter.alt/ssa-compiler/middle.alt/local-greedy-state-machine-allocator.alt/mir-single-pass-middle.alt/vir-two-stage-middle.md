- The middle stage lowers SSA IR into VIR, a separate post-allocation IR whose
  values are u32 virtual registers (10-30 per function) with terminators inlined
  into the instruction stream, intended as a backend-agnostic IR consumed by both
  the interpreter and a future native (JIT) backend.

- Register allocation runs once at the SSA→VIR boundary by graph coloring: a
  CFG-based backward-dataflow liveness analysis feeds an interference graph that
  colors 100-200 SSA values down to 10-30 virtual registers, non-overlapping
  values sharing a register; a second allocation pass (the backend window
  manager) later maps those vregs onto the interpreter's three hot-window slots.

- VIR mirrors SSA semantics 1:1 with a distinct instruction variant per
  (operation, type) carrying an inline type tag, including separate
  extension/nullable variants, operating on virtual-register indices.

- Lowering applies the register allocation as a pure rename of SSA values to
  virtual registers while translating each SSA block, phi, instruction, and
  terminator one-to-one into VIR.

## Facts

- 2025-10-18 (50b75e8e) rationale: register allocation runs once on the SSA→VIR
  boundary so every backend shares the result, and virtual registers are abstract
  indices, not physical locations, so the interpreter (hot window + backing file)
  and a future JIT (CPU registers + stack) each interpret them under their own
  constraints (code).

- 2025-10-18 (50b75e8e) rationale: recomputing live-range analysis (expensive) in
  each backend was rejected, and allocating SSA straight to physical would couple
  the IR to one backend (sourced).

- 2025-10-18 (50b75e8e) rationale: fusion is carried as frontend-emitted hints (a
  pattern plus the instruction range it covers) over the VIR (code).

- 2025-10-18 (50b75e8e) rationale: fusion hints over the VIR were chosen rather
  than fused VInst opcodes (which would bloat and de-portabilize the ISA) or
  per-backend pattern matching (which duplicates the matcher in every backend)
  (sourced).

- 2025-11-01 (95cfadd8) rationale: the interference-graph representation is chosen
  by value count (threshold 150) — a dense adjacency matrix is cache-friendly and
  O(1)-lookup but O(n²) space, so small functions use it while larger functions
  switch to sparse adjacency lists to bound memory (code).

- 2025-11-01 (fa5c0686) measurement: graph coloring reuses vregs that
  non-overlapping live ranges allow but the trivial 1:1 mapping cannot, achieving
  10-30% fewer virtual registers (~70-90 vs 100-200); the trivial mapping is
  retained as the --no-regalloc debugging fallback (code).

- 2025-11-01 (2646713a) statement: the CFG-based dataflow liveness adopts the
  external fixedbitset crate (0.5) as a new sf-core production dependency for
  compact per-block live sets; no in-tree bitset existed, so this is a new
  dependency commitment, not a reimplementation (code).

- 2025-10-26 (760b71b2) rationale: liveness and the interference graph are stored
  in dense arrays indexed by the SSA value's integer id (live ranges as
  Vec<Option<Range>>, the graph as an O(n²) boolean adjacency matrix) rather than
  hash maps, because SSA values are dense sequential integers so array indexing
  gives hashing-free O(1) insert/lookup at the cost of O(n²) memory (code).

- 2025-10-23 (fbb7e707) rationale: interference is computed against the
  post-phi-elimination program, not raw SSA live ranges, because a phi copy is
  inserted just before a predecessor's terminator, so the phi destination is made
  to interfere with every value the terminator uses, otherwise the allocator could
  color them to the same register and the copy would clobber a value the branch
  still needs (code).

- 2025-10-26 (adb91990) rationale: after VIR lowering, per-VReg metadata (use
  count, live range, type, and a computed importance score weighting
  frequently-used and loop-carried short-lived values higher) is populated to
  drive the backend window manager's residency and eviction decisions instead of
  purely structural heuristics ([[../../../xir-backend]]) (code).

- 2025-11-01 (7d2d58df) statement: liveness is computed by backward dataflow to
  fixpoint over the CFG (LiveOut[B] = union of successors' LiveIn; LiveIn[B] =
  UEVar[B] ∪ (LiveOut[B] − Def[B])), with phi sources treated as used at
  predecessor exits and parameters live at entry (code).

- 2025-11-01 (abd92456) pitfall: the interference builder originally excluded a
  phi destination from interfering with its own sources to permit coalescing; this
  was unsafe because the phi copy is emitted on the predecessor edge where the
  source is still live, so coalescing turned the copy into a no-op while the
  source was still needed (the environ_get corruption) — a phi destination
  interferes with all of its sources unconditionally (code).

- 2025-11-01 (70957244) pitfall: two phi destinations in the same block must
  always interfere, not only when both are live after the block, because their
  copies execute on the predecessor edge regardless of later use; with no
  dead-code elimination to drop an unused phi's copy, the unconditional
  interference edge is required (code).

- 2025-11-01 (c6976453) pitfall: when phi copies are inserted into a predecessor
  whose terminator reads a vreg that is also a phi-copy destination, the copy
  overwrites the terminator's input before it executes; the lowering must save the
  conflicting inputs into fresh temporaries and redirect the terminator to read
  the temps (code).

- 2025-10-24 (35113cfc) pitfall: a block left with no terminator (e.g.
  unreachable code after a br_table) must lower to an Unreachable instruction
  rather than panicking; such blocks never execute but still flow through lowering
  (code).

## Moves

- 2025-11-01 (fa5c0686) replaced [[vir-two-stage-middle.alt/trivial-baseline-regalloc]]:
  the trivial 1:1 mapping gives every SSA value its own vreg with no reuse;
  CFG-based backward-dataflow liveness plus interference-graph greedy coloring
  lets non-overlapping values share vregs, cutting vreg count 10-30% (the 1:1 form
  cannot express any reuse), and CFG dataflow models control flow correctly where
  the old sequential numbering could not (code).

- 2025-10-18 (9ff77d1b) replaced [[vir-two-stage-middle.alt/identity-window-single-stage-lowering]]:
  the old allocator gave every SSA value its own slot (identity map, no reuse) and
  the lowering targeted an interpreter-only three-register window; coloring
  overlapping live ranges collapses 100-200 SSA values to 10-30 virtual registers
  and the virtual-register IR serves both the interpreter and a future JIT (code).

- 2025-10-20 (bb0d2820) replaced [[vir-two-stage-middle.alt/identity-window-single-stage-lowering]]:
  the first-generation lowering assigned registers by a naive identity map (SSA
  value N to slot N, slot count equal to value count, no liveness, no reuse) baked
  into a single SSA-to-bytecode pass; splitting register allocation into a middle
  stage that runs liveness analysis, builds an interference graph, and colors
  values onto a bounded physical register set lets the same values share registers
  and the code-emitting backend become a thin translation with no allocation logic
  of its own (code).

- 2025-10-22 (1494e762) replaced [[vir-two-stage-middle.alt/type-specialized-vir-instructions]]:
  one instruction per operation carrying an inline ValueType tag mirrors the SSA
  IR's own variants, giving a 1:1 SSA-to-VIR lowering instead of a fan-out of
  per-type variants while still keeping types per-instruction for backend dispatch
  (code).

- 2025-10-23 (a9655e3e) replaced [[vir-two-stage-middle.alt/collapsed-extension-nullable-vir]]:
  the merged VIR instructions had no field to carry the sign-vs-zero extension and
  nullability distinctions, so packed-field gets and nullable cast/test lost their
  meaning during lowering; giving VIR its own per-variant instructions restores
  correct runtime behavior and pushes any fusion to the backend (code).

- 2025-11-04 (e9790144) replaced by [[../mir-single-pass-middle]]: the old middle
  allocated twice — SSA values were coalesced into 10-30 VIR vregs during lowering
  without knowing physical constraints, then a separate window manager
  re-allocated those vregs into 3 interpreter slots — and that two-layer split
  made rematerialization, live-range splitting, and pressure-aware optimization
  impossible because allocation decisions were made before full liveness and
  target information was available; MIR keeps unlimited vregs (1:1 with SSA) and
  defers all allocation to one target-aware pass (code).
