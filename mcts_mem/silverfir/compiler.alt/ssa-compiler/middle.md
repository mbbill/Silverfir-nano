- The middle stage lowers SSA IR to LIR by allocating a fixed physical
  register set per block and then eliminating phi nodes, producing a linear
  instruction sequence with explicit spill loads and stores
  ([[phi-elimination]]).

- Register allocation is driven by liveness analysis that groups values into
  affinity bundles — phi operands, parallel-copy pairs, and two-address
  operand/result pairs are biased toward sharing a register — assigned
  registers greedily; the shipping allocator lowers each block naively,
  honoring those bundle/hint preferences first.

- Values that cannot stay in registers are spilled to numbered stack slots;
  the production slot plan gives every value its own slot (correctness-first,
  no slot sharing). Coloring non-overlapping live ranges onto shared slots is
  an opt-in/phase-2 path that falls back to per-value slots when its checker
  fails.

- The target's register count is a parameter of the stage (eight for the XIR
  target), not hardwired; the same allocator can serve a wider-register
  target. The target descriptor supplies only the register count and the
  caller-saved register mask; per-instruction operand counts, the two-address
  rule, and commutativity are properties of the op and the structural
  validator, not the descriptor.

- The calling convention is register-based for the first eight parameters and
  results (placed in registers v0..v7) with a spill-slot fallback for any
  beyond eight; the return instruction carries its result values explicitly
  for the allocator to place into the convention's result locations.

- Each lowering is guarded by a value-flow checker (a cell dataflow proving
  each use reads its expected SSA value from its physical location), plus an
  independent structural validator (two-address legality, spill-slot bounds,
  branch-target and RPO well-formedness, CFG edge-origin equivalence).

- Call arguments are softly biased toward their call-site registers: the
  allocator hints each argument-i bundle toward register i, consumed as a
  preferred-register mask by the greedy bundle assignment.

## Facts

- 2025-11-07 (cb29713b) rationale: register allocation is placed in the middle
  stage rather than in each backend so the algorithms (liveness, Belady
  next-use eviction, loop/dataflow awareness) are written once and shared across
  future backends, with only the backend-specific constraints — per-instruction
  signatures, the two-address output-overwrites-first-input rule, the calling
  convention — exposed through a target descriptor; this supersedes the VIR-era
  backend window manager, whose problem was diagnosed as lack of liveness
  information, not its location in the backend (code).

- 2025-11-08 (023bfc2f) rationale: call arguments and results pass through spill
  slots rather than registers — all registers are caller-save and spilled before
  a call anyway, slots impose no limit on arg/result count, and the backend
  reads arguments and writes results directly from backing storage — so register
  shuffling for calls is avoided (code).

- 2025-11-21 (da2bf733) rationale: the XIR interpreter keeps the whole abstract
  register set in global state shared across the entire tail-call dispatch
  chain, so a callee clobbers every caller register; the allocator models a Call
  as clearing the entire register file — any value live across a call is spilled
  before it and reloaded after, and call results are produced directly in spill
  slots ([[../xir-backend]]) (code).

- 2025-11-08 (023bfc2f) rationale: parallel-copy resolution follows Boissinot et
  al., "Revisiting Out-of-SSA Translation" (CGO 2009) — topological order for the
  acyclic part, a single reserved scratch slot to break cycles (code).

- 2025-11-29 (d67ead12) measurement: widening the XIR target from three to eight
  physical registers cut spill/load traffic at no per-instruction cost — Perm3
  generated 1,563 handlers and Perm8 generated 14,868 (~10x) yet both measured
  ~0.5 ns per instruction; eight was chosen because the abstract registers ride
  in CPU registers through the preserve_none convention (up to 13 register
  arguments on x86-64), so eight register-args plus destination and context
  still fit without spilling the abstract register set itself (code).

- 2025-11-08 (4d2ca206) rationale: the Return terminator carries its SSA result
  values explicitly even though the signature fixes the result count, because
  the allocator needs the values to place them into the calling convention's
  result slots before the return executes ([[../ssa-frontend]]) (code).

- 2026-02-11 (1bb7185e) rationale: the allocation stage is swappable behind one
  allocator trait so the rest of the pipeline (liveness, phi elimination, edge
  moves, value-flow checker, LIR emission) stays fixed while the allocator
  evolves (code).

- 2026-02-11 (1bb7185e) statement: the stated priority order for the allocator
  redesign is correctness first, copy elimination second, compile speed third
  (sourced).

- 2026-02-08 (0fa96d4e) rationale: the bundle allocator's structural payoff is
  that bundle merging IS phi coalescing, so SSA deconstruction moves post-RA and
  ParCopy stops being a special case (a worked example cut 6 moves/iteration to
  0); live-range splitting uses half-open [from,to) intervals to remove the false
  interference inclusive endpoints caused at ParCopy points, spill weight is
  loop-depth-weighted so hot values allocate first, and rematerialization is
  extended to GlobalGet/AddImm — full predecessor diagnosis (P1–P8) in
  [[middle.fact/local-greedy-wall]] (code).

- 2026-02-13 (483c2c80) rationale: the call-argument register hint is suppressed
  only for a parameter already used at its own call-site position (param p used
  as arg p), not for every parameter, and hinted bundles get their spill weight
  doubled so the allocator keeps call arguments in registers rather than
  spilling them (code).

- 2026-02-10 (087d67cf) pitfall: sequentializing the parallel copies on a CFG
  edge can require breaking a permutation cycle (a→b, b→a), impossible without a
  free location, so the slot planner reserves one extra scratch spill slot beyond
  the per-value/per-coalesced-group slots specifically as the cycle-breaking
  temporary (code).

- 2026-02-11 (7aacd61d) pitfall: phi sources must be folded into a predecessor's
  live-out inside the backward-dataflow fixpoint, not as a one-shot post-pass
  correction; seeding phi sources into live-out only after convergence left a
  phi source that was live solely because of the phi edge never propagating
  transitively up the predecessor chain (code).

- 2026-02-11 (7aacd61d) rationale: register residency is carried across block
  boundaries by inheriting predecessor exit state, generalized from
  single-predecessor inheritance to multiple predecessors by intersecting their
  exit register maps (a register keeps a value only where all lowered
  predecessors agree) and filtering to live-in values, with not-yet-lowered
  back-edge predecessors contributing no state (code).

- 2026-02-12 (c1078fff) rationale: when no preferred/hinted register is
  available the naive lowering picks the least-costly register (score every
  register by how expensive its occupant is to displace, free registers scoring
  zero) rather than the first free register, so an arbitrary in-block destination
  evicts the cheapest live value (code).

- 2026-02-11 (882c0fe8) pitfall: the register copy handler was the one XIR
  operation still decoding operands at run time (a generic handler reading
  src/dst from immediates with a double match, cited as 45% of runtime); it was
  given a dedicated permutation signature whose output register is unconstrained
  (not tied to input 0 by the two-address rule) so the (src,dst) pair bakes into
  the handler pointer at build time ([[../xir-backend]]) (code).

## Moves

- 2026-02-10 (0c078fa6) replaced [[middle.alt/local-greedy-state-machine-allocator]]:
  the per-block single-pass state-machine allocator could not split live ranges
  or take a global view — calls flushed all registers, there was no copy
  coalescing, and only constants were rematerializable — so it was replaced by a
  regalloc2-inspired bundle allocator whose bundle merging subsumes phi
  coalescing and adds live-range splitting (code).
