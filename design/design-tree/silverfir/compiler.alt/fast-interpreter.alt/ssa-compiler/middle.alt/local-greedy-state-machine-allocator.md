- The middle-end runs SSA→LIR as an ordered sequence of standalone passes: CFG
  computation, then a phi-elimination pass that rewrites phi nodes into ParCopy
  in place (inserting critical-edge-splitter blocks), then register allocation,
  then a post-allocation LIR peephole optimizer.

- Register allocation is two phases: an Analysis phase computes immutable
  dataflow facts once (liveness via a centralized is_live_at, interference, value
  characterization, slot and frame layout), and a single stateful Allocator
  phase consumes them.

- The allocator is a local greedy forward state machine: it walks blocks in RPO
  in one pass, maintains mutable register-and-spill-slot state, makes dynamic
  eviction and rematerialization-vs-load choices, and emits LIR directly during
  the walk, with every action validated against the state before it mutates the
  state.

- Interference is queried on demand in O(values × slots) against the analysis
  facts rather than built as an O(values²) graph, and the register file tracks a
  value resident in a register and a spill slot simultaneously.

## Facts

- 2025-11-07 (cb29713b) rationale: with only a handful of physical registers
  (three at this stage) almost every value spills, so allocation is framed not
  as classic "which values get registers" but as a cache-replacement problem —
  when to load a value into the register cache and which to evict when full;
  Belady's MIN (evict the furthest next use) is the provably-optimal eviction
  policy and the chosen target, with the eviction policy kept a pluggable
  strategy for later loop-aware and rematerialization refinement (diff).

- 2025-11-08 (90b81fa1) statement: the two major transformations are split at
  different abstraction levels — phi elimination rewrites SSA IR in place (phi
  nodes become PARCOPY on edges, still virtual registers) while register
  allocation is the transformation that changes representation, taking
  phi-eliminated SSA IR to LIR (physical registers, explicit
  SpillLoad/SpillStore, no SSA); keeping values in SSA form through liveness
  means each value is defined once, simplifying liveness and enabling stack-slot
  coloring (~1000 values collapse to ~50–100 spill slots) (diff).

- 2025-11-14 (4cf780ed) rationale: register allocation and LIR lowering were
  split out of a single greedy per-instruction walk into a pure analysis phase, a
  strategy phase committing one immutable global plan, an execution phase
  emitting an abstract action stream, and a separate lowering phase, because the
  single-pass form left global decisions — phi-affinity register/slot sharing
  across blocks, interference-graph spill-slot coloring, rematerialization vs
  spill — impossible to make from purely local state (diff).

- 2025-11-18 (1b254206) rationale: the abstract-Action indirection and separate
  lowering phase were dropped for a validated state machine that emits LIR
  directly (commit validates all actions atomically before applying any) —
  roughly four times less code, fewer invariants to violate, easier to trace, at
  the cost of losing action-level testing, mitigated by testing at the LIR level;
  this was a reorganization with no expressivity change (diff).

- 2025-11-22 (af982cf2) rationale: the artificial Analysis-vs-Heuristics phase
  boundary was dissolved — most of what Heuristics computed (frame layout, slot
  coloring, phi affinity, remat costs) is a static fact belonging in Analysis,
  while the genuinely dynamic decisions (which value to evict, whether to swap
  commutative operands, where to materialize) depend on live allocator state and
  cannot be precomputed — leaving two phases: static AnalysisResult plus
  on-the-fly decisions through an Availability view and a Materializer (diff).

- 2025-11-23 (fa39233b) rationale: a read-only planner that batched a whole
  transaction before commit could not see the effect of actions it had already
  decided, so its intermediate state queries went stale; emitting each action
  immediately keeps every later decision querying current reality (diff).

- 2025-11-22 (af982cf2) rationale: the load/remat/eviction logic, formerly
  duplicated across the register-based, slot-based, terminator, and reconciliation
  planners, was unified behind one Availability/Materializer entry point that
  every place needing a value routes through (diff).

- 2025-11-21 (9303139d) rationale: because SSA values are immutable, after a
  Spill action the register and its slot hold the same valid value, so a Spill no
  longer clears the register and the "first predecessor wins" merge target can no
  longer be just a value-to-register layout — predecessors must reconcile BOTH
  the register file and the spill state, so the merge target became a full
  block-state snapshot (diff).

- 2025-11-21 (0dab4f24) rationale: spill-slot coloring needs only pointwise
  live-range overlap, not an all-pairs interference graph; checking live-range
  overlap on demand during coloring (each value carries explicit per-block
  [start,end] ranges) avoids building and storing an O(V^2) graph (diff).

- 2025-11-23 (b52a9092) rationale: merge-point exit reconciliation handles only
  the single-successor case, because critical edges are split before allocation,
  so every block with two or more successors has successors that each have
  exactly one predecessor and inherit this block's exit state directly (diff).

- 2025-11-25 (0532fb14) pitfall: over-spilling at a merge is NOT always safe
  under slot coloring — non-overlapping live ranges share a slot, so a value
  spilled on one predecessor path may correspond to a slot holding a different
  value on another path; with first-predecessor-wins inheritance a merge block
  could inherit a spilled flag and a later load read the stale slot, so all
  predecessors are forced to identical exit spill state (diff).

- 2025-11-13 (39335672) rationale: precomputing a next-use distance for every
  (instruction, live-value) pair cost ~100K entries per function; storing only
  each value's sorted use positions per block (~100 entries) and computing the
  Belady-MIN distance on demand gives the same queries at a fraction of the
  memory (diff).

- 2025-11-13 (cff984a8) rationale: a value with no future use carries nothing
  worth preserving, so its register is reclaimed at zero cost; scanning for a
  dead register before applying Belady's MIN turns a forced spill into a free
  eviction (diff).

- 2025-11-13 (198de8c6) rationale: under the two-address constraint the result
  reuses the first input's register, destroying that input; for a commutative op
  the inputs are reordered so the result lands on a dead operand's register,
  sparing a live first operand from being spilled and reloaded (diff).

- 2025-11-22 (737f609a) rationale: eviction victim selection is not pure Belady —
  within a next-use band the cost model breaks ties by eviction cheapness,
  preferring a value cheaper to rematerialize than to reload, then an
  already-spilled/rematerializable value, then a value that must be spilled; the
  displaced occupant is preserved only when live AFTER the current instruction
  (diff).

- 2025-11-17 (e011fbbe) rationale: a slot-based operation (Call/ParCopy) clears
  all physical registers, so only values live after it need preserving;
  precomputing this live-after set per slot-based instruction lets execution skip
  the dead-value spills, the same merge-spill-reduction principle extended to
  call/parcopy boundaries (diff).

- 2025-11-19 (a7eec156) rationale: after each commit the state machine clears
  every register whose value is no longer live, which lets action validation drop
  all per-occupant liveness tests — any value still occupying a register is by
  construction live (diff).

- 2025-11-19 (48fb0fe5) rationale: a batch of actions is validated atomically
  against a temporary clone of the state mutated as each action is checked, so a
  within-batch spill-then-load-into-the-freed-register validates correctly, and
  the real state is mutated only once the whole batch passes (diff).

- 2025-11-20 (cddb5581) rationale: target register constraints are validated
  generically from the target's OpInfo/OperandConstraint (per-operand-form
  input/output counts, FixedReg, SameRegAs), replacing a hardcoded
  per-instruction-signature check; this is how XIR's two-address rule is enforced
  at allocation time (diff).

- 2025-11-12 (ed23ac48) rationale: a two-address ISA (XIR, x86) writes the output
  over the first input register and forces a spill of that input if still live,
  while a three-address ISA (ARM, RISC-V) writes to an independent register;
  modeling it as a per-target boolean lets one allocator serve both ISA families
  (diff).

- 2025-11-12 (18b73732) rationale: the number of register operands an
  instruction has after lowering can differ from its SSA use/def count — calls
  lower to zero register operands (args/results travel through spill slots),
  parallel copies keep one operand per pair — so the allocator queries this LIR
  signature, never trying to allocate more register operands than the lowered
  form uses (diff).

- 2025-11-25 (0532fb14) rationale: empty-block removal is split from the
  pre-regalloc SSA CFG pass to a post-regalloc LIR pass because removing empty
  (Br-only) blocks on SSA before allocation would destroy the critical-edge
  splitter blocks and recreate critical edges the allocator depends on being
  absent; the pre-regalloc SSA pass does only unreachable-block removal (diff).

- 2025-11-25 (0532fb14) rationale: post-regalloc LIR empty-block removal
  redirects branches to bypass empty blocks and leaves them orphaned in place
  rather than physically deleting and renumbering, because renumbering destroyed
  the block-ID correspondence with the SSA IR that makes IR dumps and debugging
  tractable; the backend never emits an orphan because RPO emission cannot reach
  it (diff).

- 2025-11-14 (029c3135) rationale: values cheap to recompute (compile-time
  constants, and conceptually pure small integer arithmetic and single-input
  conversions) are rematerialized on demand instead of spilling and reloading,
  gated on the recompute being cheaper than a spill/reload round-trip and
  side-effect-free; arithmetic rematerialization is currently disabled, leaving
  only constants (which need no input registers) rematerialized, because at the
  small register count recomputing arithmetic is rarely profitable (diff).

- 2025-11-14 (8b7cb12b) rationale: small-pure-arithmetic rematerialization was
  disabled, narrowing remat to compile-time constants only — at the XIR target's
  3 physical registers recomputing arithmetic is rarely profitable and the
  lowering path could not yet recompute a SmallOp recipe without fabricating
  registers — so only constants, which need no input registers, are
  rematerialized (diff).

- 2025-11-14 (8b7cb12b) rationale: batch per-instruction spill planning is gated
  on register count — on a small register file (e.g. XIR's 3) all of an
  instruction's operands are planned together to minimize total spill traffic,
  while a larger register file has enough headroom that simpler sequential
  per-operand allocation suffices (diff).

- 2025-11-16 (8bc90fd4) pitfall: LIR lowering walks blocks in RPO order but must
  preserve each block's SSA BlockId as its LIR index, because terminators encode
  branch targets by BlockId; appending lowered blocks in RPO sequence put SSA
  block N at its RPO position, so branches targeted the wrong block and produced
  infinite loops — blocks are pre-allocated up front so each lands at its own
  block_id position (diff).

- 2025-11-18 (1b254206) pitfall: a two-address output assigned the same physical
  register as an input must spill that input to its slot before the output
  overwrites the register if the input is live after the instruction, or the
  live-through input is silently clobbered (diff).

- 2025-11-18 (62dd27bc) pitfall: the multiple result-defs of a single
  multi-value Call must mutually interfere in the graph, or two results can be
  colored to the same spill slot; the interference builder originally made each
  def interfere only with values live after the instruction, never with its
  sibling defs (diff).

- 2025-11-23 (77387f78) pitfall: a block's allocation state is seeded from a
  single predecessor's exit snapshot, but a predecessor's live-out is the union
  of all its successors' live-in sets; values live into a sibling successor but
  not this block must be cleared from the inherited register file at block init or
  they linger and can be wrongly reused as still-resident (diff).

- 2025-11-08 (db34d718) pitfall: the Return terminator's uses() returned an empty
  set, so return values were never counted as live at the return and could be
  freed early; uses() must report the returned values so they survive to the
  return (diff).

- 2025-11-10 (11f72dc7) pitfall: a phi destination is written by a separate
  ParCopy on each predecessor edge; if each edge's copy allocates a fresh
  register for the destination, the phi value lands in a different register per
  incoming edge and the successor reads garbage, so the lowering must reuse the
  destination's already-assigned register when one exists (diff).

- 2025-11-10 (11f72dc7) pitfall: the next-use distance that drives Belady's MIN
  must count a use AT the current instruction as distance 0; using
  strictly-greater skipped the current instruction's own operands, letting a
  value the instruction needs be chosen as the eviction victim (diff).

- 2025-11-12 (ef373e16) pitfall: bulk memory/table operations (memory.copy/init,
  table.copy/init) produce no value but were classified as three-input-one-output,
  making the lowering demand a nonexistent output register; they belong in the
  three-input-no-output signature (diff).

- 2025-11-11 (31aa7ff5) pitfall: loading a multi-operand instruction's inputs one
  at a time lets Belady's-MIN eviction spill an operand already loaded for the
  same instruction; each materialized operand must be added to a protected set so
  the eviction policy cannot pick it while later operands are loaded, and outputs
  are likewise protected as assigned (diff).

- 2025-11-17 (e011fbbe) pitfall: the first selective-spill implementation iterated
  only over register-resident values, so a value live across a Call but held as a
  rematerialization recipe was never written to its slot and became unrecoverable;
  the loop is driven from the strategy's values_to_spill set, materializing a
  remat-only value into a temp register and spilling it, and must include the
  operation's own argument values (diff).

- 2025-11-23 (4406b689) pitfall: the LIR peephole pass that merges consecutive
  SpillLoad/SpillStore into the batched (up to three pairs) form may only combine
  pairs whose registers are mutually disjoint, because the multi-pair handlers are
  permutation-specialized and exist only for non-overlapping register sets; a
  later load may be hoisted to join an earlier one only if no skipped instruction
  reads or writes that load's destination registers ([[../../xir-backend]]) (diff).

- 2025-11-24 (da5de8c0) rationale: post-register-allocation LIR cleanups (empty
  Br-only block removal, dropping a Const overwritten before being read,
  deduplicating self-copies) are placed after allocation rather than before
  because at this point the allocator no longer cares about critical edges, so
  empty-block removal and branch redirection that would create critical edges are
  safe (diff).

- 2025-11-19 (9b9fcee2) rationale: LIR was emitted inline as each action was
  applied; accumulating validated Actions and lowering them in one terminal pass
  over the stream decouples allocation from lowering and lets SSA instructions and
  terminators flow through the same validated stream (diff).

- 2025-11-20 (c00eb4d9) rationale: each spill was given an explicit intent
  (preserve a value, materialize an input for a slot-based op, or place a return
  value for the ABI) so the live-after / live-at / no-check exemption is tagged
  rather than inferred from instruction position, replacing a positional
  at-terminator spill-exemption heuristic (diff).

- 2025-11-11 (89b2e5f4) rationale: CFG optimization is split into a
  critical-edge-preserving SSA pass before regalloc and an empty-block removal on
  LIR after regalloc, because running block merging and empty-block removal on SSA
  before allocation would destroy the critical-edge splitter blocks and recreate
  critical edges the allocator depends on being absent (diff).

- 2025-11-13 (93c07c28) rationale: emitting a Const+spill-store per local to
  satisfy WebAssembly's zero-initialized-locals rule cost two instructions per
  local; since the call frame already allocates its spill-slot array zeroed,
  pre-assigning each local a fixed spill slot makes it zero-initialized for free
  with no emitted instructions (diff).

- 2025-11-16 (ab35dc17) rationale: the allocator's target parameter was promoted
  from a concrete config struct carrying only a register count to a target
  descriptor trait each backend implements, because the concrete config could not
  hold per-instruction operand constraints, so the planner had to derive
  signatures itself; the trait lets each backend supply op_info/signatures per
  instruction, making the allocator core genuinely backend-agnostic (diff).

- 2025-11-29 (e5f0201b) rationale: the register file's single-location map could
  not represent a value held in two registers at once, so a live-after value had
  to be spilled to memory before its register was overwritten; multi-location
  tracking lets the allocator copy it to a free register instead (diff).

- 2025-11-12 (ed23ac48) pitfall: register occupancy tracked as two
  hand-synchronized maps (value→register and register→value) bred silent
  stale-entry bugs whenever a path updated one direction and forgot the
  other; a single bidirectional structure owning both directions and
  enforcing one-value-per-register / one-register-per-value on every insert
  and removal eliminated that bug class (diff).

## Moves

- 2025-11-09 (d04cbd44) replaced [[local-greedy-state-machine-allocator.alt/mir-single-pass-middle]]:
  VIR was just SSA IR without phi nodes -- not enough differentiation to justify
  a separate IR; the real IR boundary is virtual registers (SSA IR) to physical
  registers (LIR) (diff).

- 2026-02-10 (0c078fa6) replaced by [[../middle]]: the per-block single-pass
  state-machine allocator could not split live ranges or take a global view —
  calls flushed all registers, there was no copy coalescing, and only constants
  were rematerializable — so it was replaced by a regalloc2-inspired bundle
  allocator whose bundle merging subsumes phi coalescing and adds live-range
  splitting (diff).
