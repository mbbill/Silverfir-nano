- The middle stage lowers SSA IR into a distinct mid-level IR (MIR) that
  preserves every SSA value as an unlimited virtual register (1:1 with SSA
  values), tracking each vreg's originating SSA value; phi nodes are eliminated
  into edge-attached PARCOPY instructions during this lowering, with all register
  allocation deferred to a later target-aware pass.

- The middle pipeline is a fixed phase sequence over MIR: SSA-to-MIR lowering
  (CFG computation, critical-edge splitting, phi elimination), RPO computation,
  optional MIR verification, liveness analysis, register allocation, spill-code
  insertion, and PARCOPY resolution into sequential moves, producing allocated
  MIR for backend codegen.

- Register allocation targets a stack-machine target of three physical registers
  (the top three stack-window slots), where an instruction's output overwrites
  its first input's register; the register count and per-instruction
  input/output signatures are supplied by a target descriptor, leaving the
  allocator target-parameterized.

- Spill load and spill store are MIR instructions carrying no vreg uses or defs
  of their own (physical-register management inserted by the allocator).

## Facts

- 2025-11-04 (cd72d845) statement: physical registers are assigned by a
  linear-scan allocator — intervals built from liveness segments, sorted by start
  point, scanned with expiry of inactive intervals, a free physical register
  taken per live vreg, a victim chosen and spilled when none is free (code).

- 2025-11-04 (5735533b) rationale: liveness is computed at two granularities —
  coarse block-level live-in/out and fine instruction-level live segments within
  each block — and interference is checked at segment precision so
  sequentially-defined values whose ranges do not overlap inside a block do not
  falsely interfere (code).

- 2025-11-05 (49c3bf49) rationale: the conceptual allocation result is a 2D table
  location[vreg][program-point], too large to materialize, so it is stored
  compressed as per-vreg allocation segments (start, end, location) and
  reconstructed pointwise, letting a vreg's location differ across its lifetime
  (live-range splitting) without an O(vregs × instructions) array (code).

- 2025-11-05 (49c3bf49) rationale: spilling is by live-range splitting rather
  than whole-range eviction — a vreg's lifetime is cut into segments and only the
  cold segments take a spill slot while hot segments stay in registers — which is
  the lever that makes a three-register target viable (code).

- 2025-11-05 (ee2e38f6) rationale: hot vs cold for a live segment is decided by
  use density and loop membership (a segment with many uses or whose block is in
  a loop is hot), with loop membership detected from a back-edge (a successor not
  later in program order) (code).

- 2025-11-05 (30af9b2b) rationale: cross-block ordering in the allocator uses a
  computed reverse-post-order index, not BlockId order, because BlockId order
  does not respect program order once loops are present; RPO puts loop headers
  before their bodies and makes back-edges run high-to-low index, which the
  next-use-distance and forward/backward-edge decisions depend on (code).

- 2025-11-04 (d5a578a3) rationale: PARCOPY resolution runs only after physical
  registers are known, because the move sequence and whether a temporary is needed
  depend on the allocation — only register-to-register pairs are emitted, identity
  copies collapse once src and dst share a register, and a cycle's temp can only
  be picked from the target's physical register set (code).

- 2025-11-05 (cf56385e) statement: in debug builds the allocation result is
  checked by walking every program point and asserting no two vregs live there
  share a physical register and that a vreg's split segments do not overlap — the
  executable form of the allocator's core invariant (code).

- 2025-11-05 (f7ba83d0) pitfall: the temp vreg a SpillLoad writes must be a
  register not already live at that point; the original picker was a stub that
  always returned the first allocatable register, so a reload could clobber a live
  value — the choice now scans all vregs' locations at the load's exact (block,
  inst) and takes the first register not in use (code).

- 2025-11-05 (f7ba83d0) pitfall: a register handed to a split range's hot segment
  must be kept out of the linear scan's free pool, tracked in split_allocated_regs
  and skipped by the free-register picker, so the scan cannot re-hand-out a
  register a split hot segment still holds; interval expiry was tightened to free
  a register only once its interval ends strictly before the current point (code).

- 2025-11-05 (d6ffb943) rationale: the spill-victim heuristic was changed from
  lowest-static-spill-cost to Belady's furthest-next-use — the lowest-cost
  heuristic only compared static costs and so could evict a value about to be
  reused while keeping a long-dead one, whereas Belady evicts the value whose next
  actual use is furthest away, minimizing reloads; next-use distance is exact
  same-block and a heuristic cross-block (block gap times average block length, a
  large constant for loop back-edges) since only the relative ordering matters
  (code).

- 2025-11-05 (d6ffb943) pitfall: the spill-victim search must exclude any vreg
  read by the current instruction (spilling a value needed right now emits a
  reload from a slot before its store, reading garbage); the earlier lowest-cost
  heuristic had no such guard (code).

- 2025-11-05 (bc73259c) pitfall: the spill-victim search must also exclude any
  active vreg that is a PARCOPY source on an incoming edge to the current block,
  because the edge PARCOPY's sources are read at the block transition, so spilling
  such a vreg when allocating at the block start would emit its reload before its
  store (code).

- 2025-11-05 (f7ba83d0) pitfall: use-rewriting after spill-code insertion must
  index against the block's ORIGINAL instruction count, not the live length —
  inserting SpillStore/SpillLoad shifts indices, so a post-insertion length
  misidentifies the terminator and mis-targets rewritten uses (code).

## Moves

- 2025-11-04 (e9790144) replaced [[mir-single-pass-middle.alt/vir-two-stage-middle]]:
  the old middle allocated twice — SSA values were coalesced into 10-30 VIR vregs
  during lowering without knowing physical constraints, then a separate window
  manager re-allocated those vregs into 3 interpreter slots — and that two-layer
  split made rematerialization, live-range splitting, and pressure-aware
  optimization impossible because allocation decisions were made before full
  liveness and target information was available; MIR keeps unlimited vregs (1:1
  with SSA) and defers all allocation to one target-aware pass (code).

- 2025-11-09 (d04cbd44) replaced by [[../local-greedy-state-machine-allocator]]:
  VIR was just SSA IR without phi nodes -- not enough differentiation to justify
  a separate IR; the real IR boundary is virtual registers (SSA IR) to physical
  registers (LIR) (code).
