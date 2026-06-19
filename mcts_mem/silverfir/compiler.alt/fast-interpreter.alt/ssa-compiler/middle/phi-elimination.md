- Phi nodes are eliminated during allocation, after each block is lowered: a
  fresh move-only block is generated on every phi edge carrying copies, which
  the predecessor's terminator is retargeted to branch through before reaching
  the original successor.

- The copies carry parallel semantics — all sources are read before any
  destination is written, leaving a phi destination that is also a copy source
  on the same edge uncorrupted; identity copies (src == dst) are dropped.

- The edge copies move values between spill slots, and are sequentialized at
  compile time by a solver that topologically orders the slot-to-slot copies
  and breaks dependency cycles through a single reserved scratch slot; the
  move-only block then executes the pre-ordered copies in place with no
  allocation.

## Facts

- 2025-11-04 (a3fdaebd) measurement: eager phi elimination (copies inserted at
  predecessor ends before register allocation) dominated emitted code — on the
  args_get test one function had 340 instructions, ~200 of them window
  load/store, and one block of 11 phis expanded to 22 copies and ~40 window ops
  (~60% overhead) — because the allocator could not collapse identity copies
  when register assignments were not yet known; this drove deferring ParCopy
  resolution until after allocation (code).

- 2025-11-01 (c6976453) pitfall: SSA can contain unreachable blocks carrying phi
  nodes whose incoming edge does not exist in the lowered CFG, so phi-copy
  insertion must skip any edge whose predecessor does not actually list the
  successor (code).

- 2025-11-03 (235c1cd3) pitfall: phi-copy placement must classify the edge, not
  just split-or-insert-in-predecessor: when the successor has a single
  predecessor but the predecessor has multiple successors, the copies must go at
  the BEGINNING of the successor, never the end of the predecessor — placing them
  in the predecessor executes them on all of its branches, corrupting the other
  successors (code).

- 2025-11-06 (ae3cd9e8) rationale: PARCOPY placement is decided from the actual
  post-split CFG topology, not by assuming the original SSA edges survive: for
  each phi source the pass locates the real edge and places the copy where it
  cannot clash with a terminator's inputs — at the end of a single-successor
  predecessor, at the start of the successor when the predecessor has multiple
  successors, or at the end of a landing pad — so the older need to save
  terminator inputs to temporaries is removed (code).

- 2025-11-11 (89b2e5f4) pitfall: collecting all critical edges up front and
  splitting them in a batch is unsound — inserting a splitter block renumbers
  block ids and can change whether other edges are critical, so the batch worked
  on stale ids; edges must be split one at a time, recomputing the CFG after each
  split, looping until none remain (code).

- 2025-11-11 (89b2e5f4) rationale: once phi nodes become ParCopy the branch
  terminators' value lists are redundant and are cleared, but the Return
  terminator keeps its value list because the register allocator reads it to place
  results into the calling-convention result slots (code).

- 2025-11-23 (4829a2e1) rationale: a ParCopy lowers to copies between spill
  slots; copies whose source and destination resolve to the same slot are
  dropped, and a ParCopy that becomes entirely self-copies emits no instruction
  at all (code).

- 2025-11-24 (02756381) statement: the frame layout reserves exactly one temp
  spill slot at a fixed position immediately after params and locals
  ([params][locals][parcopy_temp][temporaries]); a single slot suffices because
  the parallel-copy solver breaks cycles one source at a time, never needing more
  than one scratch location (code).

## Moves

- 2025-10-23 (fbb7e707) replaced [[phi-elimination.alt/per-predecessor-phi-copies]]:
  the per-predecessor representation keyed copies by predecessor block alone and
  so could not express which successor a copy belonged to, so a copy inserted
  before a multi-successor predecessor's terminator executed on every successor
  edge until critical edges were split with a landing-pad block that runs the
  copy only on its intended edge (code).

- 2025-11-06 (f8a05906) replaced [[phi-elimination.alt/edge-keyed-parcopy]]: a
  critical edge (predecessor with multiple successors into a successor with
  multiple predecessors) has no block to host its copies, so edge-keyed PARCOPY
  could not place them unambiguously; splitting critical edges into landing-pad
  blocks first makes every edge non-critical, so each PARCOPY lands either at the
  end of a single-successor predecessor or the start of a single-predecessor
  successor (code).

- 2025-11-24 (02756381) replaced [[phi-elimination.alt/runtime-parallel-copy]]:
  the runtime parallel-copy handler allocated a temporary Vec on every execution
  to read all sources before writing any destination; moving that work to
  compile time — a solver that topologically orders the copies and breaks cycles
  with one reserved frame temp slot — lets the runtime handler just execute the
  pre-ordered copies in place with no allocation (code).
