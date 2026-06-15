- Liveness numbers all instructions sequentially across blocks in program order
  and takes each value's live range as the half-open interval [def, last_use+1)
  over that numbering.

- The interference graph is a dense O(n²) adjacency matrix over all SSA values,
  with an edge whenever two values' live-range intervals overlap.

- Register allocation is greedy graph coloring: parameters take vregs
  0..num_params-1, remaining values are ordered most-constrained-first by
  interference degree and each takes the lowest vreg not used by an interfering
  neighbor.

- Call-argument values are allocated in a dedicated first pass and their vregs
  reserved, a defensive guard against the over-conservative live ranges
  corrupting argument setup.

## Facts

- 2025-11-01 (f668caff) pitfall: the sequential-numbering liveness was
  structurally unable to model a CFG — it extended a value's range through every
  intermediate block in program order and to end-of-function for backward uses, so
  loop/backward-edge code over-approximated liveness; the symptoms were patched
  with defensive reservations in the allocator and extra phi edges in the
  interference builder rather than fixed at the root, leading to this whole
  liveness/interference layer being discarded instead of repaired (diff).

- 2025-10-29 (8518fdc8) pitfall: greedy coloring assigns each value the lowest
  vreg not used by its interference neighbors, but when block-execution order
  differs from program order the analysis can wrongly conclude a parameter and a
  later-defined local do not interfere, so a non-parameter reuses a parameter's
  vreg; the fix reserves all parameter vregs (0..num_params) as permanently used
  during coloring (diff).

- 2025-10-29 (3bbcea91) pitfall: the same misjudged-interference failure recurs
  for SSA values consumed as call arguments when a phi result is used across a
  backwards CFG edge, so coloring runs in two passes — pass one colors all
  call-argument values and records their vregs, pass two colors the rest while
  reserving both parameter and call-argument vregs (diff).

- 2025-10-31 (d82f00d8) pitfall: live ranges are computed over a single global
  linear numbering, but block layout order need not match control-flow order; a
  value used in two blocks gets a contiguous [def,last_use] range with a hole and
  a value defined between is wrongly judged non-interfering — fixed by adding
  synthetic uses at the entry and terminator of every block a value is used in
  (diff).

- 2025-10-25 (7df9290e) pitfall: SSA blocks are not stored in execution order, so
  a value can be used at an index earlier than its definition (a backward use);
  the live range must then be conservatively extended from one index before the
  earliest use through the end of the function, otherwise touching-but-non-
  overlapping ranges fail to interfere and the slot is wrongly reused (diff).

- 2025-10-24 (b7b5ded6) pitfall: a phi source must be kept live through the entire
  predecessor block, not merely at the phi node, because the phi-elimination copy
  is inserted before that predecessor's terminator; recording a use only at the
  phi lets the allocator reuse the source's register inside the predecessor and
  corrupt the value (diff).

## Moves

- 2025-11-01 (f668caff) replaced by [[trivial-baseline-regalloc]]: the
  sequential-numbering liveness over-approximated live ranges across block
  boundaries and was patched with defensive reservations rather than fixed at the
  root, so the whole liveness/interference/coloring layer was torn down to a
  known-correct trivial 1:1 baseline pending a documented CFG-dataflow redesign
  (diff).
