- Register allocation is a trivial 1:1 mapping: each SSA value gets its own
  unique vreg (vreg i for value i), with no liveness, no interference, and no
  reuse, the vreg count equalling the SSA value count.

- Parameters occupy vregs 0..num_params-1 by convention; the mapping is a
  complete, always-correct baseline consumed by VIR lowering.

## Moves

- 2025-11-01 (f668caff) replaced [[trivial-baseline-regalloc.alt/sequential-liveness-coloring]]:
  the sequential-numbering liveness over-approximated live ranges across block
  boundaries and was patched with defensive reservations rather than fixed at the
  root, so the whole liveness/interference/coloring layer was torn down to a
  known-correct trivial 1:1 baseline pending a documented CFG-dataflow redesign
  (diff).

- 2025-11-01 (fa5c0686) replaced by [[vir-two-stage-middle]]: the trivial 1:1
  mapping gives every SSA value its own vreg with no reuse; CFG-based
  backward-dataflow liveness plus interference-graph greedy coloring lets
  non-overlapping values share vregs, cutting vreg count 10-30% (the 1:1 form
  cannot express any reuse), and CFG dataflow models control flow correctly where
  the old sequential numbering could not (diff).
