- The joint planner's lightweight stack simulation (structural fills and
  spills only, cache-free, no capacity clamp) and the rewriter's live-window
  state (`BlockState`: value window, type stack, alias tags, budget-aware
  capacity spilling and cache eviction) are two separate implementations of
  the operand-stack discipline.

- The planner's per-op structural match and the rewriter's per-op prefix match
  are maintained in parallel; a change to spill discipline must be mirrored in
  both, with the planner kept a conservative upper bound of the rewriter.

## Moves

- 2026-07-12 (35a439c7) replaced by [[stack-discipline]]: the planner's
  lightweight simulation and the rewriter's live-window discipline were two
  implementations of one stack policy that had to be manually mirrored (the
  planner a conservative upper bound of the rewriter); extracting one engine
  with measure and emit drivers makes divergence structurally impossible and
  let the later exact-plan walker reuse the same transitions (code)
