- An exact-simulation walker re-walks each semantic block through the shared
  [[stack-discipline]] engine with the solved residency armed, mirroring the
  rewriter's per-op cache decisions, and computes the exact per-block
  cached-local entry and exit rows plus the per-edge boundary-repair actions
  (`compute_exact_plan`).

- The rewriter consumes the walker's rows and repair actions as authority: it
  seeds each block's cache state from the exact entry row, publishes the plan's
  rows, and emits semantic-edge repair blocks from the plan's content-deduped
  action pool; a standing debug assertion compares the lowered exit reality to
  the plan and a structural check ties block and row counts together.

- Plan rows and repair actions live in flat arenas addressed by span handles
  with a content-hash-deduplicated action pool; the exact rows and repair
  actions add no per-block heap containers to the plan.

- Exit rows and edge actions are plan-internal: consumed by the rewriter,
  dropped with the planner, never stored on the program, never read by machine
  lowering.

- Bridge blocks and the entry repair are synthesized emit-side with their rows
  derived at synthesis time; the plan enumerates semantic blocks and edges
  only.

- The per-block planned-resident set survives as the mid-block admission
  policy consumed by local-access decisions; a call invalidates entry
  residency but not admissibility.

## Facts

- 2026-07-12 (e04f8d69) measurement: the entry filter is an identity on all of
  spectest but not on the WASI corpus (a coremark block trims two
  never-qualifying residents from a five-slot tentative set), yet reseeding
  lowering from the exact rows left native counts and MachineIR byte-identical
  everywhere — the phantom residents were already elided downstream (code).

- 2026-07-12 (2c2e010e) measurement: per-block heap containers for the exact
  rows dominated plan memory (the worst lua function retained ~430 KiB of
  planner rows into rewrite, echoing the a3a7a102 lesson); flat span arenas
  plus a content-hash-deduplicated action pool brought the middle pipeline's
  absolute peak memory to +6.8%/+2.3%/+1.1% (coremark/lua/speedtest1) over the
  pre-campaign baseline, inside a documented +10% ceiling (code).

- 2026-07-12 rationale: memprof comparisons across builds are trustworthy only
  on timeline-exact signals (span peak deltas, absolute peak bytes
  differenced); snapshot-sampled per-type maxima of short-lived allocations
  and span-boundary-relative end deltas both produced false regressions during
  this work and must not gate decisions (sourced).

## Moves

- 2026-07-12 (2c2e010e) replaced [[tentative-finalize]]: the planner's entry
  sets were intentionally tentative with rewrite observing the actual exit and
  finalizing post-hoc (filter, exit re-simulation, and post-hoc edge-repair
  derivation) — three reconciliation layers against an advisory plan; the
  exact-simulation walker computes the rows and repair actions from the same
  shared engine, deleting the reconciliation and making the plan the contract
  (code)
