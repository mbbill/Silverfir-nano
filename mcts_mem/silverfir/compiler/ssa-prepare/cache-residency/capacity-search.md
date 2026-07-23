- Residency selection is solved in two stages: an unconstrained per-slot tree
  DP over the region tree computes each candidate local's best value in every
  region conditioned on parent-region residency, then a top-down per-region
  feasibility knapsack projects those per-slot optima onto the region's
  register capacity, recursing parent-to-child with continuity-conditioned
  values (`extract_feasible_states`).

- Capacity competition is arbitrated only in the knapsack; the per-slot DP
  never sees a capacity term.

- The knapsack considers items in descending resident-subtree-potential order
  (each slot's DP value when forced resident in the region), with slot index
  as the deterministic tiebreaker.

- Feasible extraction stores its complete take/not-take backtracking choices in
  one dense matrix and stores its floating-point DP values in two rolling rows
  (`extract_feasible_states`).

## Facts

- 2026-07-12 measurement: the pricing ablation that motivated the deletion —
  removing price iterations shifted the 9-module arm64 corpus by a net −9
  native instructions (coremark +24, sha256 +8, bzip2/lz4 wins), while on
  armv7 (8 dynamic GP lanes) the priced solver ran 3.99% slower on qemu
  coremark than the price-free one (2,458±27 vs 2,556±6 it/s) with code size
  within 0.03%; the pricing had never been validated in the register-scarce
  regime it was designed for (sourced).

- 2026-07-12 (748c8416) pitfall: deleting the prices exposed that feasible
  extraction committed ancestor continuity arbitrarily on value ties — a slot
  whose benefit sits deep in the subtree shows the same resident-vs-absent
  gain at an ancestor as a weaker sibling, because the deep benefit appears in
  both branches of the difference — so the root knapsack could commit to the
  weaker local and children rationally followed; price noise had perturbed
  exactly these ties and masked the flaw (code).

- 2026-07-12 (748c8416) measurement: final gates for the deletion plus
  tie-break ordering — per-module native counts vs the pre-campaign baseline:
  bzip2 −20, lz4 −40, coremark +1, sha256 +8, sqlite +53, rest 0 (net +2 of
  1.07M); arm64 coremark 35,891±373 vs 35,840±457 reference; armv7 qemu
  coremark 2,528±34, +2.9% over the shipped priced solver's 2,458±27 and level
  with the plain ablation's 2,556±6 within run noise, with smaller code
  (sourced).

- 2026-07-12 rationale: a per-region adaptive solver strictly dominates a
  global static set in expressiveness — every global set is one point in its
  search space — so an adaptive solver losing to static indicts the search,
  not the objective; the relaxation was the failed search component while
  every objective term survived ablation, leaving exact joint search over
  residency and capacity as the unexplored ceiling (sourced).

- 2026-07-12 rationale: solver changes must be validated in the
  register-scarce regime (armv7's 8 GP lanes); arm64's lane surplus lets most
  functions cache every hot local, so capacity-arbitration differences are
  nearly invisible there — the pricing looked neutral on arm64 for months
  while costing 4% on armv7 (sourced).

- 2026-07-22 measurement: a serial bz2 profile put feasible extraction at
  7.35% inclusive. Replacing its full value matrix with two rolling rows
  produced controlled long-sample B/A/B means of 45.421 ms (rolling), 46.548
  ms (exact parent), and 44.786 ms (rolling), a repeatable 2.4-3.8% reduction.
  The existing exact-selection test and all workspace release tests passed;
  the resident decisions, full boolean backtracking matrix, item ordering, and
  strict tie rule are unchanged. The verification profile reduced feasible
  extraction to 2.93% inclusive and the enclosing joint-plan builder from
  11.58% to 7.32% (sourced).

- 2026-07-23 rejected: replacing recursive top-down extraction with a
  parent-indexed iterative region walk removed one `selected[parent].clone()`
  per region edge and bank, but the cloned rows are bit-packed `Vec<bool>` and
  were not the remaining solver cost. FFmpeg kept `solve_bank` essentially flat
  at 95 versus 96 inclusive samples; exact-parent bz2 ABBA point estimates were
  neutral (64.16 versus 64.39 ms on pair averages). Making the now-inlineable
  walk iterative also grew fat-LTO text by 4,708 bytes; forcing a separate code
  boundary grew it by 5,004 bytes. The experiment was fully reverted. Focus
  future solver work on the per-region sort/knapsack work, not parent-row
  snapshots ([[compiler.fact/startup-campaign-2026-07-22]]) (sourced).

## Moves

- 2026-07-12 (748c8416) replaced [[lagrangian-pricing]]: the price iterations
  were statically neutral on arm64 (net −9 native instructions across the
  9-module corpus) and actively harmful in the register-scarce regime they
  were designed for (armv7 qemu coremark −3.99%), and their noise masked a
  tie-commitment flaw in feasible extraction; the price-free DP with
  potential-ordered knapsack extraction is smaller and no worse on every
  measured target (sourced)
