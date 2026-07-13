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

## Moves

- 2026-07-12 (748c8416) replaced [[lagrangian-pricing]]: the price iterations
  were statically neutral on arm64 (net −9 native instructions across the
  9-module corpus) and actively harmful in the register-scarce regime they
  were designed for (armv7 qemu coremark −3.99%), and their noise masked a
  tie-commitment flaw in feasible extraction; the price-free DP with
  potential-ordered knapsack extraction is smaller and no worse on every
  measured target (sourced)
