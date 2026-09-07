- Natural-loop caching carries repeatedly read complete native frame words through proven loop entries and backedges using available preserved dynamic lanes. Canonical frame publication, alias barriers and lowering-reserved lanes remain constraints.

- Cheap impossibility checks precede predecessor and slot scratch allocation. Adding a carried parameter without changing instructions does not schedule another block-local optimization pass.

## Facts

- 2026-09-06 (3388ae67) measurement: prioritizing single-read mutable frame words over exit-only reuse removed a load but reduced CoreMark throughput by 0.5610% on AMD 9V74 and 0.4170% on Intel 8573C; all four x64 startup points regressed. Runs 34075452274 and 34075452287 do not support broadening eligibility (sourced).

- 2026-09-06 (8f69395b) measurement: general GP reuse removed three CoreMark moves but measured -0.0169% and -0.2023% on two AMD 9V74 draws, while all eight startup points regressed. Runs 34077414285 and 34077414291 do not support adopting the extra liveness work (sourced).

- 2026-09-06 (937ed3e0) measurement: combining frame publication with scoped GP reuse gained 0.5368% and 0.5432% CoreMark in two independent 12-pair AMD 7763 draws. Full20 then reproduced sort losses of 33.10% and 25.96%, with geometric losses of 1.7983% and 1.4061%. All 60 rows are in docs/x64-campaign-2026-09-06/loop-cache-register-full20-results.json; a tiny CoreMark gain cannot justify this candidate (sourced).

## Moves

- 2026-09-06 (937ed3e0) replaced [[compiler/machine-peephole/loop-frame-cache.alt/publication-and-gp-reuse]]: Two native full20 draws reproduced severe sort regressions that overwhelm the independently confirmed roughly 0.54% CoreMark gain; retain the prior conservative cache and register policy (sourced).
