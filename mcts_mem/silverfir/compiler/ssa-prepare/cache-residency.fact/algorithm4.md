commit: 41a02194

Recovered design doc (docs/ALGORITHM4.md, merged from middle/ALGORITHM4.md +
LANE_MAPPING.md), quoted from the author 2026-04-07.

The prior residency planners ALGORITHM2 (one global resident set) and ALGORITHM3
(root set plus per-loop override) treat residency stability as a structural
constraint, so they ignore the cost of ensure/drop transitions at region edges
and produce massive boundary churn.

ALGORITHM4 instead models residency as a region-tree DP that minimizes a single
total cost combining per-region access benefit, call tax, and per-edge transition
cost under per-region register-unit capacity; the special cases of one global set
and root+loop-override fall out of the same objective rather than being
hand-coded.

The shipped extraction does a recursive per-region capacity-constrained
extraction rather than the original Step-5 marginal-value projection.
