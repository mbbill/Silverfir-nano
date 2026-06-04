---
status: abandoned
---
# Stability as a structural constraint (ALGORITHM2 / ALGORITHM3)

Residency planners that treat which-locals-stay-resident as a constraint problem.
ALGORITHM2 picks one global resident set for the whole function. ALGORITHM3 adds
a per-loop override on top of a root set. Both minimize per-block frame-access
cost and treat whole-function stability (and loop-specific overrides) as hard
structural rules.

## In practice

While in force these entailed:

Must:
- ALGORITHM2: select a single resident set held for the entire function.
- ALGORITHM3: select a root resident set plus an explicit per-loop override on
  top of it.
- Minimize per-block frame-access cost when choosing the resident set(s).

Must not:
- Account for transition cost at control-flow edges when choosing residency (the
  per-block-cheap plan forces locals to be repeatedly ensured and dropped at
  edges where blocks disagree on the resident set).
