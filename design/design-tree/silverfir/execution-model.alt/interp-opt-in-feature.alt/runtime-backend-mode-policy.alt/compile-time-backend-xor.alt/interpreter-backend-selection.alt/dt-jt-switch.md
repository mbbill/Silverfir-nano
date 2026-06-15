- Function evaluation dispatches through one entry that, per process, selects
  between the match-based dt backend and the handler-pointer-table jt backend
  via an SF_INTERP-seeded atomic switch, defaulting to jt and committing to no
  single engine.

## Moves

- 2025-08-08 (d33f2413) removed: the Dt/Jt runtime switch (an SF_INTERP-seeded
  AtomicU8 selecting dt::eval vs jt::eval per process) was deleted together with
  the jt backend it selected, reverting function evaluation to dt::eval
  unconditionally; jt's faster handler-table dispatch bought nothing on the
  in-place interpreter (which has an inherent ceiling), so there was no second
  in-place engine left to choose between — the dispatch idea moved on to the
  ssa/xir backend instead (author).
