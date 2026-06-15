- A process-global AtomicBool chooses between exactly two interpreter backends,
  read at each function evaluation: the fast stack-based interpreter by default
  and the classic in-place interpreter when the baseline flag is set.

- The switch exposes only a boolean set/get; there is no representation for a
  third backend.

## Moves

- 2025-10-01 (04122214) replaced by [[interpreter-backend-selection]]: a single AtomicBool
  could only choose between two interpreters (fast vs the classic inplace
  baseline) and could not name a third; a u8-tagged backend enum lets function
  evaluation select among the classic, fast, and new SSA backends per process,
  and makes the SSA backend the default with fast/classic as explicit overrides
  (diff).
