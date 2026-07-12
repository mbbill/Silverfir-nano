- A lowered function's result slots are the low vreg indices 0..num_results; the
  caller reads results from those slots.

## Moves

- 2025-10-23 (a7a48d92) replaced by [[lowering]]: the low-index assumption could
  not express results landing in slots chosen by register allocation, so it
  mis-read results whenever a result vreg was not in 0..num_results; deriving the
  slots from the Return instruction names the actual result vregs (code).
