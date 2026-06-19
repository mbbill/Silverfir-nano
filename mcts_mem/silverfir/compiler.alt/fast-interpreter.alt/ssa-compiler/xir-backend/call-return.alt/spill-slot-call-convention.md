- At a call site the compiler materializes every argument into its spill slot,
  and the call/return handlers read all arguments and write all results through
  those spill slots; no argument or result is passed in an abstract register.

- A function's parameters are required at entry to occupy the frame's low spill
  slots (param i in spill slot i), validated by the param-slot contract.

- The return instruction encodes the spill-slot indices holding the results, and
  the caller reads results from those slots.

## Facts

- 2025-11-12 (ed23ac48) rationale: results are not forced into fixed slots — each
  Return instruction carries metadata listing the spill-slot indices holding its
  result values, so a value can stay in whatever slot it was computed in and the
  caller reads the encoded slot list (code).

## Moves

- 2026-02-13 (7f97c463) replaced by [[call-return]]: passing arguments and results
  through spill slots forced a memory copy on every call and return even for the
  common small-arity case; the register convention places the first 8 args and 8
  results in the abstract registers v0..v7 so they ride in CPU registers across
  the threaded tail-call chain with no per-call copy, falling back to slots only
  for the >8 overflow (code).
