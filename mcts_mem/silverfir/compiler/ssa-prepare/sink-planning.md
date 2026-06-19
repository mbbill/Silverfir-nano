- A backward per-block sink planner annotates each value-producing op that may
  write its result directly into a LocalSet target's cache home, marking the
  subsequent LocalSet elidable; the producer is pre-mapped into the local's cache
  register and the local-set move is dropped when the source already resides there
  (`plan_sinks`).

## Moves

- 2026-03-26 (98de6d7b) replaced [[reactive-cache-coalescer]]: the reactive
  coalescer could only patch the single immediately-preceding instruction and
  could not reason about semantic local versions or cross-instruction legality,
  so sink-legality analysis was lifted into the middle-end where the version is
  known and a producer can be proactively pre-mapped into the local's cache home
  (code).
