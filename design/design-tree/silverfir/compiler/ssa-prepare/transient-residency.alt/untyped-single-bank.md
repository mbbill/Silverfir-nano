- Prepared execution state tracks one untyped live suffix of the Wasm stack
  (`stack_height`, `spill_depth`, and a single `live: Vec<LirValue>`) with no
  per-value type; a single `tos_limit` bounds the whole live transient window and
  local-cache preference analysis ranks one list of canonical local slots
  regardless of value type (`tos_limit`).

## Moves

- 2026-03-15 (5828c3c2) replaced by [[transient-residency]]: the untyped
  single-bank live window could not keep float values in FP registers: floats
  lost their type at slot reloads and were recreated in GP transients; tracking
  each value's exact Wasm type and budgeting GP and FP banks separately lets
  floats stay FP end-to-end and removes representation-driven GP/FP churn (diff)
