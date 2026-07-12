- Machine lowering derives the preserved-lane preference itself: an
  inter-block cache-liveness fixpoint over the lowered blocks, a per-block
  backward needed-after sweep, and a forward count of local-JIT-call crossings
  per cached local, promoted by a whole-function threshold and broadcast to
  every block (`compute_local_call_cache_preferences`).

- Machine lowering re-derives each entry row's Ensure-versus-Reserve
  requirement by scanning the block's ops at lowering time, and classifies
  local-JIT calls with its own module-level inputs.

## Moves

- 2026-07-12 (30aac662) replaced by [[lane-assignment]]: the preserved-lane
  preference and the entry Ensure-versus-Reserve requirement are pure
  functions of the final SSA and module facts, so the middle computes them
  once over the final program and byte-identical output proves equivalence;
  the machine's liveness dataflow and per-block requirement re-scan are
  deleted, and only physical placement still needs machine context (code)
