- local.get and param access push a deferred LocalRef(idx) / Param(idx) leaf;
  the index is resolved against the value map (locals) or the parameter list
  only when the tree is materialized at a barrier.

- Materialization first rewrites every LocalRef/Param leaf to a materialized
  SSA value before fusion matching and linear lowering; a LocalRef/Param leaf
  surviving into linear lowering is an internal error.

## Moves

- 2025-09-30 (7accd393) replaced [[eager-value-stack]]: emitting each Wasm operation immediately as a linear SSA instruction off an operand stack fixed the computation shape before it could be inspected, leaving no tree to match superinstruction patterns against; accumulating pure operations as expression trees on an expression stack and materializing them only at barriers exposes the tree shape to fusion (diff).

- 2025-10-29 (146ff7a0) replaced by [[value-only-trees]]: a deferred LocalRef/Param leaf resolves against value_map only at materialization, so an intervening local.set rebinds the local and the unmaterialized leaf reads the wrong SSA version; capturing the current SSA value at local.get/param time pins the value WebAssembly semantics require (diff).
