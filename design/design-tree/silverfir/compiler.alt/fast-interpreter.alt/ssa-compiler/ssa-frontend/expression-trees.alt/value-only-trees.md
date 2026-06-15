- The expression tree represents only value-producing computation;
  side-effecting barriers (store, global.set, local.set, calls) are not tree
  nodes.

- Fusion is registry-driven: materialization runs the data-driven pattern
  registry over generic value-producing tree nodes and emits fused SSA
  instructions (Madd, Shladd) when a pattern matches; there are no hard-baked
  fused tree variants.

- Barrier handling is split: the decoder decides when a barrier forces
  materialization and per-barrier modules (memory, calls) materialize the
  barrier's operand sub-trees through the shared materialization path (each
  operand tree sees the fusion registry), but the barrier node itself is
  outside the tree, leaving no pattern able to fuse a barrier with its operands.

## Moves

- 2025-10-29 (146ff7a0) replaced [[lazy-leaf-trees]]: a deferred LocalRef/Param leaf resolves against value_map only at materialization, so an intervening local.set rebinds the local and the unmaterialized leaf reads the wrong SSA version; capturing the current SSA value at local.get/param time pins the value WebAssembly semantics require (diff).

- 2025-11-28 (63fc659a) replaced by [[expression-trees]]: side-effecting barriers (store, global.set, local.set) were not expression-tree nodes, so each barrier's operand sub-trees were materialized in isolation by scattered per-barrier code (memory.rs on_store, calls.rs): the address and value sub-trees still ran through the fusion registry, but the registry was never offered the barrier together with its operands, so it could never fuse a barrier with a producer (e.g. a store with a fused-address op); making barriers ExprTree variants and routing every operation through one materialize_tree_full lets the same fusion registry match patterns on barrier nodes too (diff).
