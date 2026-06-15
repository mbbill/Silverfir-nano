- Pure value-producing operations are not lowered to SSA at decode time;
  they accumulate as expression trees on an expression stack (`ExprTree`) and
  are materialized to linear SSA only when a barrier forces it.

- local.get and parameter access capture the current SSA value out of the
  value map eagerly at decode time; a later local.set on the same index
  cannot rebind a value already taken into a tree.

- Side-effecting barriers (store, global.set, local.set, calls) are
  themselves `ExprTree` variants, and every operation is materialized through
  one unified materialization entry point that performs a pure linear
  decomposition; fusion happens later as an SSA-level pass, not at tree time.

## Facts

- 2025-10-25 (2361e9cd) pitfall: the type recorded on a binary expression-tree
  node is the operand type, not the result type, because codegen selects the
  typed handler (e.g. i64 vs i32 eq) from it; for comparisons the decoder must
  record the operand width (i64/f32/f64) even though the comparison yields i32,
  or the wrong-width handler reads only the low bits (diff).

- 2025-10-31 (5bfd8ace) rationale: building expression trees and materializing
  to linear SSA only at barriers delivers, during construction at no extra
  pass, what immediate linearization would need separate passes for — trees
  left unmaterialized after an unconditional branch vanish (free DCE), fusion
  patterns are matched on tree shape before linearization, and materialization
  deferred to the latest safe point leaves room for scheduling (diff).

## Moves

- 2025-11-28 (63fc659a) replaced [[value-only-trees]]: side-effecting barriers (store, global.set, local.set) were not expression-tree nodes, so each barrier's operand sub-trees were materialized in isolation by scattered per-barrier code (memory.rs on_store, calls.rs): the address and value sub-trees still ran through the fusion registry, but the registry was never offered the barrier together with its operands, so it could never fuse a barrier with a producer (e.g. a store with a fused-address op); making barriers ExprTree variants and routing every operation through one materialize_tree_full lets the same fusion registry match patterns on barrier nodes too (diff).
