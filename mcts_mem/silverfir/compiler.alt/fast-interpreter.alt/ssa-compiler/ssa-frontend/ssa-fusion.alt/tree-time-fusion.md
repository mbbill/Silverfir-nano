- Fusion runs during expression-tree materialization: before an `ExprTree` is
  lowered to linear SSA, it is offered to a priority-ordered registry of
  fusion patterns, and the highest-priority matching pattern emits a single
  fused instruction instead of the decomposed form.

- Each fusion pattern is a matcher/emitter pair registered with a priority and
  a node-coverage count; the registry is sorted by priority with larger patterns
  winning (maximal munch), and a tree that matches no pattern falls back to linear
  decomposition.

## Moves

- 2025-11-26 (5cc62f37) replaced [[hardcoded-matcher]]: the hard-coded matcher tried patterns in written order with each pattern's logic inlined into one dispatch function, so it could neither guarantee maximal munch when patterns overlap nor add or disable a pattern without editing the core; a registry of Pattern records (matcher fn, emitter fn, priority, nodes-covered, enabled flag) sorted by priority makes patterns declarative data, tried largest-first automatically, and individually toggleable (code).

- 2025-11-28 (1fae526c) replaced by [[ssa-fusion]]: matching fusion patterns on the expression tree during materialization could only see a single unmaterialized tree and could not fuse across barriers such as local.tee; running fusion as a pass over completed SSA lets it see through those barriers and match patterns whose operands span barrier boundaries (code).
