- Fusion patterns are detected by a matcher trait whose default implementation
  calls a fixed sequence of per-pattern matcher functions (madd, then shladd)
  inline.

- Each pattern's match-and-emit logic lives in one shared module; adding or
  disabling a pattern means editing that dispatch sequence.

- Pattern priority is the written call order, not a declared node-count, leaving
  overlapping patterns not guaranteed to match maximal-munch.

## Facts

- 2025-10-11 (70085428) pitfall: the fusion budget passed at the single
  match_tree call site is a hard-coded literal 4 ("hot window capacity = 4
  registers"), but the hot window is only three slots wide since the fourth
  slot was dropped, so the max_live_regs bound over-states the real window by
  one register and admits fusion candidates whose live inputs cannot all fit
  the three hot slots — a stale budget literal left behind by the removal
  (diff).

- 2025-10-12 (1bdc24fd) statement: the store-with-binop (ST_BIN) fusion is
  attempted only when the store targets the default memory (memidx == 0);
  stores to any non-default memory skip fusion and emit a plain materialized
  store carrying the memory index (diff).

- 2025-10-18 (10c5e2c9) rationale: a "fusion hints" framing — emit base
  instructions plus optional FusionHint metadata and let a backend opt in —
  was explored and rejected, because pattern matching must run while the
  `ExprTree` still exists (it sees the whole multi-level pattern and controls
  operand materialization order); once materialized to linear SSA the tree
  structure is lost and cannot be reconstructed cheaply from def-use chains, so
  hint-based post-hoc fusion buys debuggability knobs at the cost of the
  matching context; direct tree-based emission follows LLVM SelectionDAG /
  Cranelift ISLE (diff).

- 2025-10-18 (10c5e2c9) rationale: only narrow, hardware-backed patterns are
  specialized — Madd/Msub (FMA on x86 FMA3 / ARM NEON) and Shladd (LEA / ARM
  shifted operand) — and the generic two-level Bin2L/Bin2R fusions for
  arbitrary op pairs were removed, because covering them would need ~625
  operator-pair combinations of which most map to no hardware instruction
  (diff).

- 2025-10-25 (b2109ad2) pitfall: multiply-add fusion must be restricted to
  integer types — a fused multiply-add rounds once whereas separate multiply
  then add round twice, so fusing f32/f64 mul+add changes the IEEE-754 result
  (diff).

- 2025-10-26 (e3b95723) rationale: which patterns to fuse is chosen
  data-driven, not by intuition — real applications are profiled to find hot
  executed instruction sequences and only proven-beneficial patterns are
  implemented; benefit is measured by static instruction-count reduction (not
  wall time: deterministic, architecture-agnostic, directly attributable), and
  profiling is done at the XIR level so it reflects the actual post-optimization
  executed stream (author).

- 2025-10-26 (1b35c94b) pitfall: a fusion frontend must only emit patterns the
  backend has handlers for — try_madd fused integer mul+add but the XIR backend
  had madd handlers only for f32/f64 and codegen errored on any non-float type
  with no type filter in between, so integer Madd flowed all the way to codegen
  and failed; the matcher was restricted to F32/F64 to match backend handler
  availability (diff).

- 2025-10-26 (f7ad56a9) conformance: multiply-add (FMA / Madd) fusion is
  disabled entirely, leaving Shladd the only active fusion — the WebAssembly
  spec requires each operation to produce exactly its as-written result, and
  FMA's single rounding differs observably from separate mul+add (caught by
  float_exprs.wast); re-enabling would require an explicit fast-math opt-in
  (diff).

## Moves

- 2025-11-26 (5cc62f37) replaced by [[tree-time-fusion]]: the hard-coded matcher tried patterns in written order with each pattern's logic inlined into one dispatch function, so it could neither guarantee maximal munch when patterns overlap nor add or disable a pattern without editing the core; a registry of Pattern records (matcher fn, emitter fn, priority, nodes-covered, enabled flag) sorted by priority makes patterns declarative data, tried largest-first automatically, and individually toggleable (diff).
