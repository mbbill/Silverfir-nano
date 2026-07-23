- SSA-IR is lowered to MachineIR in a single local pass that never runs a
  heavyweight graph-coloring register allocator; it relies on the middle-end
  having already bounded transient pressure and assigned canonical homes
  (`lower_module`).

- The lowering is one internal pass over each block — not a new public IR layer —
  that both expands VM-flavored ops into machine-shaped code and does the trivial
  fixed-budget register assignment: it maps the bounded transient SSA set to the
  transient register bank and tracks per-value remaining-use counts to free dead
  transients.

- One-pass allocation reuses dead input registers for results: an arithmetic
  result can stay in an input register whose value dies at that op rather than
  taking a fresh transient.

- Cached `local.get` becomes a register alias rather than a load (the still-live
  aliases of a cache register are copied out into transients only just before that
  register is overwritten), and a sink-approved result whose source transient just
  died is emitted directly into the local's cache register with no move.

- A call, boundary, or return reached with a non-empty live transient set is hard
  rejected, forcing the middle-end to publish all live SSA to slots before every
  boundary; this is the invariant that lets lowering skip stack reconstruction.

- Pointer-width fields and pointer-tagging constants are selected by target pointer
  size, letting the shared pipeline support 32-bit GP targets without per-op
  special-casing (`machine_ptr_width`).

- A null reference is encoded as the all-ones sentinel (usize::MAX); ref.null lowers
  to a usize::MAX constant and ref.is_null to an unsigned-equality compare against it
  (`RefHandle`).

## Facts

- 2026-07-22 measurement: two fixed-budget allocator predicates still scanned
  every cached-local binding and unpublished incoming parameter for each
  candidate register. Mirroring those non-linear reservations in one compact
  per-dynamic-register count preserved cache/parameter ownership (including a
  transient overlapping transfer) while changing the queries to indexed
  lookups. In the verification profile, `dynamic_reg_available` fell from
  1.80% to 0.19% self-time and `is_linear_value_reg` from 1.60% to 0.09%
  (sourced).

- 2026-07-22 measurement: controlled serial bz2 indexed/parent/indexed means
  were 42.664, 43.965, and 42.624 ms, a repeatable roughly 3.0% reduction.
  This is a representation win inside the intentionally simple fixed-budget
  allocator, not the introduction of general register allocation (sourced).

- 2026-07-22 measurement: MachineIR block-parameter lowering used an owned
  temporary vector for every scalar value (and every GP32 i64 pair), attached
  ownership metadata, and immediately drained it into the block's destination
  vector. Appending those parameters directly preserved the scalar/pair and
  owner policy while reducing `append_entry_cache_params` from 7.18% to 1.93%
  of `lower_function`. Controlled serial bz2 direct/parent/direct means were
  43.829, 44.416, and 43.856 ms, a repeatable 1.26-1.32% reduction (sourced).

- 2026-07-22 rejected: skipping GP/FP cache-layout matrices and dominator
  traversals for functions with cached cells in only one register bank made
  serial Pulldown and SpiderMonkey effectively flat. Bz2 and CoreMark showed
  only roughly 1% signals, too small for the 200-line control-flow expansion;
  the experiment and its test were fully reverted rather than retaining
  unmeasured bank-specialized complexity (sourced).

- 2026-07-22 rejected: borrowing cache-layout rows directly during incoming-edge
  scoring instead of cloning the current and predecessor rows changed serial
  bz2 from about 51.91 to 51.83 ms, inside run-to-run noise. The clones account
  for only a small fraction of `compute_block_entry_cache_params`; the
  experiment was discarded rather than retained as unmeasured complexity
  (sourced).

- 2026-07-22 rejected: `incoming_param_owns_reg` scanned the per-cell
  parameter-state vector for every candidate dynamic register, so an
  O(1) register-indexed ownership bitmap was tested. All 147 machine-focused
  release tests passed, but three serial bz2 means were 47.038, 46.871, and
  46.566 ms against the accepted 46.757 ms baseline, with no statistically
  significant comparison. The bitmap and its duplicated state maintenance were
  reverted; the theoretical local-count scan is not a demonstrated startup
  bottleneck on bz2 (sourced).

- 2026-03-12 (7bb4d7dc) rationale: the step between prepared LIR and MachineIR is
  one internal lowering pass, not a new public IR layer — it expands VM-flavored ops
  into machine-shaped code and does the trivial fixed-budget register assignment the
  backend depends on, so no general allocator is needed (sourced).

- 2026-03-17 (6612c624) statement: prepared LIR is linear SSA — within a block every
  value is used exactly once (a debug assertion enforces it); extra uses arise only
  from operand-stack spill and from edge bindings for values live across block
  boundaries, and this single-use property is what lets lowering free an op's input
  registers immediately after the op without liveness tracking (code).

- 2026-03-29 (f1099cab) pitfall: the block-parameter liveness computation enumerated
  each instruction's defined registers through an accessor returning a single
  Option<MachineReg>, silently dropping the high half of i64 pair ops; on 32-bit GP
  targets this miscomputed defined-before-use and corrupted block-param plumbing — any
  pass needing an instruction's full def set must use the for-each form that visits
  both halves (code).

- 2026-07-12 (58927160) statement: the def-set enumeration behind the f1099cab and
  22c1c30f pair-corruption pitfalls is now a single canonical visitor on the
  instruction kind; the scalar single-register accessors and the three parallel
  per-pass copies of the match were deleted, so the single-vs-pair def hazard can
  no longer drift between passes (code).

- 2026-03-13 (4ae8509d) pitfall: select takes (val1, val2, cond) and returns val1
  when cond is nonzero; the machine lowering had on_true and on_false bound to the
  wrong operands, inverting the selected value (code).

- 2026-04-27 (65ccf38f) measurement: aliasing the cached i64 lo/hi pair directly
  (instead of snapshotting it into fresh registers on every read) and skipping
  identity cache self-moves dropped the hot MachineIR move count on RV32 Mandelbrot
  from 103 to 25 and restored ESP32-C6 Mandelbrot to 29 fps (sourced).

## Moves

- 2026-04-09 (c329abab) replaced [[whole-module-borrowed-ssa]]: a borrowed whole-module SSA slice ties every function's prepared SSA to one lifetime so none can be freed until lowering finishes; taking ownership of the lowering inputs lets each function's SSA (and the semantic IR, now taken and dropped) be released as soon as it is lowered, cutting peak compile-time memory (code).
