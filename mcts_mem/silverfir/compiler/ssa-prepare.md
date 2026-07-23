- The middle-end turns Semantic IR into a prepared SSA-IR whose shape makes
  later codegen cheap; it makes most of the engine's optimization decisions
  here, not by producing final code (`prepare_function`).

- It gives every local, operand spill, call payload, and return result a stable
  canonical frame slot before lowering; no later pass invents a home for a
  value under register pressure.

- Structured control is flattened into an explicit basic-block CFG with
  unreachable blocks pruned, then simplified with cheap local cleanups:
  cache-run canonicalization, jump threading, batched single-predecessor block
  merging, and unreachable-block removal.

- The live transient window is constrained to the backend's GP/FP budgets with
  explicit Spill/Fill actions, and deep Wasm stack values are published to
  canonical operand slots; the backend never solves register pressure for the
  full Wasm stack.

- Calls and runtime boundaries are made explicit SSA operations, and cache
  residency transitions are explicit ops in the IR; later legality checks read
  the boundary contract directly from the IR rather than reconstructing it.

- Constant folding evaluates pure all-constant ops, absorbs single-use constant
  definitions into operands as inline immediates, and removes dead constants
  while arithmetic is still semantic enough to fold safely.

- Sink planning annotates which value-producing ops may write their result
  directly into a cached local's register, eliding the separate local-set,
  with the physical register choice deferred to machine lowering.

- A conditional branch whose taken target wants a canonical-frame payload but
  which still carries live payload values is lowered by synthesizing an extra
  bridge block on the taken edge that stores the payload values into the target's
  operand slots before jumping.

- Prepared LIR carries a structural validator that checks entry-block range,
  per-block param shape, edge bindings against successor params, and
  single-use of each linear-SSA value (`validate_program`).

## Facts

- 2026-03-12 (f3fca0b4) rationale: the engine is intentionally a prepared
  single-pass compiler with no traditional optimization passes and no general
  lifetime-based register allocator; it gets efficiency from Wasm stack discipline
  and from frontend preparation rather than heavyweight backend analysis — the
  frontend emits explicit spill/fill against canonical operand slots so transient
  live SSA values never exceed the backend's register budget, leaving the backend
  only to fit the bounded transient set into the fixed window and swap selected
  canonical local slots into fixed local-cache registers (sourced).

- 2026-03-10 (afdf24d9) rationale: the prepared native IR is deliberately defined
  to never reintroduce stack height, TOS-rotation, interpreter-style instruction
  streams, or legacy LIR/window semantics; it is the clean boundary the backend
  must not drift back toward (code).

- 2026-03-12 (f3fca0b4) rationale: LIR lowering is split into four steps
  (semantic/planned alignment, block-boundary shaping, straight-line body, terminator)
  because stack-height reconstruction and CFG boundary shaping are semantic
  concerns that must stay separate from body and terminator lowering; lowering
  treats stack underflow or stack-shape mismatch as an explicit internal error
  rather than silently clamping, so a semantic/planning mismatch fails loudly
  (sourced).

- 2026-04-28 (a50023c5) rationale: intermediate IRs and their side tables
  (local_slot_types, value_types, const/primitive pools) are dropped the instant
  the next stage no longer reads them, never held to function end, so on a
  constrained device peak compile RAM tracks one function's largest single stage
  rather than the sum of all stages; a transient slot-only SSA lowering that
  existed solely to let the joint-plan validator check a block count was demoted to
  `#[cfg(test)]` once the semantic CFG was found to carry that count already (code).

- 2026-03-14 (e35a000a) rationale: when flattening structured control into basic
  blocks, a plain fallthrough (an op whose single successor is the next op) does
  not start a new block; only real branch targets and the targets of multi-successor
  ops are marked block leaders, so straight-line code stays in one block instead of
  splitting at every instruction boundary (code).

- 2026-03-14 (d7901c94) rationale: an `End` op no longer forces a block split — it
  merges into its enclosing block, with the live-window reconciliation it would
  have performed as a terminator run inline when an `End` appears mid-block,
  keeping block boundaries to genuine control-flow joins rather than to every
  structured-region close (code).

- 2026-03-14 (0736b065) pitfall: the inline `End`-merge reconciliation initially
  reused the branch-edge live-window publisher, which early-returns on an empty
  live window and copies only the live SSA prefix; a block result consumed as a
  `select` operand after the `End` has already been drained out of the live window
  into the spilled prefix, so the publisher published nothing and the operand slot
  was left stale — the `End` case must instead canonicalize from the spill prefix,
  publishing spill_depth-delta values and advancing spill_depth so canonicalized
  values leave the live window (code).

- 2026-03-28 (5422ef40) rationale: `program.entry` is never redirected and never
  absorbed as a successor by CFG simplification, because machine lowering
  recognizes the entry block by id and emits the parameter-load / non-parameter-zero
  entry init there; if the entry were threaded into a loop body those
  initializations would re-run every iteration. Trivial-chain resolution also stops
  at self-loops and revisited blocks with a visited set plus a step bound to
  guarantee termination (code).

- 2026-03-23 (7d2ca7fb) rationale: a constant may be absorbed into an operand only
  when it has no use in the block terminator (edge bindings, branch/table
  conditions): linear SSA guarantees exactly one use in the op stream, so
  'no terminator use' equals 'total uses == 1', and terminator-used values still
  need a real register; the spill planner already budgeted a transient for the
  absorbed constant, so a backend that cannot encode the immediate natively can
  always materialize it into that guaranteed scratch register (code).

- 2026-03-23 (1291c814) rationale: when every operand of a pure op is constant the
  fold pass evaluates it at compile time and replaces it with a const definition,
  recording the new constant so chains collapse in one forward pass; trapping ops
  are never folded when they would trap (div-by-zero, MIN/-1 overflow, out-of-range
  trapping truncation return None to preserve the runtime trap) and folded float
  results are NaN-canonicalized to the Wasm canonical NaN bit-patterns (code).

- 2026-03-25 (6179ba88) pitfall: compile-time folding of `f32.sqrt`/`f64.sqrt` was
  removed because the std/libm sqrt intrinsics are unavailable in the no_std engine
  (the other float ops fold via in-tree soft-float helpers, but no software sqrt
  exists); the folder declines to fold sqrt and leaves it to runtime (code).

- 2026-03-27 (3a9284d1) pitfall: on 32-bit GP targets, folding a constant into an
  operand is disabled for all `i64` ops (and Select) because the gp32 pair lowering
  resolves operands via `unwrap_value()`, which panics on an inline-const operand;
  `i64` const operands stay materialized into a transient until the pair lowering
  learns to consume const operands (code).

- 2026-03-19 (c86da6a9) rationale: the LIR cached-slot type check is a role-directed
  subtype-compatibility test rather than exact storage-class equality — a StoreSlot
  source must be a subtype-compatible producer for the slot's cached type and a
  LoadSlot destination compatible the other way — so a concrete funcref stored into
  an i32-typed GP-word cache slot is rejected while subtype-compatible reference
  values that legitimately share a GP-word slot are accepted (code).

- 2026-03-26 (98de6d7b) rationale: sink planning is kept register-agnostic in the
  middle-end — it only records the legal opportunity to fold a value's production
  into a local's home, deferring to machine lowering whether to actually exploit
  it (which depends on whether the target local is resident in a cache register) —
  so the SSA pass needs no knowledge of physical register assignment (code).

- 2026-03-13 (4ae8509d) pitfall: memory.init and table.init each carry two
  immediates and the prepare lowering had them crossed — the data/mem index and
  the elem/table index were read from the wrong immediate slot, swapping the two
  segment indices; the spec order is (segment, memory/table) (code).

- 2026-03-13 (4ae8509d) pitfall: the operand slot for a taken branch's payload
  was computed from the static stack_drop alone, not from the actual stack
  height; the correct base is current_height - stack_drop - arity, so the payload
  must be read relative to where the live values actually sit on the operand
  stack (code).

- 2026-03-27 (cb1d3151) pitfall: the End prefix used to eagerly fill all of a
  structured block's results into transient registers, which can exceed the GP
  transient budget when a block produces multiple i64 values on 32-bit targets (a
  block returning 3 i64 values is 6 GP units); the fix stops filling at End —
  block results stay on the operand stack / in frame slots and the next
  instruction's own fill reloads exactly what it needs — while Else still fills
  all results because the Else branch entry needs every value visible (code).

- 2026-04-09 (b8f8fca7) pitfall: the single-predecessor merge pass (merge a block
  into its sole predecessor when that predecessor ends in an unconditional goto)
  must never merge the program entry block into a predecessor: an unreachable
  block whose only terminator is a goto into entry would otherwise absorb the
  entry block and move the entry's id, which machine lowering relies on to emit
  the parameter-load / non-parameter-zero entry init (code).

- 2026-06-20 correction: the 2026-03-27 (3a9284d1) pitfall no longer
  holds — gp32 `i64` (and Select) const operands are now absorbed on all targets,
  and the pair lowering consumes Const operands by splitting them into lo/hi
  `Imm64` (use_i64_operand_pair), so there is no `unwrap_value()` panic; const
  folding into operands is no longer disabled for `i64` ops on 32-bit GP targets
  (code).

- 2026-07-12 (58927160) statement: the slot-only SSA lowering the 2026-04-28
  (a50023c5) entry describes as demoted to `#[cfg(test)]` was fully deleted,
  together with its now-vacuous block-count plan check; the test suite anchors
  on the semantic CFG's block structure instead (code).

- 2026-07-22 (4b801ebb) measurement: single-predecessor cleanup previously
  merged one successor, physically removed it from every per-block vector,
  remapped every CFG target, and restarted the fixed point. Predecessor counts
  for the remaining live graph do not change when a predecessor absorbs its
  sole successor: only the source identity of the successor's outgoing edges
  changes. The pass now uses one predecessor-count table to absorb complete
  goto chains into tombstoned slots, then compacts and renumbers once. Serial
  Pulldown startup fell from 123.66 ms to 111.8-114.1 ms, cleanup from 12.46%
  to 4.14% inclusive, and `remove_blocks` from 6.38% to 1.63% (sourced).

- 2026-07-22 rejected: removing the per-block vector `shrink_to_fit` calls
  during SSA rewrite appeared to avoid work duplicated by final prepared-SSA
  compaction after cleanup and optimization, but serial bz2 measured 44.583 ms
  versus 44.786 and 44.495 ms for the accepted parent, inside run noise. The
  experiment was reverted; the early shrink's transient-memory bound was not
  exchanged for an unmeasured compile-time change (sourced).

## Moves

- 2026-03-12 (2ea0bb68) replaced [[two-stage-planning]]: the planning layer
  previously emitted its own intermediate planned-op IR with a rotating-TOS
  window and a hot-local register-class plan, then lowered that to LIR — two IRs
  for one preparation job; collapsing it so prepare_function produces prepared
  LIR directly removes the redundant planned-op IR and the rotating-TOS
  representation, and replaces the hot-local register-class plan with pure
  local-cache preference analysis (ranking hints, not storage kinds), since
  canonical local identity must stay slot-based and the cache swap is execution
  policy decided below LIR (code)

- 2026-03-12 (e7327ee1) dropped: the separate group abstraction layered on top of
  prepared-LIR blocks (grouping metadata and grouping mode/policy): prepared-LIR
  blocks are made the only execution-region concept above MachineIR (maximal mode
  splits blocks only at real control boundaries, and any future fusion constraint
  splits blocks earlier), so a group layer on top of blocks is redundant (code).
