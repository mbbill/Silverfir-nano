- The XIR backend allocates the three hot window registers (v0/v1/v2) itself, at
  emit time, tracking which virtual register currently occupies each slot
  (`WindowManager`); the wider register allocation lives in the backend, not in a
  separate middle stage.

- Window state is maintained as post-execution state: prepare functions choose
  slots favoring already-hot values, emit window load/store instructions, and
  return the matching handler permutation; the slot a value lands in is
  recorded as holding the operation's result without a separate tracking pass.

- Only three abstract registers are kept hot to minimize memory traffic; the
  window manager decides which VRegs stay resident and emits load/store traffic
  for the rest rather than giving every VReg a slot.

- Cold operands are batched into single win_load_2 / win_load_3 (and win_swap)
  instructions to cut the threaded-dispatch count versus one transfer per slot.

- Slot selection prefers a free slot and, when all three are occupied, evicts the
  non-protected occupant with the lowest importance score; write-back stores are
  elided for values past their last use, tracked by function-wide
  instruction-index liveness.

- Every value lives in a backing register file owned by the context as a Vec,
  indexed by value number and reached through bounds-checked indexing; the window
  load/store instructions move values between this file and the three hot slots.

- Before materializing an expression tree, a pattern matcher tries fused
  superinstruction patterns in fixed priority — constant folding first, then
  mul-add (MADD/MSUB), shift-add (SHLADD), then depth-2 binary fusions
  (BIN2_L/BIN2_R) — in greedy first-match order; a matched constant fold emits a
  single folded constant, a matched pattern emits a single fused instruction, and
  any unmatched tree falls back to linear decomposition.

- A depth-2 fusion is admitted only when the tree's Sethi-Ullman number
  (worst-case simultaneous registers, evaluating the harder subtree first) does
  not exceed the hot-window size; a fused tile is taken only when its evaluation
  fits the fixed register window without spilling.

## Facts

- 2025-10-24 (010b53d5) pitfall: the window is a write-back cache, so two
  write-back hazards must be handled before an operation — if the result will
  overwrite a slot holding a different live VReg, that occupant must be stored
  first; and if a VReg is needed in a second slot while still hot in another
  (e.g. x-x), it must be stored before being loaded into the second, or the
  duplicate occupancy loses the value (diff).

- 2025-10-24 (010b53d5) pitfall: evicting a slot must always store the current
  occupant before clearing it; an earlier eviction skipped the store when the
  occupant was in the caller's protected set, dropping that value — protection
  belongs only to slot selection (avoid choosing a slot whose value is still
  needed), never to whether eviction writes back (diff).

- 2025-10-24 (cc79153e) pitfall: for br_table the window must be flushed to the
  vreg file before the branch index is loaded into a slot; flushing after stores
  the just-loaded index back out and clears it, leaving the handler with no
  index — ordering is flush-then-load-index (diff).

- 2025-10-25 (b2109ad2) pitfall: at a conditional branch the window is flushed,
  the condition (or br_table index) loaded fresh, then the tracked state cleared;
  branch targets have other predecessors and must reload every value from its
  VReg rather than assume window contents survive the edge (diff).

- 2025-10-30 (79b3eba8) pitfall: the dead-value store-skip used program-order
  in-block deadness, so gating the end-of-block flush on it dropped values a
  successor or loop-back block still needs — the execution-order-vs-program-order
  liveness misjudgment class; the fix makes the boundary flush unconditional
  while mid-block evictions keep the deadness heuristic (diff).

- 2025-11-02 (f5414219) pitfall: the write-back hazard that stores a slot's old
  occupant before reuse must be skipped when that occupant was just loaded into
  the slot for this same operation, or a redundant load+store pair leaves the
  value where it already was (diff).

- 2025-10-24 (02b742df) pitfall: when a result VReg is written into one slot, any
  other slot still recorded as holding that same VReg must be cleared, or stale
  duplicate occupancy makes the window believe one value lives in two slots after a
  VReg is reused (diff).

- 2025-10-26 (d487c901) measurement: the score-based eviction order keeps the
  hottest VRegs resident, score = use_count * 10^loop_depth / sqrt(live_range);
  the author reports the batched-store + score-eviction + dead-store-elision +
  branch-fall-through pipeline cut window operations ~30% and bytecode 7-8% —
  data in [[window-manager.fact/window-pipeline-savings]] (author).

- 2025-09-28 (a0daf858) rationale: the lowerer tracks which value occupies each
  hot-window slot so it can skip a redundant window load when an operand already
  sits in the needed slot, and emit a register-to-register move instead of a
  backing-file reload when a value present in one slot is needed in another,
  keeping the naive per-value load/op/store-back scheme from re-reading the
  register file on every use (diff).

- 2025-09-29 (4c13a489) rationale: the per-access register load/store stopped
  null-checking the regfile and bounds-checking the index against a length on the
  dispatch path, because the index is produced by the allocator and is in range by
  construction; the check was demoted to a debug assertion and removing per-access
  error construction from the hot loop (diff).

- 2025-10-15 (0d24ab09) statement: the handler signature still passes a raw
  register-file pointer (and the memory base/size pointers), but no handler uses
  it — all register access goes through the bounds-checked Vec inside the context,
  the pointer reserved for a future optimization that did not land in this window
  (diff).

- 2025-10-08 (303bf064) rationale: the hot register window was fixed at three
  slots, reduced from four; three is the knee of a Sethi-Ullman-coverage vs
  shuffle-handler tradeoff — 2 regs cover ~60% of expression trees, 3 cover ~70-85%
  needing only three swap handlers (any 3! permutation reachable in <=2 swaps),
  while 4 regs give a marginal coverage gain but double the swap handlers and 6
  explode to fifteen; the fusion budget check was retuned from 4 to 3 to match
  (diff).

- 2025-10-14 (b493e78a) pitfall: a mul feeding the left of an add/sub must not be
  fused into MADD when the right operand is itself a multiply, or a
  difference-of-products like (x*x)-(y*y) would collapse the second multiply into
  the fused form; the matcher refuses the fusion when the RHS is also a Mul (diff).

- 2025-10-14 (b493e78a) rationale: the matched MADD/MSUB pattern is not emitted as
  a dedicated fused handler but decomposed at lowering into a separate multiply
  followed by an add/sub through the window; mapping it to a native FMA is deferred
  (no trampoline FMA handler exists yet) (diff).

- 2025-10-14 (b493e78a) pitfall: when a fused two-binary op has the same value as
  both operands (a==b), loading the second operand through the normal window-load
  path would clobber the first; the duplicate must be loaded directly into its slot
  without disturbing the other (diff).

- 2025-10-14 (b493e78a) statement: the fused two-binary lowering originally
  hardwired the i32 add/sub/mul handlers and dropped the operand type; it now
  threads the operation's value type so the same fused shape lowers to
  i32/i64/f32/f64 handlers, with unsupported (type, op) combinations raising an
  internal error naming both (diff).

## Moves

- 2025-10-13 (82b6303a) replaced [[raw-ptr-regfile]]: raw-pointer register access
  scattered unsafe blocks and unchecked add(index) through every handler; moving
  the file into Ctx as a Vec gives all register access automatic bounds checking
  with the raw-pointer handler argument retired to dead weight (diff).

- 2025-11-09 (d04cbd44) replaced by [[register-model]]: with register allocation
  moved into the middle stage, physical registers R0/R1/R2 already map
  one-to-one onto window slots v0/v1/v2, so the backend needs no WindowManager
  state and emits spills explicitly instead of allocating slots on the fly
  (diff).
