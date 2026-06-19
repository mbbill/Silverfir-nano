- The top values of the operand stack are held in a fixed set of handler-passed
  CPU registers; which physical register holds a given logical stack position
  is a compile-time function of that position's depth, with no register state
  tracked at run time (`StackTracker`).

- The number of top-of-stack registers is a single build-time constant
  (`TOS_REGISTER_COUNT`, currently four) from which every Rust and C generator
  derives variant names, register names, the per-handler lookup arrays, the
  variant-index formula, and the cyclic register-assignment; the register
  assignment uses a power-of-two mask when the count is a power of two and falls
  back to modulo otherwise.

- The build-time stack tracker maintains height minus spill-depth at or below
  the register count: pushing past the window spills the oldest cached register
  to its operand-stack slot first, and operands are reloaded on demand; the
  frame slot for spill/fill is computed as fp plus a fixed-offset frame layout,
  with no runtime stack pointer in the handler signature.

- Each opcode's behaviour is written once as an impl that operates through
  operand pointers; the build emits the per-depth-variant wrappers (D1..DN) as
  thin wrappers that receive the TOS registers by value and hand the impl
  pointers to the correct registers for that depth, keeping the handler count
  linear in opcodes rather than multiplied by depth.

- At a structured-control merge point the TOS values are spilled to their
  operand-stack memory slots, the location at which the branch path and the
  fall-through path agree; the reload is emitted lazily, only where a control edge
  actually reaches the merge and needs the values back in registers.

## Facts

- 2026-01-22 (64784da1) rationale: the TOS cache was migrated under a parallel
  validation mode — each handler computed the result both via the existing
  sp-relative path (treated as source of truth) and via the TOS registers, then
  asserted the two agreed at run time; this dual-compute shadow check let the
  register-cache rewrite be validated against the proven interpreter on the full
  spec suite before the sp path was deleted (code).

- 2026-01-22 (d23fb74a) pitfall: the shadow-validation equality check was first
  a plain 64-bit integer compare of the TOS value against the sp value, which
  false-positives on float ops because NaN propagation through the two compute
  paths can produce different NaN bit patterns for the same logical result; the
  compare was made NaN-aware to treat all NaNs as equal before asserting (code).

- 2026-01-21 (cbd55b49) rationale: a `tos_pattern = { pop, push }` in the
  handler spec drives variant generation, so each stack effect generates exactly
  the per-depth wrappers over one impl; FORCE_INLINE plus LTO collapses the
  wrapper indirection, so the depth fan-out costs code size but not dispatch, and
  the impl never names a physical register (code).

- 2026-01-21 (e56cbf15) rationale: calls spill all TOS registers to memory and a
  callee starts with a fresh state (spill_depth = arity, all params in memory,
  filled on demand) (code).

- 2026-01-21 (e56cbf15) rationale: the fresh-state call convention was chosen for
  uniformity (every function entry has identical state) and simplicity (no
  register-state hand-off across the call boundary) rather than transferring hot
  operands in registers across calls (sourced).

- 2026-01-27 (6e798de5) rationale: when a fused op nets multiple pushes that
  would overflow the four-register window, the required spills are emitted as one
  batched spill_N instruction carrying the slot and count rather than N separate
  spill_1 instructions, trading a single multi-slot store for fewer handler
  dispatches (code).

- 2026-01-22 (3ac01809) optimization: merge-point handling moved from eager fill
  (an explicit fill at IF/ELSE/END) to lazy demand-driven fill — the lazy form
  emits nothing there, marks the TOS state stale, and patches branches straight
  to the body, letting the first operation that needs a value trigger the fill;
  this removes redundant fills when a path falls through or the next op does not
  read TOS (code).

- 2026-01-27 (5f4039ba) optimization: pop-class ops (local_set, global_set,
  table_set, table_grow, drop, select, store, ternary) no longer emit a fill
  when a pop exposes spilled values; the next operation that actually reads
  operands fills them on demand (code).

- 2026-01-28 (7fdb8d42) optimization: at a block END the merge fill is emitted
  only when a control edge reaches the merge (a forward branch that spilled before
  jumping, or an IF block whose false edge reaches END); a pure fall-through
  BLOCK with no pending fixups skips the fill entirely (code).

- 2026-01-27 (049b20b4) pitfall: at END the branch-target index must be captured
  before the merge fill is emitted, not after — forward branches and back-edges
  that spilled to memory jump to that target and must execute the fill to reload
  TOS, otherwise they run on stale registers left by a preceding call_indirect;
  capturing the target after the fill would route those edges past the reload
  (code).

- 2026-02-05 (14137522) measurement: a parked experiment (the committed 5tos.diff
  patch, titled "5 tos, score 4343") bumps the register count from 4 to 5 —
  generating an extra register, a fifth variant per opcode, spill_5/fill_5, and a
  modulo register-assignment — and records a benchmark score of 4343; this is the
  concrete payoff that motivated making the register count a single tunable
  constant (the in-tree count stays 4 through this window) (code).

- 2026-06-14 rationale: four TOS registers is a binary-size-versus-benefit
  balance, not a preserve_none register-budget constraint. Each extra TOS
  register adds a depth-variant per opcode (no xir-style permutation explosion,
  but still more generated handlers and more binary size), and most functions
  never need a deeper window because of how block boundaries spill/refill the
  window — so the marginal benefit of a fifth register is small against its
  size cost. The earlier reduction from eight to four was the same
  GPR-budget/binary-size reasoning, not a separate measurement (sourced).

## Moves

- 2026-01-24 (d0c89f0b) replaced [[sp-stack-machine]]: once the shadow-validated
  TOS path was trusted, removing the sp parameter and the parallel sp computation
  makes the register cache the sole source of truth, eliminating the
  per-instruction memory round-trip the cache existed to avoid; stack addresses
  still needed for spill/fill are derived from fp plus a fixed-offset frame
  layout instead of a tracked sp (code).

- 2026-01-20 (8136fd44) replaced [[register-window-policy-slides]]: WASM
  validation fixes the stack depth at every merge point, so making register
  assignment a pure function of depth makes merge-point state automatically
  consistent — removing the register-window design's per-merge-point policy pass
  and explicit slide instructions (code).

- 2026-01-20 (8136fd44) replaced [[eight-tos-registers]]: PRESERVE_NONE provides
  12 GPRs; five go to interpreter state (ctx, pc, fp, mem, memsz), so capping the
  TOS cache at four registers leaves three reserved for a future hot-locals cache
  instead of consuming the whole register file (code).

- 2026-01-25 (b322a614) replaced [[merge-point-register-normalization]]: register
  normalization could not reconcile a branch path that drops operands with the
  fall-through path that keeps them in different registers, so merge values are
  spilled to memory — the one location both edges agree on (code).

- 2026-01-28 (1dcc1655) replaced [[hardcoded-four-register-tos]]: the register
  count 4 was baked into the design as literal D1..D4 handler-lookup arrays at
  every dispatch site, explicit (count,variant) match arms enumerating only
  variants 0..3, % 4 variant formulas, and a power-of-two & 3 register mask, so
  the count could not be retuned without hand-editing every generator and handler
  and could never take a non-power-of-two value; making it the single build-time
  constant TOS_REGISTER_COUNT that every Rust and C generator derives from —
  variant names D1..DN, register names t0..t{N-1}, the C ABI register params,
  generated per-handler handler_lookup arrays, the variant_index formula
  ((depth-1) % N)+1, and the cyclic register-assignment reg = (height-position)
  % N with the mask falling back to modulo when N is not a power of two — lets the
  whole handler set be regenerated to any register count by changing one constant
  (code).
