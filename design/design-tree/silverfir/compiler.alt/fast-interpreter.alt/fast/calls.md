- Every internal call — same-module and cross-module — is linked inline on the
  value stack rather than entering a fresh native trampoline invocation: a call
  writes its per-frame metadata above the callee frame and tail-jumps to the
  callee entry, and a return restores fp/sp from those slots and tail-jumps to the
  return PC, keeping the whole call chain inside one trampoline loop with no
  native recursion.

- Per-frame metadata is three fixed slots above the frame (return_pc, saved_fp,
  saved_module); the entry frame is marked by a NULL saved_fp sentinel and a
  same-module call writes saved_module zero, with a return paying the cross-module
  module/mem0 restore only when saved_module is nonzero.

- A module's internal functions are precompiled in two passes that let even
  forward calls resolve to a baked callee entry pointer: pass one compiles every
  function exactly once with every internal call emitted as a general internal
  call (all entry pointers then exist), and pass two patches each same-module
  internal call to a direct same-module call in place, baking the callee's entry
  pointer at that point — no function is recompiled and no separate fixup of baked
  entry pointers is needed (`precompile_module_two_pass`).

- The call site is specialized at build time: a same-module call to an
  already-compiled callee carries the callee's baked entry pointer and jumps
  straight to the callee's first instruction; the general internal path encodes
  only the callee instance and derives frame sizes at run time; an external/host
  call is a separate handler that crosses the host boundary.

- The callee's frame setup — stack-capacity check, call-depth check, and locals
  zeroing — is performed inline by the call handler itself (the same-module C
  handler, or the general/cross-module Rust path), not by any per-function
  prologue instruction; the baked entry pointer points at the callee's first
  real instruction.

- Runaway recursion is bounded by an explicit call-depth counter trapped past a
  fixed maximum, not by the native OS stack.

- The value stack is a thread-local buffer pre-allocated to a fixed maximum size
  with a stack_end overflow pointer; it never reallocates and frame pointers stay
  stable across nested calls.

## Facts

- 2025-12-14 (b9499733) statement: the native call-depth ceiling
  (MAX_CALL_DEPTH) was lowered from 1000 to 300, kept in lockstep across the C
  and Rust definitions; the native-stack run_trampoline recursion the guard bounds
  already existed at 1000 in the parent commit, and the diff carries no rationale
  for the new ceiling (diff).

- 2026-06-14 rationale: 300 was a transient tuning, not a durable limit — at that
  point the call frame was fat, so 1000-deep recursion overflowed the native
  stack and the ceiling was lowered to pass spectest; the frame size was
  optimized later, so the 300 is tied to the then-fat frame rather than a chosen
  depth (author).

- 2025-12-13 (2139b159) rationale: after a callee returns, the same-module call
  path checks neither for a trap nor for a stale cached memory pointer on the
  common case — a single dirty_flag (set only when memory.grow runs or a trap is
  recorded) gates one slow path that does the error check and refreshes the cached
  mem0 base/size; a clean return falls straight through (diff).

- 2025-12-11 (75c5eab8) rationale: a single compilation pass can only emit a
  direct same-module call to a callee already compiled, so forward calls (to
  higher-indexed, not-yet-built functions) would miss the optimization; pass two
  recompiles after every function exists, and because pass two may move a callee's
  instructions, a final in-place patch rewrites each baked entry pointer to the
  callee's final pass-two entry to avoid stale pointers (diff).

- 2025-12-11 (a40c7b7c) rationale: the two precompiled internal-call handlers
  settle into a hot/cold split — the same-module path carries its baked entry
  pointer so dispatch jumps straight into the callee prologue, while the general
  internal path encodes only the callee instance and stack-top and derives entry
  and frame sizes from the callee spec at run time, so only the same-module path
  needs its baked entry pointer patched after pass two (diff).

- 2025-12-10 (af606591) rationale: frame constant/temp initialization is done by
  a single copy of a dense per-function blob into the temps region rather than
  scatter-writing (slot,value) pairs — the constants table was changed from a
  sparse array of absolute-slot/value pairs to a dense array sized to the temp
  count (zeros for non-constant temps), so frame setup is one memcpy instead of a
  loop indexing each constant slot (diff).

- 2026-02-05 (14137522) statement: per-frame metadata grew from two slots
  (return_pc, saved_fp) to three by adding saved_module; on a same-module call
  saved_module is written 0 and on a cross-module call it holds the caller
  ModuleInst pointer, and a return reads it after restoring fp and only when
  nonzero (unlikely branch) restores ctx.current_module and refreshes mem0 — so
  the common same-module return pays nothing for cross-module support (diff).

- 2026-02-06 (6d427422) rationale: the return opcode is specialized at build time
  by arity into three handlers — return_void (arity 0), return_one (arity 1), and
  the general return (arity >= 2) — sharing one epilogue; the void variant copies
  no results and the single variant skips the result-copy loop, so the
  overwhelmingly common void/single returns avoid both the loop and the extra
  field decodes (diff).

- 2026-01-25 (b322a614) pitfall: an internal call routed through the entry-frame
  trampoline decremented call_depth twice — once in the Rust wrapper after
  run_trampoline returned and once in the C return handler that runs on the entry
  frame's return — corrupting the runaway-recursion guard; the fix removes the
  Rust-side decrement and lets the return handler be the sole owner of the
  call_depth counter (diff).

- 2026-01-25 (b322a614) pitfall: the builder computed call deltas with a +2
  metadata offset (return_pc/saved_fp placed before callee_fp), but the C handlers
  store frame metadata after the locals with a saved_fp==NULL sentinel for the
  entry frame; the deltas were corrected to drop the +2 so callee_fp lands on the
  arguments and returns write results into the caller's operand slots (diff).

- 2025-12-02 (8f02c607) pitfall: a saved call frame must record the caller's
  locals base as an index (offset in u64 slots from the value-stack base), not a
  raw pointer — the value stack can reallocate when a nested call grows it, after
  which a stored raw locals_base pointer dangles; storing the index and
  recomputing base.add(index) on return survives reallocation (diff).

- 2025-08-16 (df24eaeb) pitfall: a failed indirect call must trap, not silently
  continue — the error conditions (bad type index, out-of-range table element,
  null element, signature mismatch) previously returned the instruction's
  fallthrough edge, resuming the next instruction as if the call succeeded; they
  now take the trap edge (a zero-length memory.init/table.init still performs its
  bounds check but does not trap on a dropped segment, matching the bulk-memory
  zero-length edge case) (diff).

- 2025-08-18 (d168db0a) rationale: on RETURN the fast backend relocates the
  function's result values to the frame's result base using a validator-emitted
  (stack_offset, arity) fixup encoded in the instruction immediates — the same
  fixup path branches use — rather than computing the result location from a live
  stack pointer at exit (diff).

- 2025-12-12 (c455c3e2) statement: because the thread-local fast-interpreter stack
  is reserved at full size up front per thread rather than grown on demand, the
  maximum stack-memory constant was reduced from 8 MiB to 2 MiB to bound the
  always-resident per-thread reservation (diff).

- 2026-06-14 statement: native-stack recursion was abandoned for the inline
  value-stack call model for three reasons the diffs do not show — native
  call/return cannot express WebAssembly return_call (frame reuse); native-stack
  recursion performed worse; and a native call stack is hard to control in a JIT,
  with no graceful shutdown when it runs out. The explicit call-depth counter plus
  the fixed thread-local value stack replace it, keeping overflow recoverable
  (author).

## Moves

- 2026-02-05 (14137522) replaced [[native-recursion-with-inline-same-module]]: the
  general internal-call path still entered each callee through a fresh nested
  run_trampoline native invocation that switched module context on the native
  stack and unwound on return, so cross-module calls could not be linked inline;
  adding a third frame-metadata slot (saved_module) that records the caller's
  module on a cross-module call lets enter_unified_callee link the frame inline on
  the value stack and return the callee entry for tail-call dispatch, and
  impl_return restores the caller's module/mem0 from saved_module (zero sentinel
  for same-module), keeping the whole call chain — cross-module included — inside
  one trampoline loop with no native recursion (diff).

- 2025-12-11 (8654e952) replaced [[lookup-based-internal-call]]: the internal-call
  handler still resolved its callee by func_idx through store.instance_at_module on
  every call; once functions are precompiled the callee's entry instruction
  pointer, FunctionInst pointer, and param/result/locals counts are known at build
  time and baked into the instruction, so the hot call path does no store lookup at
  all (diff).

- 2025-12-12 (c455c3e2) replaced [[heap-grown-value-stack]]: native-stack
  recursion holds raw frame-pointer pointers into the value stack across nested
  run_trampoline calls, so the stack must never reallocate; the dynamically grown
  owned Vec is replaced by a thread-local buffer pre-allocated to the maximum size
  with a stack_end pointer for overflow detection, removing per-call growth checks
  and keeping frame pointers stable across calls (diff).
