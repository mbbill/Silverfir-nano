- WASM-to-WASM calls and returns run on an explicit interpreter call stack, not
  the native host stack: the call handler pushes a frame and tail-calls the
  callee's entry instruction; the return handler pops the frame and tail-calls
  the saved return continuation; an arbitrarily deep call chain consumes no
  native stack (`ShadowStack`).

- The call stack is one pre-allocated contiguous buffer: each frame is a fixed
  block of metadata words at negative offsets from its spill pointer followed by
  its spill slots, with frames linked by a caller-frame word; a call bumps the
  stack top and a return restores it with no heap allocation and no separate
  per-frame metadata structure.

- The first eight arguments and eight results travel in the abstract registers
  v0..v7 across the threaded tail-call chain with no per-call copy; arguments and
  results beyond eight fall back to spill slots. A function's parameters occupy
  the frame's low spill slots.

- A call to an external/host function runs its callback inline with no frame;
  call_indirect and call_ref perform a runtime structural-equivalence signature
  check against the metadata's expected type before dispatching.

- Same-module direct calls are specialized: on first execution every same-module
  function is precompiled in one pass, and the call site is patched from the
  generic call handler to a call_local (or, for <=8 args/results, call_local_reg)
  handler that pre-resolves the callee entry pointer and skips the generic
  handler's compile lookup, cross-module detection, and mem0 refresh.

- A call frame records both the callee's own result slots and the caller's
  result locations, and the return handler copies result i from the callee slot
  to the caller location; neither side's slots are constrained to match
  the other's.

## Facts

- 2025-10-16 (a3cc8801) statement: the call handler distinguishes external (host)
  callees, which run inline with their results stored back immediately and no
  frame push, from internal WASM callees, which push a frame and tail-call the
  callee's entry; only WASM-to-WASM calls participate in the explicit stack
  (diff).

- 2025-10-14 (9b3c306d) pitfall: the indirect-call function table is held behind
  a shared mutable cell, so a borrow of it must be scoped and dropped before
  recursing into the callee; holding the borrow across the nested call panics
  with already-borrowed when the callee does any further table operation (diff).

- 2025-10-24 (35113cfc) pitfall: the tail-call path that pushes a callee frame
  must route through the depth-checked push (which enforces the call-depth limit
  and traps), not a direct push onto the call stack; a direct push bypasses the
  depth check and lets deep recursion overflow without trapping (diff).

- 2025-10-25 (7df9290e) pitfall: a pushed call frame must record the callee
  function's own module instance, not the caller's current module; an imported
  function must execute in its exporting module's context, so using the caller's
  module mislinks globals, memories, and types for cross-module calls (diff).

- 2025-10-26 (8b04bc47) pitfall: the call_indirect / call_ref runtime type check
  must compare expected and callee function types by structural equivalence, not
  exact equality — for the same module instance via the module's TypeContext
  (covering GC subtyping), falling back to comparing resolved function types only
  when the instances differ; exact type-index/pointer equality rejects spec-legal
  indirect calls (diff).

- 2025-11-30 (519253b6) rationale: a frame's module and function metadata are
  held as safe lifetime-checked references borrowing from the Store rather than
  raw pointers; this is sound on the hot path because the Store's module and
  function instances are immutable for the whole of execution, so the borrow
  outlives every frame (diff).

- 2025-12-01 (aaeacbb2) rationale: on return, result words are copied straight
  from the callee's spill slots to the caller's by pointer arithmetic with no
  temporary buffer, because the caller's frame sits immediately before the
  callee's in the one contiguous buffer so the offset is a known constant (diff).

- 2026-02-12 (04f34592) rationale: call_local exists because eager module-level
  precompilation makes every same-module callee already compiled, so a same-module
  call needs none of the generic call handler's per-call runtime work — it
  pre-resolves the entry pointer, skips the store function lookup, skips
  cross-module detection, and skips the mem0 refresh (the callee shares the
  caller's module and memory) (diff).

- 2025-10-25 (73d8710c) rationale: call_ref is lowered like call and call_indirect
  rather than as a window-register operation — the funcref vreg index, arg vregs,
  and result vregs all travel in side metadata and the window is flushed before the
  call, so all calls share one control-flow lowering shape (diff).

## Moves

- 2025-10-15 (0d24ab09) replaced [[eval-env-depth-guard]]: EvalEnv was a separate
  heap structure threaded by NonNull across frames with a Drop guard to unwind the
  counter, which forced the unsafe eval_env pointer round-trip; carrying
  call_depth as a plain usize on Ctx and passing depth+1 into the callee's
  eval_internal removes the shared structure and its guard entirely (diff).

- 2025-10-16 (a3cc8801) replaced [[recursive-native-stack]]: recursive eval grew
  the native host stack one frame per WASM call; an explicit heap-allocated call
  stack in Ctx that the call handler pushes and the ret handler pops keeps WASM
  calls and returns off the native stack (diff).

- 2025-11-30 (0ebc0961) replaced [[per-frame-heap-spill]]: each call allocated a
  fresh Vec for the frame's spill slots (and another for result-slot indices),
  putting heap allocation on every call and return; a single pre-allocated buffer
  with bump-pointer push and instant pop makes call/return allocation-free (diff).

- 2026-02-07 (87f0ff12) replaced [[split-frame-metadata]]: the old call stack
  split each frame between a pre-allocated spill buffer (the spill slots) and a
  separate Vec<Frame> of borrowed-lifetime metadata structs reached behind a
  RefCell, so every push/pop touched two structures and the RefCell guarded the
  frame vector on the hot call path; collapsing both into one contiguous u64
  buffer where each frame is FRAME_METADATA_SLOTS (5) metadata words at negative
  offsets from its spill pointer followed by its spill slots removes the
  Vec<Frame>, removes the RefCell, and keeps a frame's metadata and spill area
  cache-adjacent, with frames linked by a caller_frame_start word and the root
  frame marked by its metadata sentinel (diff).

- 2026-02-13 (7f97c463) replaced [[spill-slot-call-convention]]: passing
  arguments and results through spill slots forced a memory copy on every call and
  return even for the common small-arity case; the register convention places the
  first 8 args and 8 results in the abstract registers v0..v7 so they ride in CPU
  registers across the threaded tail-call chain with no per-call copy, falling
  back to slots only for the >8 overflow (diff).
