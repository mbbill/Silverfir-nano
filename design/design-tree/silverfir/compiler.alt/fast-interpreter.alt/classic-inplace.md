- The classic backend interprets the WebAssembly bytecode directly in place,
  decoding each opcode and its immediates from the function body on every
  execution with no separate compiled form.

- Operands live on a single runtime value stack of raw 64-bit words; the same
  stack carries locals, parameters, and the operand stack within a frame, laid
  out per frame as [arguments | locals | operand stack], with locals indexed
  from the frame's base offset.

- Calls and returns are driven by an explicit heap-allocated frame stack with a
  trampoline — a call pushes the callee frame and breaks the inner dispatch
  loop, a return pops — rather than native host recursion, leaving call depth as
  data the interpreter controls.

- Structured control flow is resolved through a precomputed jump table:
  branches index a table entry that supplies the target program counter and
  the count of operand slots to drop; block exits move results without
  re-scanning the body.

- Call depth and operand-stack size are bounded by fixed limits checked at
  runtime to trap runaway recursion or stack growth.

- Reference-typed calls (call_ref) and the runtime type tests (ref.test /
  ref.cast) are interpreted in place against the same untyped value stack as
  every other opcode, resolving and signature-checking the referenced function
  or object before dispatch.

## Facts

- 2024-02-23 (9563bcd8) rationale: calls and returns are driven by an explicit
  heap `frame_stack: Vec<Frame>` with a trampoline (a call pushes the callee
  frame and breaks the inner loop, a return pops) rather than native Rust
  recursion, so call depth is data the interpreter controls rather than the
  host call stack (diff).

- 2024-03-11 (5893e9f8) pitfall: because the classic interpreter re-decodes
  immediates from the body on every execution, a call's caller frame must be
  pushed only after the call's immediate operand (callee index / type index)
  has been read; saving the frame first records a return PC that still points
  at the immediate and re-executes it on return (diff).

- 2024-03-12 (30a6cf18) pitfall: a zero-size memory.init / table.init must
  succeed even when its source data/element segment has already been dropped,
  so the dropped-segment trap check must run only for nonzero size and only
  after the out-of-bounds checks; checking dropped-ness first wrongly traps the
  spec-legal zero-size case (diff).

- 2024-03-13 (e6a547d4) pitfall: table.copy borrowed the destination and source
  TableInst RefCells unconditionally, which double-borrow-panics when src and
  dst are the same table; the same-table case must take one borrow_mut and
  copy_within so overlapping ranges shift correctly, while the distinct-table
  case bounds-checks against the source length before copy_from_slice (diff).

- 2024-03-13 (b6040fd3) pitfall: the trunc range guards used `value > i32::MAX
  as f32` style bounds, but casting an integer extremum to f32/f64 rounds to a
  nearby power of two and admits out-of-range inputs; the trap boundary must be
  the exact representable limit compared with >= / <= (e.g. >= 2147483648. for
  i32, >= 9223372036854775808. for i64), which is also why the same guards are
  reused by the saturating trunc_sat variants (diff).

- 2024-03-14 (c64a098b) pitfall: Rust's float methods do not match Wasm float
  semantics: f32::round() rounds half away from zero but Wasm nearest is
  round-half-to-even (needs round_ties_even, a nightly feature gate), and
  f32::min/max return the non-NaN operand whereas Wasm min/max must propagate
  NaN when either operand is NaN (diff).

- 2024-03-15 (e4384b79) pitfall: i32.rem_s / i64.rem_s must NOT trap on MIN % -1
  (that overflow trap belongs only to div_s); the remainder is defined as 0
  there, so the eval loop computes it with wrapping_rem and the spurious
  overflow guards were removed (diff).

- 2024-03-16 (6d3edd12) pitfall: on call and call_indirect the eval loop must
  set its module-switching flag from whether the callee is imported, so a call
  crossing into another module's (or a host) function rebinds the active module
  instance for the new frame; missing this leaves the callee executing against
  the caller's module context (diff).

- 2024-03-18 (f0f859e7) pitfall: the TABLE_SET handler popped the table index
  before the stored reference, but the value operand sits on top of the operand
  stack above the index; pop-based handlers in the in-place decoder must pop the
  topmost operand (the value) first, then the index (diff).

- 2025-06-23 (09dbe4a3) rationale: the interpreter dispatches an external
  callee inline at the CALL site without pushing a frame or entering the
  bytecode loop — it pops the argument words, converts them to typed Values via
  the callee's parameter types, invokes the host callback, checks the returned
  count against the result signature, converts the results back to raw words and
  pushes them, then continues — so the host-call boundary is exactly where the
  raw operand word meets the tagged external Value (diff).

- 2025-10-06 (7c2ec193) pitfall: because a reference on the operand stack is a
  bare RefHandle index carrying no abstract-heap-type tag, the classic backend
  cannot distinguish func/any/extern references of a plain (non-GC-heap)
  reference at runtime, so ref.test/ref.cast against an abstract heap type
  conservatively accepts a match for any of Func/Any/Extern; precise
  abstract-heap-type testing would need type information the untyped value
  representation does not retain (diff).

- 2025-10-06 (3ccf0d6f) rationale: ref.test/ref.cast against a concrete type
  accepts an object whose type is structurally equivalent to the target (not
  just an index match or declared-supertype match), via a recursive
  composite+supertype structural comparison; the recursion is capped at depth 10
  to guard against pathological or cyclic type graphs (diff).

- 2026-06-14 statement: the classic in-place interpreter is the project's
  correctness/validation oracle, not an ancestor of fast — it coexisted as a
  sibling alongside fast and the ssa-compiler in -rs. wasm 3.0 features are easy
  to implement and debug in place, so once classic runs a feature well it
  isolates faults: if classic passes and fast/ssa fail, the bug is in fast/ssa,
  not in the spec understanding. This oracle role is what justifies keeping a
  deliberately slow, straightforward in-place interpreter around (author).

## Moves

- 2026-02-14 removed: the in-place correctness oracle was one of three coexisting
  -rs interpreters (classic, fast, ssa-compiler); the -nano restart carried only
  the fast interpreter forward, and the oracle was not worth carrying because
  -nano is JIT-only at HEAD — its no-register-allocator design already makes
  debugging easy without a separate oracle engine (author).
