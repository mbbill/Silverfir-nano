- An internal call is performed by recursively invoking the per-function
  evaluator (internal_eval) on the callee over the shared value stack; the native
  call stack mirrors the wasm call stack, and the terminal handler performs no
  tail-call, unwinding back through the initial trampoline frame.

- Call depth is bounded only by the native stack: a deeply recursive wasm program
  overflows the OS stack rather than trapping at a checked limit.

## Facts

- 2025-08-14 (9ae55b99) rationale: because the hot top-of-stack register window is
  private to each frame but operands live on the shared stack, an internal call must
  flush the caller's register window down to the shared stack before handing args to
  the callee, and rebuild the window from the shared stack tail on return; the
  cached heap base/limit are also refreshed after every call since the callee may
  have grown linear memory (diff).

- 2025-08-14 (b66fe0c0) pitfall: the spill cursor and the locals base are raw
  *mut u64 pointers into the shared value stack's buffer, so any reallocation of the
  stack is fatal — a growing push would move the buffer and dangle every cached
  pointer; this forces the stack to be pre-sized to locals + max_stack_height per
  frame and never reallocate while a frame is live (diff).

- 2025-08-14 (986052bb) rationale: to satisfy the never-reallocate-while-live
  requirement the value stack stops being a growable Vec push and becomes a fixed
  pre-reserved buffer with a used watermark — push/pop index a used cursor instead
  of growing, and the buffer is pre-sized before a frame goes live so cached raw
  pointers into it stay valid (diff).

## Moves

- 2025-12-02 (22063dd8) replaced by [[inline-software-callstack]]: the prior
  internal-call path recursed into internal_eval for every callee, bounding wasm
  call depth by the native OS stack and risking native stack overflow; saving the
  caller's frame on an explicit pre-allocated call stack and jumping to the
  callee's entry (returning by popping that frame) keeps all nesting on the
  heap-backed stack and lets the depth limit be enforced explicitly (diff).
