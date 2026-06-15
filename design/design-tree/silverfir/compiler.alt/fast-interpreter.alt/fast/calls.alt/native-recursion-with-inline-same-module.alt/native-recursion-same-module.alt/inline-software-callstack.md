- Caller state across a call is saved in a software CallStack of CallFrame
  records (return PC, caller frame-pointer index, caller function and module); a
  call pushes a frame and a return pops it, all within a single flat
  run_trampoline invocation that is never recursively re-entered.

- A call handler returns the callee's entry instruction to continue the same
  trampoline tail-chain; the return handler restores the caller's PC, frame
  pointer, and cached module from the popped CallFrame, and returns the terminal
  instruction only when the root frame's CallStack is empty.

- Call-stack overflow is detected by bounding the CallStack length at a maximum;
  the value stack is a heap-backed Vec grown on demand.

## Facts

- 2025-12-02 (22063dd8) statement: the call stack is a pre-allocated Vec<CallFrame>
  (initial capacity 64) carried in the trampoline Context so no heap allocation
  happens on the hot call/return path; on call the caller's return PC, locals base,
  function/module instances, fast-blob base, and result-slot offset are saved into
  a CallFrame and the trampoline tail-jumps to the callee entry, while the terminal
  handler checks the call stack — empty means the outermost function returned and
  interpretation ends, otherwise it pops the caller frame and tail-jumps back to
  the caller's return PC (diff).

## Moves

- 2025-12-02 (22063dd8) replaced [[native-stack-recursion-internal-eval]]: the
  prior internal-call path recursed into internal_eval for every callee, bounding
  wasm call depth by the native OS stack and risking native stack overflow; saving
  the caller's frame on an explicit pre-allocated call stack and jumping to the
  callee's entry (returning by popping that frame) keeps all nesting on the
  heap-backed stack and lets the depth limit be enforced explicitly (diff).

- 2025-12-12 (c455c3e2) replaced by [[native-recursion-same-module]]: the inline
  call-stack model saved each caller's state in a software CallStack of CallFrames
  and returned the callee entry to continue one flat trampoline so calls never
  recursed on the native stack; switching to a recursive run_trampoline per call
  lets the native stack hold caller state directly, so the handlers no longer
  maintain CallFrame save/restore and a return simply terminates the current
  trampoline, with a call_depth counter (and a stack_end guard) replacing the
  CallStack's overflow check (diff).
