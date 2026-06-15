- A same-module call enters its callee through a fresh nested run_trampoline
  native invocation; the call handler returns to the caller's instruction stream
  only after that inner trampoline unwinds.

- Each call bumps a call-depth counter in the context (trapping past a fixed
  maximum) and, after the callee returns, checks a context dirty flag to run a
  memory-resync slow path.

- The callee frame pointer is reached by a precomputed compile-time fp delta, and
  the call instruction encodes params_count, locals_count, and results_count,
  letting the caller place results at callee_fp[0..results_count] after the inner
  trampoline returns.

- return copies its results to fp[0..arity) and returns the terminal instruction
  to unwind the current trampoline; it carries no caller-restore information, the
  native stack holding the caller context.

## Moves

- 2025-12-12 (c455c3e2) replaced [[inline-software-callstack]]: the inline
  call-stack model saved each caller's state in a software CallStack of CallFrames
  and returned the callee entry to continue one flat trampoline so calls never
  recursed on the native stack; switching to a recursive run_trampoline per call
  lets the native stack hold caller state directly, so the handlers no longer
  maintain CallFrame save/restore and a return simply terminates the current
  trampoline, with a call_depth counter (and a stack_end guard) replacing the
  CallStack's overflow check (diff).

- 2025-12-17 (53042ab1) replaced by [[native-recursion-with-inline-same-module]]:
  the old same-module call entered a fresh native run_trampoline invocation per
  WebAssembly call, so every call paid a native call/return, a ctx call-depth bump,
  and a post-return dirty-flag slow-path check, and the return handler could only
  terminate the inner trampoline; linking frames inline on the value stack (each
  call writes return_pc and saved_fp above the callee frame and tail-jumps to the
  callee entry, return restores fp/sp from those slots and tail-jumps to return_pc,
  the entry frame marked by a NULL saved_fp sentinel) keeps the whole call chain
  inside one trampoline loop with no native recursion, replacing the precomputed fp
  delta with sp-relative arg placement and dropping the separate results_count
  encoding in favor of the return arity (diff).
