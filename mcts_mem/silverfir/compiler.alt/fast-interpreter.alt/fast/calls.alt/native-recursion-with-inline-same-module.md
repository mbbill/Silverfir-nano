- A general internal call enters its callee through a fresh nested run_trampoline
  native invocation; the caller's instruction stream resumes only after that inner
  trampoline unwinds.

- Per-frame metadata is two fixed slots above the frame (return_pc, saved_fp); the
  operand stack base is at frame_size + 2.

- The callee module is made current by switching ctx.current_module and refreshing
  mem0 before the inner trampoline runs, and the native stack unwinding restores
  the caller's module on return.

- Only same-module calls link frames inline on the value stack; the general
  internal-callee path (including cross-module) uses run_trampoline.

## Moves

- 2025-12-17 (53042ab1) replaced [[native-recursion-same-module]]: the old
  same-module call entered a fresh native run_trampoline invocation per WebAssembly
  call, so every call paid a native call/return, a ctx call-depth bump, and a
  post-return dirty-flag slow-path check, and the return handler could only
  terminate the inner trampoline; linking frames inline on the value stack (each
  call writes return_pc and saved_fp above the callee frame and tail-jumps to the
  callee entry, return restores fp/sp from those slots and tail-jumps to return_pc,
  the entry frame marked by a NULL saved_fp sentinel) keeps the whole call chain
  inside one trampoline loop with no native recursion, replacing the precomputed fp
  delta with sp-relative arg placement and dropping the separate results_count
  encoding in favor of the return arity (code).

- 2026-02-05 (14137522) replaced by [[calls]]: the general internal-call path
  still entered each callee through a fresh nested run_trampoline native invocation
  that switched module context on the native stack and unwound on return, so
  cross-module calls could not be linked inline; adding a third frame-metadata slot
  (saved_module) that records the caller's module on a cross-module call lets
  enter_unified_callee link the frame inline on the value stack and return the
  callee entry for tail-call dispatch, and impl_return restores the caller's
  module/mem0 from saved_module (zero sentinel for same-module), keeping the whole
  call chain — cross-module included — inside one trampoline loop with no native
  recursion (code).
