- Each function's instruction stream is recompiled into an internal
  pre-decoded IR: decode work (opcode dispatch, LEB immediates) is paid once
  at IR-build time, never during execution.
- The IR is built lazily, on a function's first execution, and cached on the
  function.
- Instruction fusion is always enabled — never a feature gate.
- Targets single-threaded speed on stable Rust without JIT — native code
  generation is out of scope.
