- Every exported native invocation allocates its own operand/call stack,
  constructs a fresh NativeContext, scans all function ABIs for the maximum
  frame, and rebuilds globals, memory, table, type-canonicalization, and function
  dispatch views before entering generated code.

## Moves

- 2026-07-22 replaced by [[invocation-cache]]: rebuilding invocation state on
  every exported call dominated short workloads; Store-owned
  revision-validated reuse removes fixed work while take/return ownership
  preserves re-entrancy (sourced).
