- The fast interpreter's value stack is an owned Vec<u64> passed in through the
  Context; each call site grows it on demand before setting up the callee frame.

- Stack memory is sized dynamically per evaluation rather than from a fixed
  per-thread reservation.

## Moves

- 2025-12-12 (c455c3e2) replaced by [[calls]]: native-stack recursion holds raw
  frame-pointer pointers into the value stack across nested run_trampoline calls,
  so the stack must never reallocate; the dynamically grown owned Vec is replaced
  by a thread-local buffer pre-allocated to the maximum size with a stack_end
  pointer for overflow detection, removing per-call growth checks and keeping frame
  pointers stable across calls (code).
