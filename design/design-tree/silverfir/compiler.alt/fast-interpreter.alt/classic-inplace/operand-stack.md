- The operand stack is a single pre-allocated buffer of raw words addressed by
  an explicit stack pointer (`InterpreterStack`); pushing and popping move the
  pointer, and a multi-pop returns a borrowed slice of the buffer rather than an
  owned allocation.

- One stack is shared across the entire call chain: each frame records a base
  offset into the single buffer, and a callee's already-pushed arguments become
  the base of its locals.

- The buffer is sized from validation output, not guessed: the validator records
  each function's peak operand-stack height and the entry frame is allocated for
  locals plus that peak, removing operand-stack growth during straight-line
  execution; at every call the stack is grown to current usage plus callee
  locals plus callee peak height before the callee's locals are pushed.

## Facts

- 2025-06-22 (df17d674) rationale: the function-body validator records each
  function's peak operand-stack height as it type-checks (max_stack_height on
  FunctionSpec), and the interpreter reads it to size the execution stack from
  validation output rather than guessing, allocating locals + peak so the
  operand stack need not grow during straight-line execution (diff).

- 2025-06-22 (1641d843) pitfall: the first exact-sizing cut pre-allocated with
  Vec::with_capacity, which sets capacity but leaves the buffer length at zero,
  so every push still had to test sp >= buffer.len() and resize; the buffer must
  be allocated with length (vec![0; size]) for the stack pointer to index it
  directly — capacity is not length (diff).

- 2025-06-23 (86cfe73a) rationale: one InterpreterStack is shared across the
  entire call chain — each Frame records a local_start base offset into the
  single buffer and a callee's already-pushed arguments become the base of its
  locals — so per-function exact sizing covers only the entry frame; at every
  call the stack is grown to current_usage + callee_locals + callee_peak_height
  before the callee's locals are pushed (diff).

- 2025-10-07 (94652766) rationale: a second runtime bound caps the operand stack
  to 8 Mi RawValue words (MAX_STACK_SIZE, 1024*1024*8 at this commit; later
  reduced to 2 Mi words), checked at the call boundary by summing current usage
  plus callee locals plus callee max operand-stack height (all word counts)
  before growing the stack; this converts what would be an out-of-memory
  exhaustion of host memory into a clean exhaustion trap ('call stack
  exhausted'), complementing the pre-existing call-depth bound
  (MAX_CALL_STACK_DEPTH, 'call stack overflow') (diff).

## Moves

- 2025-06-22 (3bedfb76) replaced [[vec-operand-stack]]: a Vec used directly as a
  stack reallocates as it grows and its multi-pop allocated a fresh owned Vec on
  every call, so the operand stack is reworked into a pre-allocated buffer
  addressed by an explicit stack pointer whose multi-pop returns a borrowed
  slice, eliminating per-operation heap traffic in the interpreter hot loop
  (diff).
