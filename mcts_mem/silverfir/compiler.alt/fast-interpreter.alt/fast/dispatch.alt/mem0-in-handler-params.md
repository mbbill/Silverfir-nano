- mem0_base (the memory-0 base pointer) and mem0_size are threaded as two by-value
  parameters in the handler calling convention, passed in CPU registers on every
  tail-call dispatch.

- C memory handlers read memory 0 by dereferencing the double-pointers handed in
  as parameters.

- The C-visible context hot prefix mirrors only stack_end and call_depth; mem0 is
  not part of the context's hot fields.

## Moves

- 2026-02-05 (91b0de39) replaced by [[dispatch]]: the memory-0 base pointer and
  size were passed by value as two extra arguments in every handler's preserve_none
  signature (and dereferenced from double-pointers in the C handlers), widening the
  fixed dispatch ABI for values that change only on memory.grow; moving them into
  the C-visible CtxHot hot prefix of Context lets handlers read
  ctx_mem0_base/ctx_mem0_size directly and drops the two parameters from the calling
  convention, freeing argument registers in the hot dispatch path (code).
