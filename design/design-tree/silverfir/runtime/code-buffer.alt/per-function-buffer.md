- Each compiled function owns an optional mmap'd executable CodeBuffer, stored
  on its FastCode, that keeps its JIT'd native code alive for the function's
  lifetime.

- resolve_jit finishes the write session over the whole buffer (offset 0 to
  total length), since the buffer holds only this one function's code.

## Moves

- 2026-03-05 (c6caf745) replaced by [[code-buffer]]: a single per-module
  executable arena lets every function's native code share one mmap'd region and
  append into it (finish_write commits only the newly written range) instead of
  each function holding its own buffer (author).
