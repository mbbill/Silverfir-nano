- Finalized native code lives in a single mmap-backed module-wide W^X arena per
  module: all of the module's functions append into one executable region whose
  write/execute permission is toggled per platform and whose instruction cache
  is explicitly invalidated before execution (`CodeBuffer`).

- The buffer abstracts the OS executable-memory primitive behind a per-OS W^X +
  icache-sync contract (begin-write flips to read-write, finish-write flips to
  read-execute and flushes the written range) rather than a single allocation
  API; a new host OS is one more backing implementation.

## Facts

- 2026-03-03 (d5d82139) rationale: the code buffer stays no_std by calling
  mmap/munmap (and macOS MAP_JIT + pthread_jit_write_protect_np /
  sys_icache_invalidate) directly via extern "C" libc FFI rather than depending
  on std; the ctx_offset constants the generated code bakes into loads/stores
  mirror the runtime-context struct layout and are pinned by an offset-check so
  a layout change is caught rather than silently miscompiled (diff).

- 2026-03-25 (9891fad0) statement: the OS executable-memory primitive is
  abstracted per target — POSIX uses mmap/mprotect plus icache invalidation,
  Windows uses VirtualAlloc/VirtualFree, VirtualProtect (RW for begin_write, RX
  for finish_write) and FlushInstructionCache — written read-write then flipped
  to read-execute with an instruction-cache flush over the written range (diff).

- 2026-04-21 (a80b238f) rationale: the streaming compile path installs the
  freshly-emitted CodeBuffer directly into the module instead of a
  build-then-swap, because the swap allocated a second throwaway CodeBuffer from
  the executable allocator just to swap it out — that double allocation is
  significant on MCU targets where the OS layer serves a single fixed executable
  arena (diff).

## Moves

- 2026-03-05 (c6caf745) replaced [[per-function-buffer]]: a single per-module
  executable arena lets every function's native code share one mmap'd region and
  append into it (finish_write commits only the newly written range) instead of
  each function holding its own buffer (author).
