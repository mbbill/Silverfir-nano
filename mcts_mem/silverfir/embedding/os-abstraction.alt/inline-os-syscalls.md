- Executable-memory and guard-page allocation call host syscalls
  (mmap/munmap/mprotect/VirtualAlloc/pthread_jit_write_protect_np) through
  extern declarations inlined in each runtime file, dispatched per-OS at the
  call site.

- There is no bare-metal embedder-shim contract: a target without a hosted OS
  cannot supply its own page-management primitives.

## Moves

- 2026-04-07 (b6d6e3de) replaced by [[os-abstraction]]: each runtime file
  (code_buf, guard_pages, trap_signal) declared its own
  mmap/VirtualAlloc/pthread_jit_write_protect_np externs and dispatched per-OS
  inline, so adding a target meant editing every syscall site and there was no
  embedder-shim seam for a hosted-OS-free target; isolating every host primitive
  behind one os/ module makes adding a target a matter of adding one submodule
  and lets bare-metal (`none`) bind the same API to embedder-provided extern
  shims (code).
