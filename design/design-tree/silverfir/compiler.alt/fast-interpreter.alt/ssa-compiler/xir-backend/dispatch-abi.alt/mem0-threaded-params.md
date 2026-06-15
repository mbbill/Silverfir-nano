- The handler calling convention carries the memory-0 base pointer and byte
  size as two dedicated parameters passed through every `preserve_none` tail
  call, kept in CPU registers across the handler chain alongside the spill
  pointer and the abstract registers (`mem0`, `mem0_len`).

- Memory-touching handlers receive double pointers (`pmem0`, `pmem0_len`) through
  which `memory.grow`, call, and return write back a refreshed base/size
  after a backing-buffer reallocation or a module switch.

- The eval entry point looks up the entry module's mem0 base/size once and
  passes them as the initial trampoline arguments.

## Facts

- 2025-11-30 (12d8f841) rationale: the calling module's primary memory base and
  length are threaded through the handler signature so memory handlers skip the
  store lookup on the hot path; for same-module calls the pointers stay unchanged
  (diff).

- 2025-11-30 (12d8f841) pitfall: the cached base/length pointers go stale across a
  module boundary and after memory.grow (the memory's backing buffer can
  reallocate), so they are refreshed when crossing into a callee's module and again
  on every return to the caller's module (diff).

## Moves

- 2026-02-07 (88b3fda7) replaced by [[dispatch-abi]]: the XIR handler ABI
  passed the memory-0 base pointer and size as two dedicated parameters threaded
  through every preserve_none tail call (13-param wrapper convention), forcing
  each handler to carry mem0 in registers across the whole chain and
  double-indirecting them (pmem0/pmem0_len) so call/return/memory.grow could
  refresh them; moving mem0_base/mem0_size into a #[repr(C)] hot-field prefix of
  Ctx at fixed offsets 0/8 lets C handlers read ctx->mem0_base/ctx->mem0_size
  directly and frees two register slots in the calling convention (13->11
  params), with memory.grow/call/return refreshing via ctx.set_mem0 instead of
  writing back through the parameter pointers (diff).
