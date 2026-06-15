- The handler calling convention passes everything in registers under the
  `preserve_none` convention and chains handlers with a forced `musttail` tail
  call, keeping the abstract register window and the dispatch pointers in CPU
  registers across an arbitrarily long handler chain with no per-handler
  prologue/epilogue and no native-stack growth (`XIR_WRAPPER_PARAMS`).

- Per-op trampoline tail-chaining is the sole dispatch mechanism: every handler
  enters through the trampoline and tail-chains to the next; there is no
  computed-goto dispatcher and no interpreter fallback path.

- The trampoline entry point uses the ordinary C calling convention, not
  `preserve_none`: it must be callable from Rust across the FFI and a `musttail`
  return is impossible across a calling-convention mismatch; the entry makes a
  normal call into the first handler and returns once the terminal handler
  unwinds back to it.

- Memory-0 base and size are cached in a `#[repr(C)]` hot-field prefix of the
  handler context at fixed offsets; C handlers read them directly off the
  context rather than carrying them as dedicated calling-convention parameters,
  and the return path refreshes them unconditionally.

## Facts

- 2025-09-28 (8ca554d1) rationale: the backend commits to per-op trampoline
  tail-chaining as the sole dispatch mechanism — every handler enters via the
  trampoline and `musttail`-chains to the next, keeping hot state in registers;
  no computed-goto dispatcher and no interpreter fallback path is maintained
  (diff).

- 2025-10-08 (303bf064) rationale: the handler ABI dropped the value-stack
  pointer it originally carried, a vestige of a stack-machine ABI; the backend is
  register-based (operations read and write the register window and the backing
  file directly, never a push/pop value stack), so no handler needs a stack
  pointer (diff).

- 2025-10-08 (e95a9e38) rationale: the entry point uses the ordinary C calling
  convention while every chained handler uses `preserve_none`; the entry must be
  callable from Rust and a `musttail` return cannot cross a calling-convention
  mismatch, so the entry makes a normal call into the first handler and returns
  once the terminal handler unwinds (diff).

- 2025-10-08 (303bf064) rationale: dispatch combines `musttail` (forces JMP rather
  than CALL, so an arbitrarily long handler chain grows no native stack) with
  `preserve_none` (all arguments in registers, no callee-saved save/restore), which
  together keep the abstract register window and dispatch pointers live in CPU
  registers across the whole tail-chain instead of spilling between handlers (diff).

- 2025-10-08 (55c52766) rationale: the function result is recovered by reading the
  backing register-file slot holding the entry block's Return value after the
  trampoline unwinds, not by carrying it in a register out of the terminal handler;
  under preserve_none no register survives the unwind to the Rust caller, so the
  terminal handler just unwinds with no write-back (diff).

- 2025-10-20 (bb0d2820) rationale: the `musttail` + `preserve_none` tail-chained
  dispatch is explicitly inherited from the first-generation backend's
  trampoline; what changed under it is the register model, not the dispatch
  mechanism (diff).

- 2026-02-08 (da03882b) rationale: the recovered design note states why the
  handler ABI is `preserve_none` (up to 13 register args on x86-64): a tail call
  to a function with a normal prologue/epilogue kills dispatch throughput, and
  even a non-executed outgoing call from a handler forces the prologue/epilogue,
  so the hot handlers must avoid any non-`preserve_none` call and slow paths are
  pushed out of line (diff).

- 2026-02-08 (da03882b) rationale: the Return handler unconditionally refreshes
  the cached mem0 base/size on every return, not only on cross-module returns,
  because a `memory.grow` inside the callee may have reallocated and invalidated
  the base pointer even when caller and callee share a module, and the
  always-refresh cost is an O(1) lookup (diff).

- 2025-10-11 (d9c0ccce) pitfall: the first cached memory-0 base/length was a raw
  pointer into the memory's backing buffer, so any operation that can reallocate
  that buffer (memory.grow, bulk copies into memory 0) must null the cache before
  returning or later loads/stores read through a dangling pointer — the hazard
  that first led to dropping the cache entirely (diff).

## Moves

- 2025-10-13 (10a69247) dropped: the per-execution mem0 base+size pointer cache —
  the cached raw base pointer dangled whenever memory.grow reallocated the backing
  buffer, and keeping it valid forced cache-invalidation code into every bulk
  memory op; loads/stores now always go through the store's safe borrow (diff).

- 2025-09-17 (e85e902d) replaced [[regs-base-pointer-abi]]: a regs_base
  pointer-to-memory handler signature cannot carry VM values in CPU registers
  across musttail tail-calls, so a hot window of VM values v0..v3 is passed by
  value in argument registers across the whole tail-chain, the abstract working
  set staying in registers with zero prologue/epilogue between handlers (diff).

- 2026-02-07 (88b3fda7) replaced [[mem0-threaded-params]]: the XIR handler ABI
  passed the memory-0 base pointer and size as two dedicated parameters threaded
  through every preserve_none tail call (13-param wrapper convention), forcing
  each handler to carry mem0 in registers across the whole chain and
  double-indirecting them (pmem0/pmem0_len) so call/return/memory.grow could
  refresh them; moving mem0_base/mem0_size into a #[repr(C)] hot-field prefix of
  Ctx at fixed offsets 0/8 lets C handlers read ctx->mem0_base/ctx->mem0_size
  directly and frees two register slots in the calling convention (13->11
  params), with memory.grow/call/return refreshing via ctx.set_mem0 instead of
  writing back through the parameter pointers (diff).
