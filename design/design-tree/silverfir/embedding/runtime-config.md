- Runtime sizing (executable code-arena size, linear-memory page ceiling, Wasm
  operand/call stack size) comes from a write-once global the embedder installs
  before any instance, not from compile-time constants (`RuntimeConfig`).

- The hosted default reproduces the former fixed numbers, while the bare-metal
  (`sf_os_none`) default is zeroed; a call site that reads an unconfigured
  size returns a clean not-configured error rather than allocating against a
  zero size (`runtime_not_configured`).

## Facts

- 2026-04-21 (a80b238f) rationale: the Wasm operand/call stack, previously the
  compile-time constant constants::MAX_STACK_SIZE (a fixed 2 MiB), is folded
  into the same write-once runtime config as RuntimeConfig.wasm_stack_bytes: the
  constant is removed and both eval paths now size the per-invoke u64 stack from
  runtime_config().wasm_stack_bytes; the hosted default preserves the former
  2 MiB while the bare-metal (sf_os_none) default is zeroed, so a
  stack_slots == 0 guard returns runtime_not_configured rather than allocating a
  zero-length stack when the embedder forgot to install a config (diff).

## Moves

- 2026-04-21 (b206d2aa) replaced [[hardcoded-runtime-sizes]]: the bare-metal
  target cannot fit the hosted defaults (12-16 MiB code arena, wasm32 page
  ceiling) into a few hundred KiB of SRAM, so the fixed constants are replaced
  by a write-once global the embedder installs before any instance, with hosted
  defaults preserving the old numbers and the bare-metal default zeroed to force
  a clean error (diff).
