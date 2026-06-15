- The hottest call and return handlers (call_local_reg, return_void, return_reg)
  are written in safe Rust, like all runtime-touching handlers; only
  pure-computation operations are in C.

## Moves

- 2026-02-13 (b374bdb6) replaced by [[handler-language]]: the hottest call/return
  handlers (call_local_reg, return_void, return_reg — roughly 95% of calls) are
  reimplemented in C so they inline with zero overhead into the preserve_none
  trampoline wrappers; they manipulate the shadow stack frame, current module, and
  current function directly through a repr(C) view of Ctx, calling back into Rust
  only for the one operation that needs the store (mem0 refresh via
  xir_refresh_mem0) (diff).
