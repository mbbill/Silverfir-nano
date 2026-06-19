- The C-versus-Rust split for handler bodies is drawn on a hot-path-versus-cold-
  path line, not a pure-computation-versus-runtime-touching line: pure-computation
  operations are in C, and the hottest runtime-touching handlers (the common
  call/return forms, slot copies) are also in C, reaching directly into runtime
  state through a `#[repr(C)]` view of the context and calling back into Rust only
  for the rare operation that needs the store; only cold runtime paths stay in
  Rust.

- A runtime-touching handler written in C reads the runtime state it needs off the
  shared `#[repr(C)]` context (spill base/top, current module/function, mem0
  base/size) at fixed offsets, with the shadow-stack frame layout mirrored into
  the trampoline header, letting the C side manipulate frames without a Rust call.

## Facts

- 2025-11-30 (2a41e65f) rationale: memory load/store are runtime-touching yet were
  moved into C with a split path — memory index 0 (the overwhelmingly common case)
  is handled inline in C through the direct memory pointer, while a non-zero index
  falls back to a Rust helper, so only the cold path pays the language-boundary
  cost (code).

- 2025-11-30 (68ee614e) rationale: basic spill load/store moved to C because they
  are pure pointer operations, but register-to-register copy and multicopy (the
  parallel-copy / phi-elimination handlers) stay in Rust because they select source
  and destination register dynamically by index, which the per-permutation C
  handlers do not express (code).

- 2026-02-11 (8f12b0e9) rationale: slot-to-slot copy operations were later moved
  from Rust to C because they are pure spill-array pointer manipulation touching no
  Rust runtime state, so a C leaf handler incurs zero FFI overhead — the same
  in-C-when-pure split the backend applies to spill load/store (code).

- 2026-02-12 (892cfa72) rationale: load/store lowering emits a memory-0-
  specialized handler whenever the static memory index is 0 (the dominant
  single-memory case): the general handler decodes the index, branches on it, and
  carries a multi-memory slow path, whereas the mem0 variant takes the offset
  directly and accesses the cached mem0 base with none of that overhead (code).

- 2026-02-13 (b374bdb6) statement: moving the hot call/return handlers to C
  required widening the C-visible context struct to expose spill base/top, store,
  and current function, and exporting the shadow-stack frame layout into the
  trampoline header, so the language boundary for a handler is no longer
  pure-computation-vs-runtime-touching but hot-path-vs-cold-path, with the C side
  reaching directly into runtime state (code).

## Moves

- 2026-02-13 (b374bdb6) replaced [[hot-handlers-in-rust]]: the hottest call/return
  handlers (call_local_reg, return_void, return_reg — roughly 95% of calls) are
  reimplemented in C so they inline with zero overhead into the preserve_none
  trampoline wrappers; they manipulate the shadow stack frame, current module, and
  current function directly through a repr(C) view of Ctx, calling back into Rust
  only for the one operation that needs the store (mem0 refresh via
  xir_refresh_mem0) (code).
