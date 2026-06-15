- The JIT keeps two distinct runtime-boundary systems rather than one generic
  helper mechanism: a runtime-call system for Wasm call sites that must
  round-trip through runtime dispatch, and a preserved-helper system for
  engine-internal helper-backed operations.

- The runtime-call system handles `call` / `call_indirect` / `call_ref` sites
  that cannot transfer directly into another compiled MachineIR frame —
  including host/WASI targets — is lowered as an inline MachineIR runtime call,
  and uses frame slots as its argument and result transport (`runtime_call`).

- The preserved-helper system handles engine-internal ops such as
  `memory.grow`, `memory.copy`, `table.grow`, `table.init`, `ref.test`,
  `ref.cast`, and `struct` accessors; it is owned by the native backends rather
  than by MachineIR and uses a fixed native-stack I/O layout (`preserved`).

- Direct compiled-to-compiled Wasm calls are a MachineIR `Call` form (direct or
  runtime-resolved-but-compiled target); only the round-trip path is
  `CallRuntime`. Preserved helpers are never a MachineIR call form — they are
  backend lowering choices for ordinary MachineIR instructions.

- Foreign C-ABI argument/result registers are boundary-only facts usable only
  inside a runtime-call or preserved-helper entry sequence, after MachineIR
  state has been made unavailable there; regular lowering must not treat them
  as general temporaries.

- Every foreign-boundary runtime entry crossed from generated code is a Rust
  `extern "C"` entrypoint that returns a `u32` status (0 ok / nonzero error): a
  failing entry stashes its `WasmError` in the runtime context and returns the
  error status rather than using multi-value returns.

## Facts

- 2026-03-06 (37c40ffe) rationale: the native backend's model is direct
  native-entry addressing — every kept instruction conceptually has a native
  entry address, which lets nh disappear (freeing a register), makes direct
  code-to-code chaining natural, and confines bridge stubs to cold transitions
  only (host/import calls, call_indirect, memory.grow, trap slow paths) so the
  normal-ABI prologue/epilogue cost never sits on the hot path (diff).

- 2026-03-07 (2df6a982) rationale: hot native code must execute by direct
  code-to-code control flow (no interpreter-style pc, no Instruction-stream
  dispatch, no handler pointers); bridge stubs on the hot path would reintroduce
  the normal-ABI prologue/epilogue cost per opcode and defeat the native backend,
  so the custom native VM ABI stays internal to generated code and cold wrappers
  cross to Rust helpers via a uniform platform-ABI signature, with wrapper
  metadata owned by the compiled native artifact (lifetime = native code
  lifetime) (author).

- 2026-04-01 (2a753247) rationale: static ABI/layout metadata (call-link layout,
  per-function frame regions) is split out of the executable instruction
  vocabulary into a separate MachineModuleAbi record; read-only external-call
  metadata is interned (dedup by encoded bytes) into the machine module constant
  pool so the same payload is not repeated in every instruction and the ISA layer
  treats it as an opaque const reference (diff).

- 2026-04-01 (2a753247) statement: the runtime boundary is split into two sibling
  modules that share only low-level status/error plumbing — runtime/external/
  (Wasm calls resolving to external/host handlers, lowered by MachineIR as an
  inline runtime call using frame slots for arg/result transport) and
  runtime/preserved/ (engine-internal helper-backed ops like memory.grow/copy and
  table.grow/init, owned by the native backends, using a fixed native-stack I/O
  window with op-code dispatch through one preserved_entry) (diff).

- 2026-04-01 (db81af27) rationale: caller-saved does not mean free-right-now — a
  backend may treat a caller-saved physical register as disposable only if it is a
  dedicated scratch-pool register or it is inside a boundary protocol that has
  already made the relevant MachineIR state unavailable; foreign C_ARG*/C_RET*
  registers are boundary-only facts (usable while entering the external-call or
  preserved-helper runtime entry) and must not be used as general temps in regular
  lowering, because they overlap mapped transient/cache registers (author).
