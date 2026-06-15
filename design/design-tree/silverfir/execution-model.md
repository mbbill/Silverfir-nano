- The execution-backend enum carries a single kind: every Wasm function is
  compiled to native code and there is no interpreter or fusion tier to select
  among at build or run time (`BackendKind::Native`).

## Facts

- 2026-06-14 rationale: the kill-fact for dropping the interpreter tier and
  going JIT-only is the ceiling gap — the fast interpreter peaks at roughly
  Cranelift's baseline (Winch) tier, about as fast as an interpreter can get,
  while the JIT exceeds Cranelift's optimizing JIT and reaches V8-class; keeping
  an interpreter tier that can never close that gap was not worth its weight once
  the JIT was working (author).

## Moves

- 2026-04-07 (38809e62) replaced [[interp-opt-in-feature]]: the interpreter
  (base) and fusion execution tiers had already been shelved behind disabled
  cargo features and the native JIT carried all execution; this commit deletes
  the entire interp subsystem, the fast-interp C-handler/trampoline build
  pipeline, and their backend enum variants, collapsing the execution-backend
  enum to a single Native kind and making the engine JIT-only (diff)
