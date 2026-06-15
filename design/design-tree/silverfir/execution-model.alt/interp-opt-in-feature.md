- The interpreter (Base) backend is an opt-in non-default build feature; without
  it the engine reports no interpreter backend compiled in and Base resolution
  returns an error, while the native MachineIR backend is the intended execution
  path.

## Moves

- 2026-03-12 (61b3fac8) replaced [[runtime-backend-mode-policy]]: the
  interpreter/fast-interp backend, its build.rs codegen (handlers.toml,
  preserve_none ABI check, C trampoline), and the Base backend kind are moved
  behind a non-default `interp` feature so a build without it has no interpreter
  tier at all and Base resolution returns an error; this gates the legacy
  execution tiers off by default as the native MachineIR backend becomes the
  intended execution path (diff).

- 2026-04-07 (38809e62) replaced by [[execution-model]]: the interpreter (base)
  and fusion execution tiers had already been shelved behind disabled cargo
  features and the native JIT carried all execution; this commit deletes the
  entire interp subsystem, the fast-interp C-handler/trampoline build pipeline,
  and their backend enum variants, collapsing the execution-backend enum to a
  single Native kind and making the engine JIT-only (diff)
