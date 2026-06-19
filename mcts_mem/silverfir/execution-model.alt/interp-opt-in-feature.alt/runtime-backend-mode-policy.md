- Backend selection is a runtime policy resolved against compiled availability
  (BackendMode Auto/Base/Fusion/Native): Auto prefers native, then fusion, then
  base, with graceful fallback; the interpreter (Base) is always compiled in and
  Auto falls back to the proven base pipeline.

## Facts

- 2026-03-06 (1f205034) statement: the backend-mode policy is moved wholesale out
  of interp/fast/mod.rs into a sibling vm/backend.rs module, and at the same step
  the former BackendMode::Jit / BackendKind::Jit variants and the JitEmitter type
  are renamed to Native (the 'jit' policy string kept only as a parse alias) — the
  naming pivot that recasts the embedded JIT as a first-class native backend
  selectable alongside base and fusion, before the engine later collapses to
  native-only (code).

## Moves

- 2026-03-06 (7f69e7af) replaced [[compile-time-backend-xor]]: the old design
  forced micro-jit and fusion to be mutually exclusive at compile time and only
  offered a binary on/off fusion-disable knob, so a single build could not carry
  all backends or pick among them per run; BackendMode lets jit, fusion, and
  base coexist in one build and be selected at runtime, with Auto resolving to
  the highest-priority compiled backend and falling back further on runtime
  failure (e.g. JIT arena allocation) (code)

- 2026-03-12 (61b3fac8) replaced by [[interp-opt-in-feature]]: the
  interpreter/fast-interp backend, its build.rs codegen (handlers.toml,
  preserve_none ABI check, C trampoline), and the Base backend kind are moved
  behind a non-default `interp` feature so a build without it has no interpreter
  tier at all and Base resolution returns an error; this gates the legacy
  execution tiers off by default as the native MachineIR backend becomes the
  intended execution path (code).
