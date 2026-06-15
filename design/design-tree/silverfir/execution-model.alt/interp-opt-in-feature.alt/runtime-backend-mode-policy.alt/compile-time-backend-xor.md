- The fast interpreter requires micro-jit and fusion to be mutually exclusive
  features: enabling both is a `compile_error!`.

- The backend is fixed at compile time by which of micro-jit / fusion features is
  enabled; the only runtime control is a global boolean that disables fusion
  (used by discover-fusion profiling to see the raw unfused instruction stream).

## Moves

- 2026-02-14 replaced [[interpreter-backend-selection]]: the -rs runtime u8 switch
  selected among three coexisting interpreter backends (classic, fast, SSA) per
  process because -rs needed a runtime oracle to cross-check engines; -nano needs
  no such oracle-switch — with no real register allocator debugging is easy
  without one, the fast interpreter was ported from -rs already correct, and
  -nano ran wasm 2.0 for most of development so the wasm core was stable
  (failures point to new features, not the base) — so the surviving single engine
  selects its execution tiers (micro-jit, fusion) by a compile-time feature gate
  instead of a runtime backend switch (author).

- 2026-03-06 (7f69e7af) replaced by [[runtime-backend-mode-policy]]: the old
  design forced micro-jit and fusion to be mutually exclusive at compile time
  and only offered a binary on/off fusion-disable knob, so a single build could
  not carry all backends or pick among them per run; BackendMode lets jit,
  fusion, and base coexist in one build and be selected at runtime, with Auto
  resolving to the highest-priority compiled backend and falling back further on
  runtime failure (e.g. JIT arena allocation) (diff)
