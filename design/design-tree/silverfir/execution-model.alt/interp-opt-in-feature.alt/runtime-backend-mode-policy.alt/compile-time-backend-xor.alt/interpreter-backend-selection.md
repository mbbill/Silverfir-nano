- A single process-global, u8-tagged atomic switch (`InterpreterBackend`)
  selects among the classic, fast, and SSA backends; it is read at each
  function evaluation, leaving a loaded module committed to no backend.

- The switch is a runtime global, not a build-time feature gate: all backends
  stay linked in one binary and the selection is changed per run without
  recompilation, set once by the embedder before execution.

- The SSA backend is the default when no explicit selection is made.

- All three backends present the same per-function entry: given a function
  instance, store, and argument words, they run the body and leave its result
  words on a value stack the caller reads back.

- Each backend keeps its own internal code form, built lazily on first call to a
  function (with an optional eager build at instantiation); loading a module
  commits to none of them.

## Facts

- 2025-08-14 (11d99815) rationale: the backend is chosen by a runtime global
  (an atomic toggled from the CLI --baseline flag) read inside function
  evaluation rather than a build-time feature gate, so both interpreters stay
  linked in one binary and are switchable per run without recompilation (diff).

- 2025-10-05 (0cad057e) statement: the global default backend was switched from
  the SSA-compiling backend to the classic in-place interpreter at this point
  (the AtomicU8 switch initializes to Classic); the default is moved back to SSA
  later in history (diff).

- 2025-10-26 (d631247b) statement: the global default backend was changed from
  the Classic in-place interpreter to the SSA-compiling backend, making SSA the
  backend used when no explicit selection is made (diff).

- 2026-06-14 statement: the default flips between SSA and Classic track which
  engine is being trusted at the time — Classic is the easy-to-get-correct
  oracle, so the default is moved to Classic while a tricky feature is being
  brought up (a green Classic isolates the fault to SSA) and moved back to SSA
  once SSA handles that feature; the flips are a development convenience riding
  on Classic's oracle role, not a standing preference for either engine (author).

## Moves

- 2025-10-01 (04122214) replaced [[two-way-boolean-switch]]: a single AtomicBool
  could only choose between two interpreters (fast vs the classic inplace
  baseline) and could not name a third; a u8-tagged backend enum lets function
  evaluation select among the classic, fast, and new SSA backends per process,
  and makes the SSA backend the default with fast/classic as explicit overrides
  (diff).

- 2026-02-14 replaced by [[compile-time-backend-xor]]: the -rs runtime u8 switch
  selected among three coexisting interpreter backends (classic, fast, SSA) per
  process because -rs needed a runtime oracle to cross-check engines; -nano needs
  no such oracle-switch — with no real register allocator debugging is easy
  without one, the fast interpreter was ported from -rs already correct, and
  -nano ran wasm 2.0 for most of development so the wasm core was stable
  (failures point to new features, not the base) — so the surviving single engine
  selects its execution tiers (micro-jit, fusion) by a compile-time feature gate
  instead of a runtime backend switch (author).
