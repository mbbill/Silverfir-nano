- The engine carries two execution engines: the native JIT (the default,
  eager, full 3.0 feature set) and the [[interpreter]] (the folded stack
  machine, wasm 2.0 + multi-memory coverage). There is no automatic
  tiering: the engine is chosen per instance by the embedder (the CLI
  exposes it as a flag), and an instance runs entirely on one engine.

- What the two engines share is the module layer — parsing and validation,
  the opcode decoder, the value-type model — and the WASI host layer. They
  share no runtime state: the store, the instance model, and the entity
  and heap representations belong to the JIT, and the interpreter carries
  its own instance type with its own flat host-dispatch boundary.

- Neither engine touches the other's execution machinery: the interpreter
  never sees the JIT's IRs, register allocator, or encoders.

- Which engine runs a module is one selector, [[engine-selection]],
  separate from which ISA the JIT emits for ([[backend-selection]]).

- The interpreter executes only through its own native dispatch chain,
  whose handlers are generated per target at build time and linked into
  the binary; it allocates no executable memory and does not require the
  JIT subsystem to be compiled in. On a target with no generated engine,
  interpreter instantiation fails with a clean error.

- Both engines cover the same backend set, and each is validated against
  the spec suite on every one of them.

## Facts

- 2026-07-23 rationale: the interpreter tier was reopened performance-first
  for two scenarios the JIT cannot serve — platforms that forbid runtime
  code generation, and tier-0 execution while the JIT compiles — with the
  explicit constraint that no large fusion pattern library returns (size
  explosion and app-dependent coverage were its recorded costs) (sourced).

- 2026-07-23 measurement: CoreMark on Apple M-series, release, 5 runs
  each: interpreter 5454.6±48.9 vs JIT 39314.3±505 — the interpreter runs
  at 13.9% of the JIT on the same module and WASI stack (code).

- 2026-07-25 rationale: build-time handler generation is what lets the
  interpreter serve its first stated purpose, running where runtime code
  generation is forbidden or impossible (code)

- 2026-07-25 rationale: it also turns engine size from a heap allocation
  into a link-time budget, which on an MCU is flash (code)

## Moves

- 2026-07-23 replaced [[jit-only]]: a new no-fusion interpreter
  (the folded stack machine) reopened the execution tier for platforms
  where runtime code generation is impossible and for tier-0 startup,
  measured at 13.9% of the JIT on CoreMark; the JIT remains the default
  engine (sourced)

