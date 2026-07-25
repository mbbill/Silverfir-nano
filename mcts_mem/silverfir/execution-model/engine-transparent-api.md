- One instance type covers both engines. Instantiating, binding imports,
  and calling an export by name with typed values are written once and
  compile against whichever engine the build has (`Instance`).

- The engine's inner representations are gated on the engine features. A
  single-engine build carries a univariant wrapper: no discriminant, and
  every dispatch inside it folds to one arm.

- Import declarations belong to neither engine. The interpreter's raw
  host boundary — u64 operands, memory as a slice, import named by
  (module, name) — is adapted from those declarations inside the engine.

- Each engine keeps an escape hatch for what only it can answer: the
  JIT's entity model, the interpreter's dispatch statistics. Reaching one
  returns nothing when another engine is running.

- An instance owns its module rather than borrowing it.

- An export can be resolved once into a handle and called repeatedly,
  with results written into a caller-owned slice. Calling by name stays
  available and does both lookups itself.

## Facts

- 2026-07-25 measurement: engine transparency costs 5,312 bytes of flash
  on the Cortex-M33 firmware (303,136 -> 308,448), which is the typed
  import path — refcounted host callbacks, value conversion, and the
  signature lookup — linking into a build that previously drove the raw
  boundary directly (code).

- 2026-07-25 pitfall: a borrowed module forces every embedder to
  manufacture a lifetime that outlives the instance, and the spec runner
  did it by leaking one module per instantiation. Nothing referenced the
  module from a predecoded function, so the borrow bought nothing (code).

- 2026-07-25 pitfall: reading an import's names through `&self` while the
  dispatcher is held through `&mut self` forced a heap-allocated copy of
  both names on every host call; the fields are disjoint and borrowing
  them separately removes it (code).

- 2026-07-25 rationale: dispatch resolves an import by scanning the bound
  list rather than hashing, because import lists are short and a hash map
  is a poor trade in an engine that has to run without an allocator's
  worth of headroom (code).

- 2026-07-25 statement: the adapter carries numeric values only, and a
  host import whose signature is not numeric is rejected when it is bound
  rather than when it is called (code).

- 2026-07-25 measurement: resolving an export once and writing results
  into a caller-owned slice cuts JIT call overhead from 97.3 ns to 47.7 ns
  on a trivial function (2.04x), and interpreter overhead from 1745.7 ns
  to 1604.8 ns (1.09x) (code).

- 2026-07-25 rationale: the interpreter gains little because its per-call
  cost is not the lookup — it allocates and zeroes a fixed 2 MiB operand
  stack inside every invocation, which dwarfs everything else at this
  scale (code).

- 2026-07-25 pitfall: holding a function's signature across the
  invocation requires cloning it, and on a short function that clone
  costs more than the call; reading the signature after the call instead
  is what turned a measured slowdown into a gain (code).

- 2026-07-25 statement: host state reaches an import callback by being
  captured, since an import may be any owning closure; no separate
  store-data channel is needed for state that one callback owns (code).

## Moves

- 2026-07-25 replaced [[per-engine-instance-types]]: an embedder had to
  write one code path per engine and hand-roll the interpreter's raw host
  boundary itself, which every embedder in the tree had duplicated (code)
