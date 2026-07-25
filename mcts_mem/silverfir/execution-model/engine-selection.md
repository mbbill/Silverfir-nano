- One selector names which engine runs a module, and its cases are exactly
  the engines the build has. Naming an engine that was left out is a
  compile error at the embedder's call site, not a runtime failure
  (`Engine`).

- A build with a single engine makes the selector a zero-sized value. It
  carries no storage, and every switch on it folds to its one arm.

- An embedder that names no engine gets the JIT where it is compiled in
  and the interpreter otherwise.

- The engine axis is separate from the ISA axis. Which machine the JIT
  emits for is [[backend-selection]], is fixed at build time, and is
  reported through its own accessor rather than folded into the engine
  value.

## Facts

- 2026-07-25 measurement: an embedder that names the JIT's instance type
  unconditionally links the entire JIT-side runtime substrate into an
  interpreter-only binary. Gating that one call site moved the stripped
  CLI from 1,147,184 to 1,081,024 bytes; by symbol origin the JIT-side
  `vm/` went 41,964 -> 3,824 bytes and its unwind tables 131,032 ->
  112,248 (code).

- 2026-07-25 pitfall: a zero-sized selector does nothing on its own — the
  saving comes from the embedder's arms being gated on the same features,
  because a call site that names both engines links both (code).

- 2026-07-25 statement: the residue after gating is drop glue for the
  import types, which the interpreter path constructs too, so it is shared
  use rather than JIT code that survived (code).

## Moves

- 2026-07-25 replaced [[backend-mode]]: three separate types described one
  choice and none of them could name the interpreter, so an
  interpreter-only build reported its engine as unavailable instead of as
  the interpreter, and no build could fold the choice away (code)
