- Validation runs per function over the same streaming decode walk: a
  per-function validator is registered as an `OpcodeHandler` and type-checks
  each instruction as it is decoded, so there is no separate validation pass
  over a materialized instruction list (`FunctionValidator`). Imported
  functions are skipped; only locally-defined function bodies are validated.

- Type-checking is the spec's abstract-stack algorithm: a value stack of value
  types and a stack of control frames are carried in a context (`Context`).
  Each instruction pops the operand types it consumes and pushes the result
  types it produces; a mismatch is an invalid-module error.

- A control frame records its kind (function, block, loop, if, else), its
  start/result types as a function type, the value-stack height at frame entry,
  and an unreachable flag (`ControlFrame`). The recorded entry height is what
  bounds pops: popping below the current frame's height is a stack underflow.

- Stack-polymorphism after an unreachable point is modeled with a distinct
  "unknown" value type: once a frame is marked unreachable, popping from an
  empty-down-to-frame-height stack yields `unknown`, which type-checks against
  any expectation. Branches, `return`, and `unreachable` mark the current frame
  unreachable.

- Operand expectations are expressed as type predicates, not exact types: a pop
  takes a predicate over value types, and a per-type predicate (plus
  any/num/ref predicates) decides acceptance, with `unknown` accepted by every
  predicate. This lets the polymorphic and reference-family checks share one
  pop path.

- A block type immediate is resolved to a concrete function type before
  type-checking the block: the empty and inline-value-type forms map directly,
  and a type-index form indexes the module's function-type vector.
