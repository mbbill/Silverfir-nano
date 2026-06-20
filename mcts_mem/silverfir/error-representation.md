- A single fallible error type (`WasmError`) carries every failure the engine
  raises: the spec-phase variants malformed (binary decode), invalid
  (validation), unlinkable (instantiation / linking), and trap (execution); the
  engine-internal variants exhaustion, exit, and internal; plus the two
  exception-handling variants exception (uncaught wasm exception surfaced to the
  embedder) and host_throw (host-side catchable throw, VM-internal).

- The category tracks the phase at which a module is rejected, not the kind of
  the failure; the same condition is reported under whichever phase first
  detects it.

- `WasmError` is a non-`Copy` enum (`Clone`); its common variants carry
  `&'static str` messages and stay const-constructible and allocation-free,
  while two variants carry heap payloads: exception holds an
  `Option<String>` module tag name and host_throw holds a `Vec<Value>`
  (`WasmError`).

## Facts

- 2024-03-11 (e774b726) statement: the malformed/invalid split is not strictly
  decoder-vs-validator — the validator (not the decoder) rejects an unknown
  binary version, a missing datacount section for memory.init/data.drop, and a
  nonzero reserved immediate byte, yet reports all of these as Malformed even
  though they surface during semantic validation; binary.wast forced these to
  be Malformed rather than Invalid (code).

- 2024-03-13 (b622e314) pitfall: a call_indirect signature mismatch was reported
  as Invalid (a static validation error), but the spec makes it a runtime trap,
  so it must be Trap; the malformed/invalid/trap split must follow when the
  failure is detected (decode/validate vs execution), not the kind of failure
  (code).

- 2024-03-13 (055cac01) pitfall: link-time segment failures were reported as
  Invalid; an active segment with a negative offset or an out-of-bounds memory
  range is an Unlinkable failure (instantiation/linking), and a bad opcode in a
  constant expression is Invalid (validation) not Malformed (decode) — the
  category tracks the phase that rejects the module (code).

- 2026-04-22 statement: the exception-handling feature made WasmError non-Copy
  (it derives Clone) by adding two heap-carrying variants — Exception
  (Option<String>) and HostThrow (Vec<Value>); the common spec-phase and
  engine-internal variants still carry &'static str, remain const-constructible,
  and allocate nothing, so allocation is confined to the two EH variants (code).

## Moves

- 2026-04-10 (6d716c87) replaced [[boxed-string-error]]: the heap-boxed
  String-carrying error forced an allocation on every error path and could not
  be Copy or const-constructed, so error construction pulled in alloc::format
  and a Box per error even on the cold trap/validation paths; making WasmError a
  Copy enum of &'static str messages removes all allocation from the error path
  and lets errors be built in const fns, at the cost of dropping dynamic
  interpolation (offending values no longer appear in the message text) (code)
