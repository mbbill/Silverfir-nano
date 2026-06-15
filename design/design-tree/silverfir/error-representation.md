- A single fallible error type (`WasmError`) carries every failure the engine
  raises, tagged with a category that names the phase that detected it:
  malformed (binary decode), invalid (validation), unlinkable (instantiation /
  linking), and trap (execution).

- The category tracks the phase at which a module is rejected, not the kind of
  the failure; the same condition is reported under whichever phase first
  detects it.

- `WasmError` is a `Copy` enum whose variants carry `&'static str` messages:
  error construction allocates nothing and errors can be built in const fns, at
  the cost of dropped dynamic interpolation — the offending value no longer
  appears in the message text (`WasmError`).

## Facts

- 2024-03-11 (e774b726) statement: the malformed/invalid split is not strictly
  decoder-vs-validator — the validator (not the decoder) rejects an unknown
  binary version, a missing datacount section for memory.init/data.drop, and a
  nonzero reserved immediate byte, yet reports all of these as Malformed even
  though they surface during semantic validation; binary.wast forced these to
  be Malformed rather than Invalid (diff).

- 2024-03-13 (b622e314) pitfall: a call_indirect signature mismatch was reported
  as Invalid (a static validation error), but the spec makes it a runtime trap,
  so it must be Trap; the malformed/invalid/trap split must follow when the
  failure is detected (decode/validate vs execution), not the kind of failure
  (diff).

- 2024-03-13 (055cac01) pitfall: link-time segment failures were reported as
  Invalid; an active segment with a negative offset or an out-of-bounds memory
  range is an Unlinkable failure (instantiation/linking), and a bad opcode in a
  constant expression is Invalid (validation) not Malformed (decode) — the
  category tracks the phase that rejects the module (diff).

## Moves

- 2026-04-10 (6d716c87) replaced [[boxed-string-error]]: the heap-boxed
  String-carrying error forced an allocation on every error path and could not
  be Copy or const-constructed, so error construction pulled in alloc::format
  and a Box per error even on the cold trap/validation paths; making WasmError a
  Copy enum of &'static str messages removes all allocation from the error path
  and lets errors be built in const fns, at the cost of dropping dynamic
  interpolation (offending values no longer appear in the message text) (diff)
