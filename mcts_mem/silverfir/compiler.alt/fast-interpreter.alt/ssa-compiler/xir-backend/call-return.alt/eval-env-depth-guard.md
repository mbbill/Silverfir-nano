- Recursion depth is tracked by a single per-evaluation environment created once
  per top-level evaluation and threaded into every nested frame as a raw pointer
  (`EvalEnv`); entering a call increments the counter and returns an RAII guard
  whose Drop decrements it, and the limit is checked against the environment's
  maximum.

## Facts

- 2025-10-12 (2efc9d66) rationale: the backend executes a Wasm call by recursively
  re-entering the evaluator from inside the call op handler, reusing the native
  call stack instead of managing explicit frames; because each Wasm frame consumes
  a real native frame, a software depth limit is required to trap exhaustion
  before the native stack overflows, and the limit is kept conservative for that
  reason (lowered 65536 -> 1024 -> 512 to fit under the native stack) (code).

## Moves

- 2025-10-15 (0d24ab09) replaced by [[call-return]]: EvalEnv was a separate heap
  structure threaded by NonNull across frames with a Drop guard to unwind the
  counter, which forced the unsafe eval_env pointer round-trip; carrying
  call_depth as a plain usize on Ctx and passing depth+1 into the callee's
  eval_internal removes the shared structure and its guard entirely (code).
