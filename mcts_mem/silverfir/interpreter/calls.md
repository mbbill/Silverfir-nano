- All frames of an invocation live on one contiguous value stack and
  overlap: a callee's frame base IS its caller's staged-argument slot;
  argument passing and result return move no values.

- Calls and returns between local functions run entirely inside the
  native chain: a native return stack holds `(return pc, frame base,
  cell base)` records; the callee path zeroes its fresh locals; return
  copies results to the frame base and pops a record.

- The driver plants a sentinel record at each activation boundary whose
  cell routes a native return back to Rust, which is how host calls and
  the invocation root compose with native calls on one shared record
  stack.

- Indirect calls resolve inside the native chain: the handler indexes a
  per-function-index callee-info table, compares one precomputed
  canonical type id, and enters the shared activation path; table 0's
  base and length come from the per-entry state, and every guard
  failure (index bounds, null entry, type mismatch, import callee)
  bails to the slow path.

- Call depth and value-stack budget are both enforced natively and trap
  as call-stack exhaustion; the value stack is fully zeroed at
  invocation start and callee locals are re-zeroed per call.

## Facts

- 2026-07-23 measurement: on one CoreMark run the predecessor paid 62.7M
  call + 62.7M return exits from the native chain, each with a heap
  allocation and two value copies; native overlapped calls removed them
  entirely and moved the score 3295 → 4124 (+25%) (code).

- 2026-07-24 rationale: type checking reduces to one integer compare
  because type equivalence is an equivalence relation — the link pass
  numbers the equivalence classes densely (linear scan over the used
  types) and stores each function's class id in the callee-info table,
  so the handler never walks type structure (code).

- 2026-07-24 rationale: refreshing table 0's base/length from the entry
  state on every chain entry makes table.grow and table.set need no
  invalidation protocol: table mutation only happens on the slow path,
  and re-entry re-reads the moved storage (code).

- 2026-07-24 measurement: native call_indirect removed the two largest
  remaining non-float exit populations — 5.9M exits on the lua json
  workload and 5.5M on sqlite speedtest — and the op no longer appears
  in any benchmark's exit profile; the exits were C-library
  function-pointer dispatch, not wasm-level indirection (code).

## Moves

- 2026-07-23 replaced [[heap-frames]]: every call paid a heap allocation,
  two value copies, and a native-chain exit and re-entry; the overlapped
  contiguous stack makes argument and result movement structural (code)
