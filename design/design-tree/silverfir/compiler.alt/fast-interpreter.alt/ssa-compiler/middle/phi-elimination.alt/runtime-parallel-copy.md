- A parallel copy between spill slots (ParCopy) carries a list of (dst, src)
  slot pairs with parallel semantics: the runtime handler reads every source
  into a temporary buffer before writing any destination, leaving all copies
  observing the pre-copy slot values.

- The atomicity required by parallel semantics is enforced at run time by the
  handler's two-phase read-then-write, allocating a temporary buffer sized to the
  pair count on each invocation.

## Moves

- 2025-11-12 (78e72df4) replaced [[runtime-parallel-copy.alt/compile-time-register-moves]]:
  resolving parallel copies into register move sequences with topological
  ordering and temp-slot cycle breaking was complex; spilling all live registers
  like a call and emitting a single ParCopy that operates entirely on spill slots
  (reading every source into a temp buffer before writing any destination) is
  correct by construction, leaving register-keeping as a later optimization
  (diff).

- 2025-11-24 (02756381) replaced by [[phi-elimination]]: the runtime
  parallel-copy handler allocated a temporary Vec on every execution to read all
  sources before writing any destination; moving that work to compile time — a
  solver that topologically orders the copies and breaks cycles with one reserved
  frame temp slot — lets the runtime handler just execute the pre-ordered copies
  in place with no allocation (diff).
