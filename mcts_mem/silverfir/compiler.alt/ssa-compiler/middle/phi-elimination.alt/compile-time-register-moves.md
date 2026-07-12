- Parallel copies are resolved at compile time into a sequence of
  register-to-register Copy / SpillStore / SpillLoad instructions, following the
  Boissinot et al. (CGO 2009) out-of-SSA algorithm.

- Dependency cycles among the copies are broken by routing one value through a
  dedicated temporary spill slot.

- Destinations and sources are allocated to physical registers, and the
  parallel-copy semantics are preserved by ordering the emitted sequential
  register moves.

## Moves

- 2025-11-12 (78e72df4) replaced by [[runtime-parallel-copy]]: resolving
  parallel copies into register move sequences with topological ordering and
  temp-slot cycle breaking was complex; spilling all live registers like a call
  and emitting a single ParCopy that operates entirely on spill slots (reading
  every source into a temp buffer before writing any destination) is correct by
  construction, leaving register-keeping as a later optimization (code).
