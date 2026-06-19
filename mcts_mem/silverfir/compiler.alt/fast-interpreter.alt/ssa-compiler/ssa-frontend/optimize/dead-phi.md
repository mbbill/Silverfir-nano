- Dead-phi elimination removes phi nodes whose result never transitively
  reaches a real use: phi-aware liveness seeds only non-phi uses as essential
  and propagates liveness backward through phis to a fixpoint; a phi is live
  only if a non-phi use reads it (directly or through other phis).

## Facts

- 2025-11-24 (bd443f1f) rationale: dead-phi elimination is needed because SSA
  construction inserts phi nodes conservatively for every local at loop
  headers, even locals never read after the loop; without this cleanup those
  never-used phis survive into phi elimination and become wasted parallel
  copies and spill slots (code).

## Moves

- 2025-11-29 (ff490da8) replaced [[use-set-liveness]]: treating any value that appears as a phi source as used keeps dead phi chains and cycles (a value flowing only between phi nodes, e.g. v19 -> v411 -> v494 -> v411) alive forever; phi-aware liveness instead seeds only non-phi uses as essential and propagates liveness backward through phis to a fixpoint, so a phi is live only if its result transitively reaches a real use (code).
