- Dead-phi elimination removes phi nodes whose destination is in no use-set,
  where the use-set counts a value as used if it appears as an operand of any
  instruction, terminator, or other phi node's source.

- The pass iterates to a fixpoint: removing one dead phi can make another phi
  (used only as that phi's source) dead; it repeats until a round removes
  nothing.

## Moves

- 2025-11-29 (ff490da8) replaced by [[dead-phi]]: treating any value that appears as a phi source as used keeps dead phi chains and cycles (a value flowing only between phi nodes, e.g. v19 -> v411 -> v494 -> v411) alive forever; phi-aware liveness instead seeds only non-phi uses as essential and propagates liveness backward through phis to a fixpoint, so a phi is live only if its result transitively reaches a real use (diff).
