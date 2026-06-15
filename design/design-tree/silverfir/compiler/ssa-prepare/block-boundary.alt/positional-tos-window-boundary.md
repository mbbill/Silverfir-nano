- Each LIR block declares a positional boundary: a full logical stack_height on
  entry plus a bottom-to-top vector of live TOS-window values (deeper values
  already published to canonical operand slots).

- Each LIR edge carries the successor's stack_height and its outgoing live
  TOS-window values positionally, bottom-to-top.

- Block-boundary validation checks that an edge's stack_height and TOS-window
  length match the target block's declared stack_height and TOS length.

## Moves

- 2026-03-12 (455661a0) replaced by [[block-boundary]]: a positional stack-order
  block contract forced edges to remap branch payload as live boundary SSA,
  reintroducing hidden stack policy that disagrees with the canonical slot-based
  branch layout; replacing it with successor-declared live params plus explicit
  edge bindings lets taken branch payload travel through canonical operand slots
  (published during frontend preparation, reloaded by the target's prepared
  prefix) and lets backend lowering reconcile bindings into real registers or
  moves without a positional contract (diff).
