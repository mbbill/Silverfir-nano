- Every semantic op carries optional generic control side-channels: a next
  field for the fallthrough successor and an alt field for the branch successor;
  If/Else/Br/BrIf validate by checking these optional fields are populated, and
  a br_table entry's branch target is an Option validation checks per entry.

## Moves

- 2026-03-09 (ab127bb7) replaced [[impure-semantic-ir]]: the old semantic layer
  leaked stack-machine and TOS-cache state (variant, pre_height, spill/fill)
  forward into the backend, forcing backend codegen to deduce register behavior
  from stack metadata; purifying semantic IR pushes that policy down into a
  dedicated planning stage so each later pass reasons about loops/calls/locals
  without reconstructing them from low-level code (diff).

- 2026-03-12 (2ea0bb68) replaced by [[semantic-ir]]: a generic next/alt side
  channel on every semantic op duplicated fallthrough into every instruction and
  could not distinguish a branching op from a straight-line one, forcing later
  preparation to rediscover control meaning from generic side fields; making
  fallthrough implicit by order and attaching explicit targets only to branching
  ops (if's false target, else's end target, br/br_if/br_table targets) keeps
  the semantic layer honest — only branching ops carry control targets (diff)
