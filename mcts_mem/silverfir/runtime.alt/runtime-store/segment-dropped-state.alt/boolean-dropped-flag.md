- An element or data instance carries only a boolean dropped flag
  (`Cell<bool>`); instantiation does not materialize passive-segment contents,
  and dropping just sets the flag.

## Moves

- 2025-10-11 (653fe38b) replaced by [[segment-dropped-state]]: a passive segment
  must retain its materialized references/bytes so later table.init/memory.init
  can read them, so the materialized store has to exist regardless and its
  emptiness already encodes dropped-ness, making a separate boolean flag
  redundant (code).
