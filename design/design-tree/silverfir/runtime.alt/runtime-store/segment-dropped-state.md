- An element or data instance encodes its dropped state by the emptiness of its
  materialized segment store (`is_dropped`); instantiation materializes the
  passive segment contents (active and declarative segments are applied/validated
  then dropped immediately), and dropping clears that store.

## Moves

- 2025-10-11 (653fe38b) replaced [[segment-dropped-state.alt/boolean-dropped-flag]]:
  a passive segment must retain its materialized references/bytes so later
  table.init/memory.init can read them, so the materialized store has to exist
  regardless and its emptiness already encodes dropped-ness, making a separate
  boolean flag redundant (diff).
