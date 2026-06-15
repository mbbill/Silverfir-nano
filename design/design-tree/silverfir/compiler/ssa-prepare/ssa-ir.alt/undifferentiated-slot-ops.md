- A LoadSlot op reads a value from a canonical frame slot into an SSA value and a
  StoreSlot op writes an SSA value into a canonical frame slot; the same two ops
  serve both local access and deep-stack spill/fill, with no version tag and no
  distinction between the two roles (`LoadSlot`, `StoreSlot`).

## Moves

- 2026-03-26 (98de6d7b) replaced by [[ssa-ir]]: a single LoadSlot/StoreSlot pair
  could not distinguish canonical-local access from deep-stack spill/fill nor
  carry the semantic local version that sink planning needs to prove the old
  version of a local is dead before redirecting a producer into the local's home
  (diff).
