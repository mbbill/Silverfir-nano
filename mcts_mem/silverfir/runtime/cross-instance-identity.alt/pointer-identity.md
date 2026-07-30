- An instance's identity across instance boundaries is the raw address of its
  store. Shared registry entries carry that pointer alongside a local index,
  and resolving one dereferences it (`*mut Store`).

- A dying instance is handled by a teardown protocol rather than by
  representation: dropping a store scans the shared registries and nulls every
  entry that names it. Each dereference site's safety argument rests on that
  scan having run.

- Dispatch views over the registry are cross-instance and are rebuilt when a
  revision counter says the function registry changed.

- The interpreter does not use this model. It mints its own opaque function
  reference identities, keeps a publication map to hand them out, and overloads
  the host-reference encoding to carry them. The two engines answer reference
  type tests differently for the same value.

## Moves

- 2026-07-30 (bc7cbb03) replaced by [[cross-instance-identity]]: raw store
  pointers made cross-instance identity unfalsifiable — correctness rested on a
  Drop scan poisoning registry entries and on every dereference honouring that
  contract, nothing at all handled aliasing, and the interpreter refused the
  model and grew a second funcref identity beside it (code).
