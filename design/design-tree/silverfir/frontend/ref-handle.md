- WebAssembly values carry their reference identity through a tagged handle
  (`RefHandle`) whose bit layout distinguishes null, special, host, and pool
  references, and which is sized to the target pointer width.

- The layout uses a SPECIAL_TAG plus an EXTERN_TAG in the high bits and a
  pool-payload bit inside the payload; one handle distinguishes null, special,
  host, extern, and pooled (i31 / GC) references and routes pooled handles to the
  per-store ref registry.

## Facts

- 2025-10-06 (3ccf0d6f) rationale: the extern hierarchy is an orthogonal tag
  bit (bit 61) on the same RefHandle, so extern.convert_any and any.convert_extern
  are implemented by toggling that bit on the existing handle rather than
  allocating or wrapping; only GC (struct/array) and i31 references may carry
  the extern tag, while funcref and exnref are disjoint hierarchies and cannot
  be converted (diff).

- 2025-10-06 (4e6eb23b) limitation: any.convert_extern / extern.convert_any
  keep the reference in the same address space and only flip the extern tag
  bit, so a value round-trips losslessly (any->extern->any) but there is no
  bridge to a real host/embedder representation; a host-backed embedder would
  have to convert internal GC/i31 values to and from the host's own object
  representation instead of reusing the handle (diff).

- 2025-10-07 (94652766) rationale: the four-bit tag (bits 60-63: i31, GC,
  extern-hierarchy, host) makes host-origin and extern-hierarchy orthogonal, so
  a host value wrapped as extern and a GC object retagged as extern coexist: the
  host bit is what lets a cast distinguish an opaque host value (matches any
  only) from a GC value (matches eq/i31/struct/array) after any.convert_extern
  (diff).

- 2026-04-16 (9ff58dcd) statement: when generated code uses 32-bit GP slots on
  a wider host, a reference cannot be stored by reusing the pointer-width
  RefHandle bit layout directly (the host tag bits sit above bit 31); the shared
  value-encoding layer defines a separate compact 32-bit ref encoding
  with its own special/extern/pool tag bits (TARGET32_REF_*) that preserves the
  null/special/host/pooled/extern split in 32 bits, and routes refs to/from
  RawValue through RefHandle::encoded() instead of the raw field (diff).

## Moves

- 2025-10-04 (eac9a06d) replaced [[plain-index-ref]]: a plain usize index
  carried only a null sentinel and could not distinguish a GC-heap reference
  from an inline i31 value from a funcref, nor encode an i31 payload inline;
  tagging the high bits lets one word carry all reference kinds without a
  separate type word (diff).

- 2026-04-16 (9ff58dcd) replaced [[two-way-tag-layout]]: the two-way
  host/extern tag layout could not express the GC/i31 reference class introduced
  by wasm 3.0 — it had only a host bit and an extern bit and no encoding for a
  store-pooled reference; the layout is re-cut into a SPECIAL_TAG plus an
  EXTERN_TAG plus a pool_payload bit inside the payload so one RefHandle
  distinguishes null, special, host, extern, and pooled (i31/GC) references and
  routes pooled handles to the per-store ref registry (diff).
