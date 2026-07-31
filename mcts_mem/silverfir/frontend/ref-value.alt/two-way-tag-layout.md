- RefValue is a pointer-width tagged index whose only tags are HOST_TAG and
  EXTERN_TAG (high bits, with a TAG_MASK over both): is_host tests the host bit
  and is_extern tests the extern bit, externref sets both, and raw_value strips
  the tag mask.

- Reference identity recognises exactly two non-null classes — host and extern —
  with no encoding for a store-pooled (i31 / GC) reference.

## Moves

- 2026-04-16 (9ff58dcd) replaced by [[ref-value]]: the two-way host/extern tag
  layout could not express the GC/i31 reference class introduced by wasm 3.0 —
  it had only a host bit and an extern bit and no encoding for a store-pooled
  reference; the layout is re-cut into a SPECIAL_TAG plus an EXTERN_TAG plus a
  pool_payload bit inside the payload so one RefValue distinguishes null,
  special, host, extern, and pooled (i31/GC) references and routes pooled handles
  to the per-store ref registry (code).
