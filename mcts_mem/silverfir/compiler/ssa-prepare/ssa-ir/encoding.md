- Each SSA-IR instruction is a flat fixed-size 16-byte record (`SsaInst`) with
  packed 4-byte operands; per-op payloads — primitive ops, constants, call ops —
  are interned into program-level pools rather than carried inline, while
  overflow args live block-level in `SsaBlock.extra_args` (indexed by
  `SsaInst.meta`), keeping `SsaBlock.ops` cache-dense.

- Primitive-pool IDs remain first-seen insertion-order IDs, while a separate
  half-full open-addressed hash table maps a primitive value back to its pool
  ID. Hash collisions are resolved by probing plus full `PrimitiveOpKind`
  equality; hashes are absent from the IR encoding.

## Facts

- 2026-04-13 (782a6dfb) constraint: the instruction opcode is encoded in a u16
  (non-primitive variants 0..=9, primitive-pool indices shifted by
  PRIMITIVE_BASE=10), which caps the number of distinct primitive ops a single
  function may hold at u16::MAX-PRIMITIVE_BASE+1; interning past that ceiling
  returns an internal error rather than overflowing the opcode (code).

- 2026-06-20 statement: overflow args are not program-level interned — they live
  block-level in `SsaBlock.extra_args` (indexed by `SsaInst.meta`) and are not
  deduped; only const_pool / primitive_pool / call_ops are program-level interned
  (the 2026-04-13 move's "extra args" listing is imprecise) (code).

- 2026-07-23 (2598d4da) measurement: the sorted primitive side index made every
  emitted primitive binary-search and repeatedly order-compare the large
  payload enum; the comparison closure alone held 103 serial FFmpeg samples.
  A deterministic open-addressed side index reduced `rewrite_function` from
  1,131 to 999 samples (11.7% absolute) and the whole sampled compile from
  5,947 to 5,756 samples (3.2%). Exact-parent ABBA startup medians averaged
  5.570 s for the hash index versus 5.676 s for the sorted index (1.9% faster);
  fat-LTO text grew only 220 bytes. FFmpeg's complete native index remained
  byte-identical and all 357 release tests passed
  ([[compiler.fact/startup-campaign-2026-07-22]]) (sourced).

## Moves

- 2026-04-13 (782a6dfb) replaced [[enum-variant]]: the per-variant enum carried
  heap-allocated arg/result vecs and inline 64-bit constants on every op,
  bloating the block op stream; a flat fixed-size record with payloads interned
  into program-level pools (primitive ops, constants, call ops, extra args) and
  packed 4-byte operands keeps SsaBlock.ops cache-dense and shrinks SSA-IR
  memory (code)
