- Each SSA-IR instruction is a flat fixed-size 16-byte record (`SsaInst`) with
  packed 4-byte operands; per-op payloads — primitive ops, constants, call ops —
  are interned into program-level pools rather than carried inline, while
  overflow args live block-level in `SsaBlock.extra_args` (indexed by
  `SsaInst.meta`), keeping `SsaBlock.ops` cache-dense.

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

## Moves

- 2026-04-13 (782a6dfb) replaced [[enum-variant]]: the per-variant enum carried
  heap-allocated arg/result vecs and inline 64-bit constants on every op,
  bloating the block op stream; a flat fixed-size record with payloads interned
  into program-level pools (primitive ops, constants, call ops, extra args) and
  packed 4-byte operands keeps SsaBlock.ops cache-dense and shrinks SSA-IR
  memory (code)
