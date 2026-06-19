- Opcode decoding is a streaming interface: a decoder pushes ops to a handler
  that can pull one or several ops at a time (`OpcodeHandler`); consumers never
  need a fully materialized op array for a function body.

- Opcodes are modeled as separate prefix-namespaced enums — single-byte,
  0xFC-prefixed, 0xFD-prefixed, and the wasm 3.0 0xFB GC prefix — each
  generated from one macro table and unified under a tagged `WasmOpcode` the
  decoder dispatches on.

- The decode is exposed as a consumer-driven pull stream (`OpStream`) that
  lazily decodes opcodes on demand and offers a multi-opcode lookahead window;
  a handler can inspect several upcoming opcodes before consuming them.

- The decoder brackets a body's opcode stream with on_decode_begin and
  on_decode_end callbacks, giving handlers a place to run per-function setup and
  finalization around the opcode walk.

- Every relaxed-SIMD opcode is accepted and given a single deterministic
  interpretation: those with an exact non-relaxed equivalent are rewritten to it
  at decode time (relaxed swizzle to swizzle, relaxed trunc to trunc_sat, relaxed
  min/max to min/max, relaxed q15mulr to q15mulr_sat), and the remainder
  (lane-select, MADD/NMADD, dot) are given fixed deterministic native lowerings
  keyed on the relaxed opcode itself (`is_relaxed_simd_opcode`).

## Facts

- 2024-01-30 (80306a9a) statement: block-type immediates are decoded into
  Empty / single-value-type / type-index forms by peeking the first byte — 0x40
  is empty, a byte below 0x80 is a value type, otherwise the bytes are an s33
  signed-LEB type index (negative is rejected as malformed) (code).

- 2026-04-11 (89d889fb) rationale: the lazy op decoder accumulated every
  decoded op of a function body in one growing buffer, so a single-consumer
  decode's peak buffer scaled with the whole body length even though consumed
  ops are never revisited; the decoder now drains the consumed prefix once it
  exceeds a 256-op threshold and tracks a `decoded_base` offset so cursor
  indices stay absolute, bounding the live op buffer to a sliding window — but
  compaction is disabled (via `retain_decoded_ops`) whenever more than one
  handler shares the stream and an earlier handler may still re-read the prefix
  (code).

- 2026-04-19 (77d429c7) rationale: the spec lets each relaxed-SIMD op pick one of
  several results per lane; this engine pins every supported relaxed op to a
  single deterministic behavior rather than exposing host-dependent results, so a
  relaxed op never reaches the backend as a distinct nondeterministic kind and
  instead aliases a deterministic op or is given a fixed deterministic native
  lowering (code).

## Moves

- 2025-08-13 (c7ae92e5) replaced [[push-broadcast-decoder]]: the push callback
  handed handlers one opcode at a time with no way to look ahead; a pull stream
  lazily decodes on demand and exposes a multi-op lookahead window (code).
