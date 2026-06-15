- A constant expression's end is found by decoding each opcode and consuming
  its immediate until the terminating `end` opcode, not by scanning the raw
  bytes for the first 0x0b.

- The decode boundary admits a fixed whitelist of wasm 3.0 GC allocation
  instructions under the 0xFB prefix (struct.new, struct.new_default,
  array.new, array.new_default, array.new_fixed, array.new_data,
  array.new_elem, ref.i31, plus the any/extern conversions); any other 0xFB
  opcode in a constant expression is rejected as malformed (`parse_constexpr`).

## Facts

- 2024-02-04 (8e4cafc9) pitfall: the opcode-walk's catch-all arm built a
  WasmError::Malformed for an opcode not legal in a constant expression but
  discarded it (constructed the Err value as a statement and never returned
  it), so an illegal opcode silently fell through instead of failing; the arm
  must break the decode loop with that error (diff).

- 2025-06-21 (1216f037) rationale: an unrecognized opcode encountered while
  decoding a constant expression is reported as malformed (a binary-shape error
  at the decode boundary), not as invalid by the validator (diff).

## Moves

- 2024-02-04 (a03f7fcc) replaced [[byte-scan-boundary]]: scanning the raw bytes
  for the first 0x0b mis-terminated a constant expression whenever 0x0b
  appeared inside an opcode's immediate (e.g. a constant's LEB128 bytes);
  decoding each const-expr opcode and consuming its immediate finds the real
  terminating end opcode (diff).
