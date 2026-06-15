- A constant expression is delimited by linearly scanning its bytes for the
  first 0x0b and splitting the payload there, returning the bytes up to and
  including that byte.

## Moves

- 2024-02-04 (a03f7fcc) replaced by [[constexpr-boundary]]: scanning the raw
  bytes for the first 0x0b mis-terminated a constant expression whenever 0x0b
  appeared inside an opcode's immediate (e.g. a constant's LEB128 bytes);
  decoding each const-expr opcode and consuming its immediate finds the real
  terminating end opcode (diff).
