- Binary lowering selects the XIR handler from a match on the operation alone;
  every binary op of a given kind lowers to the same handler regardless of operand
  type.

## Moves

- 2025-10-11 (4b55a6c2) replaced by [[lowering]]: op-only dispatch could not select
  between the existing i32 and i64 handlers, so i64 binary ops lowered to the i32
  handler; keying on (type, operation) routes each to its type-specialized handler
  (diff).
