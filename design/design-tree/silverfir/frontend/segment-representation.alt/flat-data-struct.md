- A data segment is a single struct holding an is_active bool, a memory index,
  an optional offset expression, and its bytes.

- The offset-expression accessor returns the Option and active-segment code
  unwraps it; a passive segment has no offset.

## Moves

- 2024-02-16 (01f2a6db) replaced by [[segment-representation]]: the flat struct
  used an is_active bool with an optional offset expression that was unwrapped
  for active segments; an enum makes the active variant carry the offset
  expression and the passive variant omit it (diff).
