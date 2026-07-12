- Bulk memory and table operations that need three operand registers allocate a
  heap array of the three register indices and store its raw pointer in the
  instruction's immediate field.

- The heap operand array is never freed once its pointer is stored in the
  instruction.

## Moves

- 2025-10-13 (10a69247) replaced by [[instruction-format]]: the heap side-table
  allocated (and leaked) one Box per bulk-op instruction and added a pointer
  dereference per execution; the three register indices fit in the instruction's
  own b/c immediate fields (code).
