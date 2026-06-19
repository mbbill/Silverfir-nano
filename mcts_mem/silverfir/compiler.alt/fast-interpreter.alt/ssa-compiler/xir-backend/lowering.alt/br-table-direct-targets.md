- The lowered br_table instruction's jump table holds one pointer per target block
  plus the default, computed directly from each target block's start offset; no
  per-edge code runs between the table dispatch and the target block.

## Moves

- 2025-10-12 (2bee2d7a) replaced by [[lowering]]: a single jump table pointing
  straight at target blocks cannot carry the per-edge phi assignments each br_table
  target needs, so each table entry now points at a small stub that performs that
  target's phi stores before jumping to the real block (code).
