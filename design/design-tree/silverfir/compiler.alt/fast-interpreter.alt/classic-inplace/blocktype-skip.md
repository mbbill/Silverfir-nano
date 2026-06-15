- When the in-place decoder reaches a structured-control opcode (BLOCK, LOOP,
  IF) it skips the blocktype immediate by parsing its actual encoding — empty
  type, single value type, or a signed-LEB type index — to advance the program
  counter; reference-type and multi-byte type-index blocktypes are skipped
  correctly.

## Moves

- 2025-10-06 (71725b1f) replaced [[leb-i32-blocktype-skip]]: reading the
  BLOCK/LOOP/IF blocktype as a single signed-LEB i32 only advances the program
  counter correctly for the empty type and single-byte value-type forms; a
  structured reference-type blocktype (0x63/0x64 followed by a heap type) or a
  multi-byte type index is then mis-skipped and desynchronizes decoding, so the
  blocktype is parsed by its actual encoding (empty / value type / signed-LEB
  type index) to advance the pc (diff).
