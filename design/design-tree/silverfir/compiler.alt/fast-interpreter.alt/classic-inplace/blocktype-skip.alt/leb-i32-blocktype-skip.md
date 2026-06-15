- BLOCK, LOOP, and IF skip their blocktype immediate by reading a single
  signed-LEB i32 and discarding it, relying on the blocktype fitting in one
  signed-LEB value.

## Moves

- 2025-10-06 (71725b1f) replaced by [[blocktype-skip]]: reading the
  BLOCK/LOOP/IF blocktype as a single signed-LEB i32 only advances the program
  counter correctly for the empty type and single-byte value-type forms; a
  structured reference-type blocktype (0x63/0x64 followed by a heap type) or a
  multi-byte type index is then mis-skipped and desynchronizes decoding, so the
  blocktype is parsed by its actual encoding (empty / value type / signed-LEB
  type index) to advance the pc (diff).
