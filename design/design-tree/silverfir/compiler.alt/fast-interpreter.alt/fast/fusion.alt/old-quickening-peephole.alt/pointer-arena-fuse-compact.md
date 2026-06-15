- The IR builder decodes each function into an index-based temp list, then
  allocates the final boxed Instruction arena, patches raw alt pointers into it,
  runs the fusion peephole over that pointer arena in place, and then runs a final
  compaction pass over the same pointer arena.

- Fusion leaves nop placeholders in the stream and the compaction pass that would
  remove them is disabled (it keeps every slot); fused-out second operands remain
  as executed nop dispatches.

- The fusion peephole reads and writes the concrete Instruction slice directly
  (handler pointer plus imm0/imm1/imm2 fields).

## Moves

- 2025-09-12 (40f4a141) replaced by [[old-quickening-peephole]]: compaction had to
  remove the nop placeholders left by fusion, but doing so on the boxed Instruction
  arena means re-patching every raw alt pointer and every br_table relative offset
  against a reallocating buffer, so nop-dropping was left disabled for stability;
  moving fusion and compaction onto an index-keyed TempInst IR (alt and br_table
  targets as indices remapped through an old->new table) lets compaction actually
  drop the fused-out nops before pointers are ever materialized (diff).
