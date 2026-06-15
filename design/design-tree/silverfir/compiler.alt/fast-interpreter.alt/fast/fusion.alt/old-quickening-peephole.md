- After the fast backend builds a function's instruction stream, a peephole pass
  rewrites adjacent opcode runs (local.get+arith, const+arith, arith+local.set,
  const+local.set, local.get+load, const+store, and 3-grams) into single fused
  superinstruction handlers.

- Fusion matches on raw handler-function-pointer equality, not opcode bytes; a
  fused run is encoded in place, the leading slot becoming the superinstruction
  while trailing slots are overwritten with nop placeholders that the fused
  handler skips, keeping instruction indices and branch targets valid.

- A candidate run is vetoed when any of its ops is an inbound branch target;
  fusion never spans a label.

## Facts

- 2025-09-12 (35a02cfb) rationale: a fused pair is encoded in place — the first
  slot's handler/immediates become the superinstruction and the second slot is
  overwritten with nop while the fused handler returns pc.add(2) to skip it; the
  nop placeholder is kept rather than removed so instruction indices and alt branch
  targets stay valid (diff).

- 2025-09-12 (4d1ce6e7) rationale: fusing a 64-bit-constant store cannot be
  expressed when the instruction header carries only two u32 immediates (both
  consumed by the constant, none left for the offset); the header's 8-byte
  alignment-padding word was repurposed into a free-form imm2 field to carry the
  offset, unblocking the const+store fusions previously skipped (diff).

- 2025-09-12 (c03f7dea) pitfall: the fusion label-guard must exclude a pair when
  EITHER op is a branch target, not only the second — guarding solely the second
  slot still allowed fusing a pair whose first op is a jump target, so a branch
  landing on that first slot would execute a two-slot superinstruction it was not
  meant to, corrupting the stack (diff).

- 2025-09-13 (7dc9b0a2) rationale: a fusion whose last op is a branch is written
  into the branch slot itself rather than the leading slot, so the branch's alt
  target pointer is preserved in place while the preceding compare/const ops
  collapse to nops; this is what lets the peephole fuse compare-and-branch idioms
  without breaking the precomputed control-flow edges (diff).

- 2025-09-13 (e862005a) rationale: the compaction pass drops not only fusion nop
  placeholders but every structural control op (block, loop, end) as a no-op,
  because the fast backend resolves all structured control flow into precomputed
  alt-pointer branch targets before dispatch; their handlers are kept only as
  debug_assert(false) traps to catch a structural op that wrongly survived
  compaction (diff).

- 2025-12-02 (22063dd8) pitfall: the shift-and-add address-computation fusions
  (and the load+tee fusion) bake a single base+offset memory access into one
  superinstruction implicitly targeting the default memory; the original patterns
  matched any load/store regardless of its memory-index immediate, so under
  multi-memory a load against a non-zero memory would be silently fused onto a
  default-memory access — the fix gates every such fusion on memidx == 0 (diff).

- 2025-12-05 (7b7867ae) pitfall: the quickening gating default was flipped from
  opt-in to opt-out (SF_ENABLE_QUICKENING, default off, replaced by
  SF_DISABLE_QUICKENING, default on); the removed NOTE recorded the historical
  hazard that had kept it off — br_table targets point to END instructions dropped
  during post-fusion compaction, and the inbound-label marking that forbids fusing
  across labels did not correctly account for the post-compaction target locations
  (diff).

## Moves

- 2025-09-12 (40f4a141) replaced [[pointer-arena-fuse-compact]]: compaction had to
  remove the nop placeholders left by fusion, but doing so on the boxed Instruction
  arena means re-patching every raw alt pointer and every br_table relative offset
  against a reallocating buffer, so nop-dropping was left disabled for stability;
  moving fusion and compaction onto an index-keyed TempInst IR (alt and br_table
  targets as indices remapped through an old->new table) lets compaction actually
  drop the fused-out nops before pointers are ever materialized (diff).

- 2025-09-13 (d4600172) replaced [[shorter-first-fusion-ordering]]: a shorter rule
  matched first under pair-first ordering, so 4-gram superinstructions could never
  form (e.g. local.get; i32.const; i32.shl; i32.add was consumed by the 3-gram
  shli_local_imm before the 4-gram i32_add_shli_local_imm could match);
  longest-match-first is required to emit them (diff).

- 2025-12-06 (b7b5dc6a) removed: the slot-tracking builder eliminates the
  local/const copies that the fused super-instructions existed to collapse, so the
  table-driven fusion pass (fused_patterns.toml + the peephole quickener + the
  hand-written fused handlers) was removed wholesale rather than ported to the new
  instruction encoding (diff).
