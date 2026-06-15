- Consecutive Wasm opcodes are fused at IR-build time into single
  super-instruction handlers, generated from a pattern table by a build-time
  code generator; a fused-op's per-operand fields pack into a fixed budget of
  three 64-bit immediate slots with no field spanning a slot boundary, and a
  candidate whose fields do not fit is rejected.

- A branch-class op (br_if, if_) may appear only at the end of a fusion
  pattern, never in the middle; a fused pattern that ends in a branch uses
  nonlinear dispatch (br_if) or guard-check fall-through (if_), and a fused
  superinstruction containing a branch spills its live TOS registers to their
  canonical operand slots before the handler runs.

- Fusion is a default-on compile-time size/speed knob (`fusion` cargo feature):
  disabled, the fused handlers and their C wrappers are compiled out and the
  builder falls back to one handler per opcode.

## Facts

- 2026-02-22 (4bb1de83) rationale: fusion eliminates roughly two-thirds of
  dispatches in CoreMark; local.get/set/tee is ~38% of dispatches and the top-10
  successors of local.get cover 88-92% of its occurrences, so a small pattern
  set absorbs most local-access dispatch (author).

- 2026-02-15 (30664a41) measurement: the no-fusion build is ~200KB but ~40%
  slower (wasm3-class); a full set adds ~500KB, but fusion has diminishing
  returns — ~100KB of fused handlers already recovers ~80% of full-fusion
  performance, and the built-in default set captures ~90% of the benefit so
  custom per-app fusion yields only incremental gains (diff).

- 2026-03-01 (a05de669) measurement: draft-paper ablation (Apple M4, TOS window
  + preloading always on) finds fusion the dominant optimization — fusion-only is
  1.70-2.04x faster than hot-local-cache-only across SHA-256/bzip2/LZ4, and the
  hot-local cache alone performs near the wasm3 baseline because it cannot
  overcome per-instruction dispatch without fusion creating multi-instruction
  handler bodies; the two are synergistic (combined SHA-256 speedup 3.29x exceeds
  the product of individual gains because a fused handler delivers
  dispatch-elimination and memory-elimination in one body) (author).

- 2026-02-17 (d4b68b3c) rationale: a br_if is treated as "simple" (eligible for
  the fast br_if handler) only when the target block has arity 0 and the branch
  unwinds no stack; non-simple br_if needs the general path (diff).

- 2026-02-17 (577445eb) rationale: if_ is fused only when its block type is
  empty, to avoid threading the CompileContext into the fusion matcher to
  resolve complex block types; a non-branch pattern ending in a comparison
  yields when the next op is br_if/if_, preferring to fuse the compare into the
  branch (diff).

- 2026-02-19 (79573575) rationale: averaging each pattern's per-workload
  frequency across workloads diluted patterns hot in one workload but unused in
  others; the merge takes the max frequency per pattern instead so hot patterns
  are not washed out before scaling to the reference total (diff).

- 2026-02-19 (f0c2a5c6) rationale: local-index fields in fused-op encodings were
  halved from 16-bit to 8-bit so longer/denser patterns fit the three-slot
  budget, at the cost of bailing the fuse to the unfused path whenever a remapped
  local index is >= 256 (diff).

- 2026-02-21 (f4a8503d) rationale: the matcher generator moved from a
  by-length longest-first sequential scan with eager per-opcode immediate cloning
  to by-first-opcode dispatch plus a two-phase match (check all opcodes, clone
  immediates only on a full match), avoiding cloning immediates the matcher
  discards and improving compile speed (diff).

- 2026-02-21 (7bd4154c) rationale: a global `-ffp-contract=off` on the
  trampoline C suppressed mul+add->FMA contraction everywhere, losing FMA where
  Wasm permits it; an empty-asm `FP_MUL_BARRIER` emitted on the mul result only
  inside fused patterns that also contain a float add/sub enforces strict
  sequential float evaluation precisely where contraction would change the
  result, leaving every other float mul free to contract (diff).

- 2026-02-17 (194dbd17) pitfall: the encoding-feasibility check must simulate
  real slot-packing (fields packed into three 64-bit slots without spanning a
  boundary) rather than summing field bit-widths against a 192-bit budget — a
  naive bit-sum admits patterns the real packer rejects when fields straddle
  slot boundaries (diff).

- 2026-02-18 (00191661) pitfall: auto-naming bare encoding-field names from a
  per-width count collides when a pattern mixes i32+i64 constants or br_if+if_
  branches; the count must be taken across all width/opcode variants of a
  category so the bare name is used only when the category truly has one member
  (diff).

- 2026-02-17 (5e9345de) pitfall: a handler whose successor is never the
  sequential next instruction must use nonlinear dispatch (always reload the next
  handler pointer); br_table and any fused pattern ending in br_if are forced
  nonlinear (diff).

- 2026-02-21 (04adbbfc) pitfall: a non-branch fused pattern can have net push +1
  but a higher peak intermediate push, so spilling on net push instead of peak
  push skips a spill the non-fused path emits, leaving the fused and non-fused
  paths disagreeing on spill_depth for following instructions (diff).

## Moves

- 2026-03-07 replaced by [[compiler]]: the interpreter's preserve_none
  handler-threaded model and its embedded micro-JIT retained interpreter-shaped
  overhead and could not port to RISC-V/ARM32/MCU targets, so a native
  code-generation backend owning its own VM ABI replaced the whole interpreter
  execution era (diff).
