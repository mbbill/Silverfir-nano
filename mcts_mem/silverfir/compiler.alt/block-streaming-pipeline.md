- Peak compile memory is bounded by roughly one basic block rather than by one
  whole function.

- Each block is lowered and emitted into the code buffer, then dropped, before
  the next block is processed.

- Backends consume machine instructions one at a time through a begin-block,
  emit-instruction, end-block protocol with a one-instruction lookahead, rather
  than being handed a whole block at once (MirBlockSink).

- Cross-block and whole-program optimizations are reapplied on top of the
  per-block stream by an optional buffer-and-rewrite stage — a single-producer,
  single-consumer middleware that collects blocks, rewrites them, and re-streams
  them — present only when a RAM budget allows it.

- A single function-wide cached-local set (parameters first, by index) is fixed
  up front rather than recomputed per block.

## Facts

- 2026-04-30 rationale: the attempt targeted memory-tight devices (Pico 2 /
  ESP32-class, ~512 KB RAM) whose per-function intermediate forms can exceed
  available heap, so that functions too large to compile whole could still be
  compiled (sourced).

- 2026-04-30 measurement: at full RAM budget the per-block streaming output was
  byte-identical to the buffered whole-function baseline across all eight WASI
  benchmarks, while the streaming-only tight-budget path produced ~10-20% larger
  native code but compiled functions that otherwise exhausted memory; the
  quality cost was confined to the sub-function budget, not to streaming itself
  (sourced).

- 2026-04-30 (92a3a400) statement: the joint planner derives one cached-local
  assignment from the whole function's control-flow graph and operand stream at
  once, and the lowering helpers read the full SSA program by reference
  throughout, so releasing planner and SSA state block-by-block is not
  expressible without reworking those interfaces (code).

## Moves

- 2026-04-30 (92a3a400) replaced by [[compiler]]: the whole-function joint
  planner and cross-block optimizations both need more than one block live at
  once, so per-block streaming would either lose codegen quality or buffer the
  whole function back and defeat the one-block memory goal (sourced)
