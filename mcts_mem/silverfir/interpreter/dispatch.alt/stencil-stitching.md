- A second bank of every handler came out of the same build-time generator
  walk as the dispatch bank: the same bodies, with the dispatch tail removed.
  A runtime copied those bodies rather than branching to them.

- A link-time pass stitched each maximal straight-line run of cells into one
  native block, copying those bodies back to back, appending a single
  dispatch, and rewriting the run's first cell to enter it. Runs broke at
  branch targets and at any cell the bank did not cover.

- No copied instruction carried a relocatable target. A body that had to
  leave for a shared stub inverted its condition and branched through a
  register loaded from the entry-state block, which published the stubs'
  absolute addresses.

- The compiled block was an overlay on the cells rather than a replacement:
  the pc named the current cell at every instruction boundary, every cell
  stayed independently executable, and traps, bails and the slow path stayed
  the chain's.

- Operand immediates were filled in after the copy: a cell field that was a
  frame offset became the immediate of the frame access itself. Patch sites
  were located by labels the assembler resolved rather than by counted
  instruction positions, and a field that was not a frame offset, or one too
  wide for the immediate, left its cell on the chain.

- The pc advance was hoisted out of the bodies as well: the generator marked
  which bodies still needed the pc, and the runtime materialized it only
  before those and once at each run's exit.

- The tier needed executable memory at run time, taken from the JIT's code
  buffer. One backend carried a bank, and the tier was off by default.

## Facts

- 2026-07-26 measurement: against the same interpreter build, medians over
  interleaved rounds on an M4 — removing the dispatch between cells alone is
  1.30x, baking the operand immediates takes it to 1.47x, and hoisting the pc
  advance to 1.61x. Coverage on CoreMark is 93.1% of cells at a mean run of
  6.5 cells (code).

- 2026-07-26 measurement: the gain is cost per dispatch and not work skipped —
  STREAM's dispatch counts are identical across the two builds (2,568,038,230
  against 2,568,040,092) with its solution validating on both (code).

- 2026-07-26 measurement: hoisting the pc advance is a wash within noise
  rather than a win. Only about a third of the bank's bodies can skip the pc
  at all, since anything that reads a cell field, sets the pc, or can leave
  for a stub still needs it, and the fixups cost about what they save. Per
  benchmark against the previous step, sha256 went 1.52 to 1.84 and bzip2
  1.36 to 1.66, but STREAM Add went 2.37 to 2.16 and Scale 2.16 to 1.95
  (code).

- 2026-07-26 measurement: the tier reaches 0.58x of Winch, wasmtime's
  single-pass compiler, on the corpus median — 0.35x on lua fib, 1.03x on
  mandelbrot — and 0.27x of this project's own optimizing JIT (code).

- 2026-07-26 measurement: what it buys instead is compile time. Linking
  sqlite's 1423 functions costs 34.1 ms against the chain's 25.9 ms and the
  optimizing JIT's 428.5 ms; lua is 13.1 ms against 10.5 and 106.3 (code).

- 2026-07-26 measurement: the price is size in both directions — 463 KB of
  binary for the second bank and its tables, and about 26 bytes of runtime
  executable memory per compiled cell against the 32 bytes that cell's
  dispatch entry already costs. The optimizing JIT's OPTIMIZED code is
  smaller than the stitched code for the same module: CoreMark 78 KB against
  174 KB, lua 792 KB against 1500 KB (code).

- 2026-07-26 rationale: the ceiling is the slot-per-value model rather than
  the dispatch. Outside the accumulator edge and the pinned locals every
  value round-trips through a frame slot, so a three-operand integer op is
  three memory accesses however it is emitted, and closing the remaining gap
  means holding operand-stack values in registers instead (code).

- 2026-07-26 statement: that reproduces with numbers what [[micro-jit]]
  concluded from the other direction — that the remaining gap to a real
  compiler is structural rather than peephole-sized — reached there by a JIT
  shaped as one more kind of handler, and here by taking the dispatch out of
  the handlers themselves (code).

- 2026-07-26 rationale: the overlay is what made correctness free. The spec
  suite result was unchanged at every step and every benchmark checksum
  validated, because a cell whose body the bank lacks, or whose operand will
  not fit an immediate, simply stays on the chain — coverage is a performance
  question and never a correctness one (code).

- 2026-07-26 rationale: position independence, not code generation, is what
  the runtime cost. Reaching the stubs through the entry-state block instead
  of by branch reduced emission to a copy plus an immediate OR, with no
  relocation pass and no encoder of any kind (code).

- 2026-07-26 statement: that lesson does not carry back to
  [[runtime-emission]], whose cost was the encoder itself rather than the
  emission (code).

## Moves

- 2026-07-26 replaced by [[dispatch]]: removing the dispatch between cells is
  worth 1.30x and baking the operand immediates another 1.13x, but the result
  still lands at 0.58x of a real single-pass compiler and 0.27x of this
  project's own JIT, because what remains is the slot-per-value model rather
  than the dispatch — and it pays for that with the runtime executable memory
  build-time handler generation exists to avoid (code)
