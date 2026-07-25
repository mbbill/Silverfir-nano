- The interpreter is a folded stack machine: predecoding folds every pure
  routing opcode (`local.get/set/tee`, consts) into the operand and
  destination fields of fixed 32-byte instruction cells; only semantic
  ops, materialization movs, and control flow dispatch
  (`vm/interpreter/`).

- Temps are addressed by wasm stack height in one flat frame
  `[params | locals | temps]`; there is no SSA, no phi elimination, and
  no parallel-move shuffling.

- `local.set/tee` retro-patches the producer's destination field when a
  soundness guard holds (known producer, same control region, target local
  neither read nor written since the producer, no hazard flush); the region
  counter bumps at every emitted branch and every live merge.

- There is no discovered-fusion pattern library. A small closed fixed-combo
  family is part of the instruction set instead: fused compare-and-branch
  ops (a compare whose sole same-region consumer is a conditional branch or
  `if` guard becomes the branch; inverted senses are closed within the set),
  and `i32.eqz` folds into an inverted compare or a flipped branch sense.

- Every op without a native handler executes through one shared Rust
  single-instruction executor reached by a uniform exit/re-enter protocol;
  that slow path also carries host calls, `call_indirect`, and rich trap
  messages.

- Coverage is wasm 2.0 plus multiple memories (a memory index packs into
  the static offset's high bits); SIMD, exception handling, GC, tail calls,
  and memory64 are rejected with clean errors.

## Facts

- 2026-07-23 measurement: folding rates on the wasi benchmark corpus
  (foldsim v4, three hostile reviews): `local.get` folded 95.8%,
  `local.set` dst-folded 75.1%, `tee` 98.9%, consts 97.7%; predicted
  dispatch/old-basis ratio 0.489 on CoreMark (code).

- 2026-07-23 measurement: dynamic verification on the real pipeline,
  CoreMark 11000 iterations: 3.904G dispatches / 9.387G old-basis ops =
  ratio 0.416 vs the 0.489 static prediction (fused branches and eqz
  folding postdate the model); movs 10.6% of dispatches vs 9.0 predicted;
  355K dispatches vs 853K old-basis ops per iteration (code).

- 2026-07-23 measurement: the fused compare-branch family removed 8.9% of
  CoreMark dispatches (25.16G → 22.92G) (code).

- 2026-07-23 measurement: pre-accumulator, the native chain averaged
  0.73 ns ≈ 2.1-2.3 cycles per dispatch on Apple M-series — throughput
  near the naive-handler floor, with every operand round-tripping through
  frame memory (code).

- 2026-07-23 statement: the deleted [[fast-interpreter]] ran the full
  unfused dispatch stream (its local get/set/const all dispatched) at a
  comparable CoreMark score — roughly 0.94 effective cycles per dispatch,
  with TOS and hot locals register-resident (sourced).

- 2026-07-23 pitfall: `br_if` whose label is the function body is a
  conditional return; lowering it through a branch helper that ignored the
  conditional op emitted an unconditional return (the executed spectest
  subset never covered the case — caught by a targeted unit test) (code).

- 2026-07-23 rationale: the no-pattern-library constraint is the author's:
  the historical fusion set cost ~2.9 MB and its coverage was
  app-dependent; an interpreter aimed at small deployments must not be
  bigger than the JIT (sourced).
