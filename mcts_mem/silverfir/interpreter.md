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

- Coverage is wasm 3.0 less SIMD and GC, which are excluded by design: a
  v128 lane and a GC object are representation changes rather than more
  handlers, and both are rejected by name. Exception handling is carried
  only where a throw's handler is in the same function; crossing a call
  needs the native chain's return stack unwound and is rejected by name.

- A memory or table index packs into the static offset's high bits, except
  where a 64-bit offset needs all of them -- those carry a side-table index
  instead, and never reach a native handler.

- A memory, table or global that is imported or exported IS the substrate's
  shared entity; a purely private one is a private array the dispatch chain
  indexes directly. Accesses to a shared entity are denied a native handler
  (`TableEntries`).

- A function reference leaving this instance is named by the EMBEDDER, not
  by the engine: a local index means nothing to whoever reads it, and
  resolving a name means calling into another instance, which the embedder
  owns and an engine-side registry could only reach through a raw pointer
  into storage the embedder may move (`FuncRefHost`).

- Instantiation hands the instance back when a segment traps, and runs the
  dispatch chain, element segments, data segments, then the start function
  in that order (`new_partial`).

## Facts

- 2026-07-26 (be0da7c2) rationale: an entity another instance can reach must
  BE the shared one rather than a copy, because both sides write it and a
  copy stops agreeing at the first store; the tiering mirrors the JIT's
  `FixedLocalOnly` / `Generic` split, which exists because its native code
  reads a projection whose layout it controls rather than the entity itself
  (sourced).

- 2026-07-26 (979c5001) rationale: naming a cross-instance function
  reference belongs to the embedder because resolving one means calling into
  another instance, which needs `&mut` to it while the caller is borrowed --
  the embedder owns both, where an engine-side registry could only reach it
  through a raw pointer into storage the embedder is free to move (sourced).

- 2026-07-26 (979c5001) rationale: a trapping instantiation still hands back
  its instance because element segments run before data ones, so their
  writes -- possibly into a table another instance holds -- stand, and
  anything they reference has to stay callable (sourced).

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
