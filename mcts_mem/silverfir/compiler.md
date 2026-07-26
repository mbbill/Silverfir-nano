- The compiler is streamable end-to-end and per-function: each stage consumes
  its input and produces its output incrementally for one function, and on the
  native backends the streaming pipeline never holds a fully materialized
  whole-module IR (the IR-dump configuration retains the batch pipeline that
  does). Hosted eager builds may run a bounded number of these
  independent per-function pipelines concurrently; low-memory/no_std builds
  keep the single-stream path.

- The pipeline pushes structural choices up into the early, structured stages
  and ISA-specific cleverness down into small late rewrites, rather than
  running a large speculative global optimizer.

- Values are split into three classes with different residency policies:
  canonical locals (stable frame slots, hot ones additionally pinned to cache
  registers), deep Wasm stack / call payloads (always canonical operand slots),
  and short-lived transients (the only class that participates in register
  allocation). This split removes the need for a heavyweight global allocator.

## Facts

- 2026-07-22 measurement: the FFmpeg startup campaign found accidental
  whole-function scratch, repeated scans, temporary containers, and duplicated
  planning work rather than an inherently superlinear compiler pipeline; the
  measured checkpoints and retained/rejected changes are recorded in
  [[compiler.fact/startup-campaign-2026-07-22]] (sourced).

- 2026-03-06 (37c40ffe) rationale: the backend split must not duplicate the
  middle of the pipeline — Wasm decode, stack tracking, neutral lowering / IR
  construction, finalization, and branch-target metadata stay shared across
  backends; only the final execution backend differs, establishing the
  shared-pipeline / per-backend-tail principle the later compiler stages inherit
  (sourced).

- 2026-03-09 (ab127bb7) rationale: the hard invariant is that after the
  backend-facing IR the backend works only with registers, frame slots, and
  immediates/targets and must not reason about logical stack height, spill depth,
  or TOS validity; grouping must stay before backend IR because it depends on
  stack-machine discipline, but once groups form the stack model collapses into
  explicit register/memory behaviour (sourced).

- 2026-03-11 (c4102007) statement: a root invocation enters the native ABI once
  from Rust and stays native until final return or trap — local calls and
  call_indirect must not route back through Rust or any non-native executor
  (mixed-mode execution is forbidden); a public shim exists only at the runtime
  boundary to seed the root call-link area and owns no execution semantics
  (sourced).

- 2026-04-07 (f09ae8ee) rationale: when inlining existed, its size budget was
  held deliberately low to mirror LLVM-level inlining — inline only trivial
  wrappers and tiny arithmetic helpers, never large leaf computations like CRC
  routines; raising the threshold to admit big eligible leaf bodies was an
  explicit non-goal (sourced).

- 2026-03-29 (e7796dca) rationale: the engine is fast not because it runs a large
  optimizer but because the pipeline is arranged so the most profitable
  optimizations become cheap, low-risk local rewrites — keep Wasm structure long
  enough to make global-ish choices cheaply, turn deep stack state/calls/locals
  into explicit canonical homes early so the backend never needs general-purpose
  register allocation, and preserve just enough semantic shape in MachineIR for
  small late peepholes and instruction selection to recover native-quality code
  (sourced).

- 2026-06-11 statement: the fast interpreter did not lose to a JIT — it became
  one: the single-pass compiler-interpreter method transferred directly into
  the single-pass micro-JIT, and the micro-JIT then evolved into the
  three-stage IR pipeline; -rs's lesson that dispatch count, not memory
  access, is the interpreter's bottleneck is why the compiler became the
  product (sourced).

- 2026-06-14 rationale: there is no traditional register allocator anywhere in the
  pipeline, even at HEAD — Wasm's operand stack is already a linear SSA, and
  capping the windowed top-of-stack guarantees the live values fit the register
  budget by construction, so the allocator's job is done before any
  spill/interference/coloring pass would run; the compiler-pipeline complexity
  that sank the xir line as an *interpreter* (where every extra instruction is a
  dispatch cost) is acceptable here because this is a JIT — a mov is nearly free
  on register-renaming hardware — which is exactly the wall the interpreter arc
  hit and the JIT clears (sourced).

- 2026-06-14 rationale: nano is a streaming, function-by-function compiler targeting
  esp32/pico2-class devices, so it holds at most one in-flight function's IR and
  per-function output size is a hard footprint constraint — growth that enlarges
  a compiled function (e.g. leaf inlining) costs memory that the target may not
  have, which weighs against optimizations a whole-module compiler would take
  for free (sourced).

## Moves

- 2026-03-07 (bc6c91c8) replaced [[compiler.alt/fast-interpreter]]: the
  micro-JIT was embedded inside the handler-threaded preserve_none fast
  interpreter and its generated code retained interpreter-shaped overhead
  (loop-boundary dispatch, repeated memory-metadata loads, hybrid JIT/handler
  transitions), and its dependence on preserve_none could not port to
  RISC-V/ARM32/MCU targets; the native backend instead owns a self-defined VM
  ABI entered through a global-asm trampoline that threads native-entry
  addresses directly, so it no longer behaves as one more kind of
  fast-interpreter handler (code)

- 2026-04-11 (89d889fb) replaced [[compiler.alt/whole-module-batched-pipeline]]:
  the batched pipeline decoded every function's SemanticProgram up front and
  held the whole module's semantic IR live in one Vec<Option<SemanticProgram>>
  across a separate fixed-point inlining phase and a separate prepare phase, so
  peak compile-time memory scaled with the total decoded size of the module;
  streaming each caller (decode, inline retained leaf seeds, lower immediately)
  keeps only the tiny retained inline-candidate set plus one in-flight caller
  live at a time, holding the whole module's semantic IR never in memory at once
  (code)

- 2026-04-14 (fc7c2f74) dropped: semantic-IR leaf-function inlining — inlining
  grows compiled function size, and a streaming function-by-function compiler
  targeting esp32/pico2 cannot afford bigger functions (more memory per
  in-flight function); the gain was marginal anyway because algorithm-4 is not
  smart enough to place hot-local swaps optimally, so a larger inlined function
  gets *worse* local-cache allocation (more locals to cover) — footprint cost
  plus negative interaction with the local cache, not a refactor casualty
  (sourced).

- 2026-04-30 (92a3a400) replaced [[compiler.alt/block-streaming-pipeline]]:
  the whole-function joint planner and cross-block optimizations both need more
  than one block live at once, so per-block streaming would either lose codegen
  quality or buffer the whole function back and defeat the one-block memory goal
  (sourced)
