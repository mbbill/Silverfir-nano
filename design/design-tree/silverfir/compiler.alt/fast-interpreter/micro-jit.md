- On JIT-capable platforms the interpreter compiles each basic block to native
  code with a template micro-assembler instead of dispatching pre-compiled
  handlers: consecutive JIT-able IR ops are grouped into one straight-line
  native handler whose machine-code address becomes the handler pointer, and the
  dispatch chain interleaves JIT'd groups and built-in C handlers under one
  `preserve_none` register contract; ops it cannot handle keep their C handlers.

- The JIT selects operand and result registers per opcode from the compile-time
  TOS depth using the same modulo-four depth-variant convention as the
  interpreter handlers; dispatching out of a JIT group into an interpreter
  handler reads operands from the right registers.

- A compiled group defers a value's move into its canonical TOS register until a
  group boundary (slot-location window), fuses a trailing compare into a
  conditional terminator; the boolean is never materialized, caches recently
  stored frame locals in dedicated registers, and compiles void/single-result
  function return inline (pop frame, reload hot locals, dispatch to the caller's
  resume instruction).

## Facts

- 2026-03-02 (16c7e03e) rationale: the central thesis — because the
  interpreter's TOS register window plus L0/L1/L2 hot-local cache already pin
  every hot value to a fixed physical register, a JIT for fused blocks needs no
  SSA, no register allocator, and no optimizing compiler, only a micro-assembler
  emitting 1-2 ARM64 instructions per opcode from the known mapping plus a
  trivial alias/constant tracker; the architecture that made the interpreter fast
  is exactly what makes its JIT trivially small (~10-20 KB), positioned as the
  first Wasm JIT small enough for MCU-class A-profile/RV64 devices (diff).

- 2026-03-03 (2ff0b000) statement: the embedded-size positioning that is the
  project's reason-for-being — ~250 KB total (the ~230 KB core plus a ~10-20 KB
  micro-assembler) against a runtime-size comparison table putting it 60-400x
  smaller than V8/Wasmtime/Wasmer/WAMR JITs — is recorded in full in
  [[micro-jit.fact/embedded-size-positioning]]; the collapse is possible because
  the TOS + L0/L1/L2 architecture already supplies the register assignment, so the
  "compiler" is just a template assembler (author).

- 2026-03-03 (2ff0b000) rationale: fresh per-group emission was chosen over
  copy-and-patch (stitching pre-compiled handler blobs) because emitting with
  knowledge of the concrete immediates and register state enables cross-op
  constant folding and alias elimination that blob-copying cannot express; and
  over baseline-JIT single-pass register allocation (Winch/Liftoff) because the
  TOS + L0/L1/L2 window already supplies a fixed pre-determined register mapping,
  so register allocation is skipped entirely (author).

- 2026-03-03 (2ff0b000) rationale: the micro-JIT IS the fusion system, not a
  layer on top — it reuses the builder pipeline through depth-variant selection
  and spill/fill insertion and replaces only the final handler-assignment stage,
  removing static fusion's three limits at once: the 3-immediate encoding
  budget, the finite pattern set, and the workload-dependent discovery step
  (diff).

- 2026-03-06 (37c40ffe) rationale: `preserve_none` must become an optimization,
  not a core requirement — it is available only on a limited target set while the
  project also targets RISC-V/ARM32/MCU, so missing it must hurt performance but
  never block correctness or portability; this is the wall that the micro-JIT,
  embedded in the preserve_none interpreter, could not clear (diff).

- 2026-03-06 (37c40ffe) rationale: the native-backend roadmap's planned
  end-state kept the handler-based interpreter family (base + fusion) as a
  permanent fallback backend alongside the new native backend — `interp/` for
  handler dispatch, `native/`/`jit/` as a sibling owning its own VM ABI — with
  fusion positioned as the second-best option and a multi-tier platform matrix
  (native, else fusion/base on targets without preserve_none/executable memory,
  else normal-ABI+musttail); a future implementor weighing whether to retain the
  interpreter should know this dual-family architecture was the explicit plan
  when the native backend was born but was reversed within a month — interp
  gated off by default (61b3fac8) then deleted outright, leaving the engine
  JIT-only ([[execution-model]]) (author).

- 2026-03-06 (37c40ffe) rationale: the roadmap's conclusion is that the remaining
  gap to a real JIT is structural, not peephole-sized — the lagging loop/memory
  kernels are already inside JITed code, yet the micro-JIT's generated code still
  retains interpreter-shaped overhead (repeated dispatch at loop boundaries,
  repeated memory-metadata loads and bounds-check setup, hybrid JIT/handler
  transitions) because it is shaped like just-one-more-kind-of-handler reusing
  the fast interpreter's lowered IR and dispatching at group exits (diff).

- 2026-03-05 (32a13cf6) rationale: once the JIT emits Spill/Fill as plain
  str/ldr against the frame, spill/fill stop being group boundaries and become
  JIT-able; emit_op asserts the emitter's running height equals the IR op's
  pre_height, pinning JIT register selection in lockstep with the lowering's
  resolved variants (diff).

- 2026-03-06 (4c24eec1) limitation: a group may end in an unconditional `br`
  terminator only when the branch needs no stack fixup (stack_drop==0 or
  arity==0); a br that must both drop stack and carry results falls back to the
  1:1 base handler (diff).

- 2026-03-06 (a72fde5a) rationale: a GPR-only TOS-slot alias array could not
  express a slot living in an FP register, forcing every float handler to bounce
  GPR->FP->GPR; tracking each slot's location as GPR or FP register keeps float
  values resident in FP registers across the group (diff).

- 2026-03-06 (85d7bc23) rationale: the FP-only frame-local alias could not cache
  an integer frame local because a general-purpose TOS register is not stable
  across stack shifts; backing the alias with a dedicated stack-shift-stable
  register lets integer frame locals be cached too (diff).

- 2026-03-06 (c4a5abde) rationale: a single-entry frame alias cached only one
  frame local at a time; a fixed-size cache over two dedicated registers with
  epoch-based LRU caches several, and binary ops propagate the surviving
  operand's slot-location through the result so an aliased operand is not lost
  (diff).

- 2026-03-06 (47d06889) rationale: micro-JIT tail-fusion fuses float compares
  into the branch as well as integer compares, emitting the fcmp and branching
  on the resulting flags directly (diff).

- 2026-03-06 (08bd0fb4) rationale: float unary ops and int<->float conversions
  allocate a fresh FP result register and leave the result FP-resident, instead
  of round-tripping GPR->FP-scratch->op->FP-scratch->GPR on every float op
  (diff).

- 2026-03-05 (b547290c) pitfall: a JIT group is a single straight-line handler
  with one entry point, so it must not span an IR op reachable by a
  non-fallthrough branch; group formation precomputes every incoming branch
  target and breaks the group before any such index, or a branch would land
  mid-group with no entry label (diff).

- 2026-03-06 (e544d4fa) rationale: a local mitigation inside the doomed
  micro-JIT — mem0_base/mem0_size loaded once per group into dedicated cache
  registers on first memory access and reused, instead of reloaded per
  load/store — superseded wholesale by the native re-architecture (diff).

- 2026-03-03 (d193a463) rationale: the JIT's depth_variant and tos_reg
  reproduce the interpreter's cycling-modulo-four TOS register convention
  exactly, so JIT-emitted code and the surrounding C handlers agree on which
  physical register holds each operand at every group boundary — without this
  lockstep, dispatching out of a JIT group into an interpreter handler would
  read operands from the wrong registers (diff).

## Moves

- 2026-03-07 replaced by [[compiler]]: the interpreter's preserve_none
  handler-threaded model and its embedded micro-JIT retained interpreter-shaped
  overhead and could not port to RISC-V/ARM32/MCU targets, so a native
  code-generation backend owning its own VM ABI replaced the whole interpreter
  execution era (diff).
