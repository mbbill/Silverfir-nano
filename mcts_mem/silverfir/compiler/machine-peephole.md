- A small set of MachineIR peepholes, run after lowering and before backend
  emission, recovers native-quality patterns the fixed-shape lowering set up on
  purpose. Most passes remain block-local (constant deduplication,
  store-to-load forwarding, load reuse, indexed-memory fusion, copy propagation,
  and instruction-selection fusion); a bounded set of structural whole-function
  passes uses explicit CFG proofs for dead parameters, loop-carried context/frame
  values, invariant address bases, and canonical memmove recognition
  (`peephole::optimize`).

- Every peephole rewrites only short-lived transient values; fixed-role and
  cached-local registers carry meaning across the whole block and are never
  folded, propagated, or eliminated; a transform that cannot prove its
  target transient declines. Semantic linear-value versus cached-local
  ownership is read from explicit MachineIR metadata, not inferred from
  register-number layout (`MachineRegOwner`).

- No peephole rewrites a value across the GP/FP register-bank boundary; a
  cleanup never moves a float into a GP register or vice versa; copy
  propagation tracks a float-spilled-to-GP value's GP->FP alias separately and
  rewrites a later float use back to its original FP register.

- Instruction-selection fusion builds higher-level but still ISA-neutral
  MachineIR ops that map to real ISA forms (bitfield-extract from shift+mask,
  shifted-operand binops, test-bits from mask+compare-zero), which each backend
  maps or decomposes late (`fuse_isel`).

- Load-to-load reuse runs twice — once before indexed-memory fusion and once
  after.

- Store-to-load forwarding runs before and after copy propagation.

## Facts

- 2026-07-22 (8a89ce77) pitfall and decision: both loop peepholes ran a
  newly allocated whole-CFG reachability DFS for every numerically backward
  edge. For an existing edge `source -> target`, the old proof that `target`
  reaches `source` is exactly the statement that both endpoints share a
  strongly connected component. The retained iterative Kosaraju analysis
  computes all components once, preserves the numeric-header policy, and is
  shared by loop-frame reuse and loop-address hoisting (sourced).

- 2026-07-22 (8a89ce77) measurement: after the dead-parameter worklist fix,
  replacing per-edge DFS with SCC discovery moved repeated serial bz2 startup
  from 57.066 ms to 51.909 and 52.621 ms (about 8-9%). The former
  `block_reachable_from` hotspot (4.4% self / 5.1% inclusive) disappeared;
  loop-address hoisting fell from 9.3% to 6.7% inclusive and loop-frame reuse
  from 6.4% to 4.4%. Against same-binary Wasmtime Cranelift at 45.078 ms,
  Nano's bz2 ratio is about 1.16x (sourced).

- 2026-07-22 (70e165e0) pitfall and decision: dead block-parameter
  elimination scanned every instruction once per parameter, cloned and
  stripped the terminator once per parameter, then repeatedly rescanned every
  parameter and CFG edge to reach a fixed point. On serial bz2 this pass was
  14.9% self / 17.4% inclusive. The retained implementation scans each block's
  instructions once using a sorted flat register-to-parameter table and
  propagates edge-carried usefulness through a flat reverse-dependency
  worklist; it deliberately remains a small late cleanup rather than adding a
  general liveness framework or register allocator (sourced).

- 2026-07-22 (70e165e0) measurement: repeated serial bz2 startup runs fell
  from 67.180 ms to 56.587 and 57.066 ms (about 15%); the post-change profile
  put dead-parameter elimination at 1.0% self / 3.2% inclusive. A same-binary
  Wasmtime Cranelift rerun measured 45.078 ms, narrowing Nano's ratio from
  1.51x to 1.27x. Three focused worklist/definition-barrier tests, all 349
  workspace release tests, and native/emu64/emu32 spectests passed (sourced).

- 2026-07-22 (fa131723) pitfall: loop-frame reuse scanned every block and used
  repeated linear loop-membership tests to discover entry sources before it
  checked whether the loop exits even contained compatible frame reloads. On
  Pulldown-cmark this made the pass roughly 75% of active serial compiler
  samples; checking exits first and using dense membership reduced same-binary
  startup from about 259 -> 59.8 ms parallel and 375 -> 138.9 ms serial while
  preserving the optimization (sourced).

- 2026-07-22 (faf4b05f) pitfall: loop-address hoisting claimed the first
  preserved dynamic lane after lowering had already derived the function's
  preserved-clobber set; the ARM64 body then modified that lane without a
  save/restore, corrupting its caller and making Lua Sunfish trap out of bounds.
  Preserved-clobber metadata must be regenerated from the final MachineIR after
  every late peephole that can introduce a register definition (sourced).

- 2026-07-22 measurement: running forwarding again after copy propagation and
  carrying a proven-invariant context load through the loop moved
  counter-global 617 -> 560 -> 415 us; the original pass order hid the exact
  store/reload pair until after the only forwarding pass (sourced).

- 2026-07-22 rationale: the whole-function loop passes are deliberately not a
  general optimizer or register allocator. They introduce at most a proved
  edge parameter/value, preserve canonical frame publication, and require every
  loop entry/backedge to establish the same value before removing a reload or
  hoisting address arithmetic (code).

- 2026-07-22 pitfall: proving an invariant register only by absence of
  instruction definitions ignored CFG edge rebinding and made JSON parsing
  non-terminating. Every retained loop proof must check edge arguments on every
  entry and backedge; instruction-local def/use evidence is insufficient across
  a cycle (sourced).

- 2026-07-22 rationale: recognizing Rust's overlap-safe byte-copy loop is a
  strict whole-function semantic pattern, not generic loop vectorization: the
  matcher proves the direction dispatch, forward and backward loops, byte
  load/store shape, and common length before replacing it with one MachineIR
  `MemoryCopy` (code).

- 2026-07-22 measurement: canonical memmove recognition reduced JSON parse
  2.729 -> 2.553 ms and reverse-complement 16.43 -> 5.37 us, demonstrating that
  library idiom recovery can dominate local instruction selection when the Wasm
  producer did not emit `memory.copy` (sourced).

- 2026-07-22 pitfall: cached-cell entry loads that appear dead in linear
  MachineIR may establish mutable cache state consumed on another path; deleting
  them as ordinary dead loads trapped word-count. Fixed-role/cached ownership
  remains semantic even when a local def/use scan finds no immediate consumer
  (sourced).

- 2026-05-16 (111053dd) rationale: the load-to-load reuse peephole is run a
  second time immediately after indexed-memory fusion, because fusing address
  arithmetic into indexed loads exposes additional redundant loads that the
  first reuse pass (run before fusion) could not yet see (code).

- 2026-03-14 (3ad22658) rationale: the engine's no-heavyweight-optimizer rule
  is "no heavyweight optimizer as the primary strategy," not "no optimization
  ever": small block-local cleanups (slot forwarding, copy propagation,
  constant folding into operands, conservative load/store forwarding) are
  allowed as long as they stay mechanical, conservative, ownership-preserving,
  and introduce no whole-function dataflow dependence or general register
  allocator (sourced).

- 2026-03-14 (e8fbf460) rationale: the passes fold and rewrite only transient
  (single-use SSA) registers, identified by index at or above first_transient
  = fixed-reg-count + hot-local-count; the pass is handed that transient
  boundary rather than rediscovering liveness, because fixed-role and
  cached-local registers carry meaning across the whole block and must never be
  disturbed (code).

- 2026-03-17 (1b0dcee7) rationale: constant deduplication (replacing repeated
  materializations of one non-zero constant with copies from the first) runs
  before constant-folding-into-operands so a shared non-trivial constant keeps
  its single defining instruction and is copied to consumers, rather than fold
  re-materializing it independently into each consumer; zero is deliberately
  excluded from dedup because fold can inline Imm64(0) for free (str xzr, cmp
  #0, fcmp #0.0), so deduplicating zero would regress (code).

- 2026-03-25 (72c2a8e6) pitfall: copy propagation may elide a move only when
  both source and destination are transient registers in the same bank — a
  move from a fixed or cached-local register into a transient often acts as a
  value snapshot / lifetime separation, not an alias, and eliding it
  miscompiles; a move also cannot be elided across a CallHelper barrier (which
  clears the alias map) unless its destination is provably dead after the
  barrier (found via a SQLite miscompile) (code).

- 2026-03-28 (e0306ed2) rationale: fuse_isel must run after copy propagation,
  because copy propagation rewrites the register indirections that expose the
  adjacent ShrU+And and shift+binop pairs; run before it, the fusion finds
  almost no matches (code).

- 2026-03-28 (e0306ed2) pitfall: the intermediate register an instruction-
  selection fusion eliminates must be transient and provably dead after the
  fused op — cached locals and fixed registers are implicitly live across block
  boundaries, so the within-block liveness scan cannot prove them dead, and
  eliminating one drops a value another block expects to read (code).

- 2026-03-28 (e0306ed2) measurement: on the Coremark func12 CRC loop
  instruction-selection fusion eliminated 128 MachineIR instructions (34
  BitfieldExtractU/UBFX + 94 IntBinaryShifted shifted-operand fusions) (code).

- 2026-03-29 (22c1c30f) pitfall: the per-register tracking shared by copy
  propagation, constant dedup, store-to-load forwarding, and load reuse was
  invalidated through a defined_reg that returns one register, leaving stale
  aliases/constants/tracked-stores on the high half of i64 pair ops that define
  two registers; on 32-bit GP targets the leaked tracking propagated wrong
  values and corrupted SHA-256, fixed by enumerating both dst_lo and dst_hi
  (for_each_defined_reg) — the same single-vs-pair defined-register hazard as
  in the regalloc liveness path (code).
