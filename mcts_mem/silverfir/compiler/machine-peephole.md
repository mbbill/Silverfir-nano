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

- The first per-block traversal also publishes conservative precursor facts
  used to skip later block-local passes that provably cannot match. This is
  scheduling only: the established transformation order remains unchanged,
  and test/debug builds run the original unconditional sequence on a clone and
  require exact MachineIR equality after every block.

## Facts

- 2026-07-22 rejected: a whole-function necessary-condition gate scanned all
  MachineIR instructions before loop analysis and skipped address hoisting when
  no index register appeared in at least two mem0 accesses. The extra scan was
  paid by every multi-block function and repeated serial Pulldown runs centered
  around 116 ms with no improvement over the 106.5 ms accepted baseline. The
  experiment was reverted; candidate pruning must reuse loop-analysis data
  rather than add another whole-function instruction pass (sourced).

- 2026-07-22 rejected: sharing one immutable predecessor/SCC analysis between
  address hoisting and frame-value reuse preserved CFG topology but did not
  remove either pass's repeated natural-loop predecessor closures. Serial
  Pulldown measured 116.4 ms immediately after the optimized rebuild and 109.5
  ms on repeat against the 106.5 ms accepted baseline. The experiment was
  reverted; future loop-region work must eliminate or compactly represent the
  closures themselves rather than merely share their graph inputs (sourced).

- 2026-07-22 rejected: a shared inner-to-outer loop driver computed each
  natural-loop predecessor closure once and immediately ran address hoisting
  followed by frame-value reuse. It avoided retaining every loop set and made
  all seven serial startup cases 1.7-4.6% faster (3.0% geometrically), but it
  changed the established global pass priority: inner-loop frame reuse could
  claim a lane before an outer address hoist saw it. Against the exact parent,
  matrix-multiply execution regressed from 28.42 to 30.12 ms (about 6%), with
  smaller roughly 2% regressions in Mandelbrot and spectralnorm. The experiment
  was fully reverted. Future sharing of the 10.66%-inclusive
  `natural_loop_nodes` hotspot must preserve all address hoists before any
  frame-value reuse, or prove the new priority improves generated code
  (sourced).

- 2026-07-22 rejected: retaining every exact natural-loop membership set from
  address hoisting for later frame-value reuse preserved the original pass
  order and moved Pulldown from about 112.4 to 107.0 ms. It was not retained:
  2x, 4x, and 8x block-count caps captured almost none of the gain, showing
  that expanded nested memberships are substantially superlinear on this
  workload. An O(number-of-loops) cache for contiguous block ranges avoided
  that memory growth, but both rebuilding node slices and consuming the ranges
  through statically specialized iterators regressed Pulldown to 120-130 ms.
  A future solution needs a loop forest/region traversal native to both passes,
  not a wrapper around the current expanded-set APIs (sourced).

- 2026-07-22 rejected: replacing per-loop visited/result allocation with a
  reusable generation-marked scratch buffer required sorting discovered nodes
  to preserve the existing ascending block order. On serial Pulldown this
  regressed 123.66 -> 135.57 ms (9.6%); the experiment was discarded. The
  follow-up removed the sort and allowed discovery order, but still measured
  112.99 and 113.83 ms against a 112.41 ms baseline. The remaining
  `natural_loop_nodes` cost is overlapping predecessor traversal, not primarily
  vector allocation (sourced).

- 2026-07-22 rejected: collecting all loop definitions once and sorting
  indexed-access candidates removed the apparent access-count x
  instruction-count scan in address hoisting, but serial bz2 remained about
  52.1 ms versus the 51.9-52.6 ms post-SCC range. The experiment was discarded
  because the suspected scan was not material on the measured workload
  (sourced).

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

- 2026-07-23 measurement: dead-parameter elimination cloned and then dropped
  every block terminator solely to clear CFG edge arguments before checking
  direct control/call/return uses. Clone and drop accounted for 20.9% of the
  pass in a serial FFmpeg profile. An exact non-edge source visitor preserves
  the distinction between edge-carried dependencies and direct terminator
  reads without materializing the stripped copy; clone/drop disappeared from
  the verification profile. Exact-parent serial bz2 means were 36.84/37.27 ms
  for the parent and 36.84/36.78 ms for the candidate (about 0.65% favorable
  on the pair averages, with overlapping intervals). FFmpeg's complete
  SSA/MachineIR index remained byte-identical and all 356 release tests passed
  (sourced).

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

- 2026-07-22 pitfall and decision: sharing a strongly connected component
  proves only that a CFG edge participates in some cycle; it does not prove
  that a numerically backward edge targets that cycle's natural-loop header.
  The SCC predicate manufactured overlapping pseudo-loops in complex
  components, so structural loop peepholes now accept a backedge only when its
  target dominates its source. Future loop-region work must preserve that
  dominance condition rather than recover the cheaper but weaker SCC test
  (sourced).

- 2026-07-22 measurement: on Pulldown-cmark, the SCC-plus-numeric predicate
  classified 1,325 loop headers and expanded 1,124,306 block memberships,
  including 877,011 memberships in one 2,629-block function. Requiring
  dominance found 289 natural loops and 4,710 memberships for the whole module,
  reducing expanded loop-region work about 239x. Serial startup moved from
  106.54 to 90.68-91.75 ms; SpiderMonkey moved from 2,112.7 to 1,882.9 ms and
  FFmpeg from 7,544.7 to 7,020.9 ms (sourced).

- 2026-07-22 pitfall and decision: the signed 32x32-to-64 pair-multiply
  recovery pass initialized register-value state and scanned every block on
  64-bit targets even though pair-valued i64 MachineIR is produced only for
  32-bit GP targets. The block-local and cross-edge forms are both now gated by
  the target's GP width; target-specific analyses must decline before
  allocating or scanning when their source IR cannot exist (sourced).

- 2026-07-22 measurement: in a 7,421-sample serial FFmpeg profile, the
  32-bit-only sign-extension analysis consumed about 30% of block-local
  peephole time on ARM64. Skipping it reduced that block-local share from 7.71%
  to 6.51% of the profile and moved the full serial FFmpeg criterion mean from
  7,020.9 to 6,943.8 ms (1.10%); Pulldown's smaller movement remained within
  noise (sourced).

- 2026-03-29 (22c1c30f) pitfall: the per-register tracking shared by copy
  propagation, constant dedup, store-to-load forwarding, and load reuse was
  invalidated through a defined_reg that returns one register, leaving stale
  aliases/constants/tracked-stores on the high half of i64 pair ops that define
  two registers; on 32-bit GP targets the leaked tracking propagated wrong
  values and corrupted SHA-256, fixed by enumerating both dst_lo and dst_hi
  (for_each_defined_reg) — the same single-vs-pair defined-register hazard as
  in the regalloc liveness path (code).

- 2026-07-22 correction and decision: the earlier rejection of sharing loop
  graph inputs was conditional on repeated expanded pseudo-loop closures
  dominating both passes. After the dominance-based latch correction reduced
  Pulldown's expanded memberships about 239x, that condition lapsed. Address
  hoisting changes instructions, block parameters, and edge arguments but not
  CFG targets, so address hoisting and frame-value reuse now share one immutable
  predecessor/dominance analysis (sourced).

- 2026-07-22 measurement: sharing the corrected loop graph moved serial
  Pulldown from 91.287 to 88.748 ms (2.78%, statistically significant).
  Serial bz2 measured 46.757 ms and FFmpeg moved 6,943.8 to 6,909.7 ms in the
  favorable direction but within noise; SpiderMonkey was unchanged. The change
  was retained for its clear Pulldown win and removal of duplicate CFG
  allocation without changing pass priority or loop membership (sourced).

- 2026-07-22 rejected: `recognize_memmove::edge_args_are` allocated a temporary
  `Vec<MachineValue>` each time it compared an edge with an expected register
  list, up to six times in a fully matched candidate. Direct zipped comparison
  removed those allocations, but three serial bz2 means were 46.591, 46.441,
  and 46.441 ms against the accepted 46.757 ms baseline; repeated Criterion
  comparisons found no significant change. The experiment was reverted rather
  than retaining unmeasured cleanup. Matcher-local small-vector allocation is
  not a demonstrated explanation for the remaining startup gap (sourced).

- 2026-07-22 measurement and decision: store-to-load forwarding and
  load-to-load reuse each run twice per block, and each invocation allocated a
  fresh tracking vector. Moving the two trackers into the per-function
  `BlockOptCtx` preserves pass order and exact invalidation semantics while
  reusing capacity across passes and blocks. The verification profile no
  longer found either peephole beneath `RawVec::grow_one` (sourced).

- 2026-07-22 measurement: controlled serial bz2 scratch/parent/scratch means
  were 42.664, 42.985, and 42.654 ms, a repeatable roughly 0.75% reduction
  below Criterion's default practical-change threshold. The broader serial
  run measured 84.925 ms for Pulldown, 1,770.8 ms for SpiderMonkey, and
  6,489.5 ms for FFmpeg; FFmpeg improved 2.29% significantly because its many
  multi-block functions amortize the trackers more often (sourced).

- 2026-07-23 rejected: the adjacent initial store-forwarding and load-reuse
  passes were expressed as independent per-instruction transfers and composed
  into one block traversal without changing the required post-indexed-fusion
  reuse pass or post-copy-propagation forwarding pass. A regression test
  preserved sequential tracker invalidation and FFmpeg's full SSA/MachineIR
  dump was byte-identical, but exact-parent measurements did not reproduce:
  bz2 moved 36.66 -> 36.07 ms in one 30-sample pair and 37.49 -> 37.77 ms in
  the repeat; FFmpeg moved 6.450 -> 6.056 s in one order and 6.592 -> 6.739 s
  in the reverse order, with severe thermal outliers. The refactor was fully
  reverted. Keeping the two tiny trackers allocated across blocks is proven;
  combining only one pair of the four ordered memory-value scans is not
  (sourced).

- 2026-07-23 (a60efd62) decision: constant deduplication is already the first
  mandatory block traversal, so it now reports conservative facts about loads,
  stores, moves, address adds, and instruction-selection precursors while doing
  its existing work. The scheduler uses those facts only to skip passes that
  cannot match; it neither combines transformations nor changes their order.
  Test/debug builds retain a zero-release-cost oracle that runs the former
  unconditional sequence on a cloned block and asserts exact output equality,
  guarding future pass-pattern changes against stale feature classification
  (sourced).

- 2026-07-23 (a60efd62) measurement: in adjacent serial FFmpeg profiles,
  `optimize_block` fell from 7.19% to 5.82% inclusive (19.0% relative) and the
  complete MachineIR peephole stage from 15.59% to 14.25% inclusive (8.6%
  relative). FFmpeg's complete SSA/MachineIR/native index remained
  byte-identical across 14,290 functions, and all 356 release tests passed
  with the unconditional oracle enabled. Fat-LTO text grew by 316 bytes.
  Alternating short bz2 timings were strongly order/thermal sensitive, so no
  end-to-end wall-time percentage is attributed (sourced).

- 2026-07-23 rejected: dead-parameter elimination returned immediately when a
  function had no block parameters and skipped direct-use/dependency scans for
  individual parameter-free source blocks. Although FFmpeg contains many such
  blocks, they are concentrated in cheap functions; the expensive functions
  still carry parameters. Two serial FFmpeg candidate profiles measured the
  pass at 4.21% and 3.88% inclusive versus 3.81% for the exact parent, so the
  gates were reverted. Future work on this pass must reduce the weighted graph
  work in parameter-carrying functions rather than specialize empty cases
  (sourced).

- 2026-07-23 rejected: terminator non-edge operands were visited once per
  block and mapped back to parameter nodes instead of testing the tiny
  terminator once per parameter. The first implementation changed output
  because one physical register can name both a linear and a cached block
  parameter; preserving semantics requires marking every equal-register node,
  not one binary-search hit. The corrected version kept FFmpeg byte-identical
  but measured 4.07% inclusive versus 3.81% for the exact parent, so it was
  reverted. Terminators have few non-edge operands and parameter lists are
  short enough that reverse lookup plus duplicate expansion costs more than
  the established small repeated match (sourced).
