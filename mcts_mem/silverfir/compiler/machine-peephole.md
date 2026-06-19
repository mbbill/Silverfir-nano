- A small set of block-local MachineIR peepholes, run after lowering and before
  backend emission, recovers native-quality patterns the fixed-shape lowering
  set up on purpose: constant deduplication, store-to-load forwarding,
  load-to-load reuse, indexed-memory fusion, copy propagation, and
  instruction-selection fusion (`peephole::optimize`).

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

## Facts

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
