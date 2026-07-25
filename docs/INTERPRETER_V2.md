# Interpreter v2: The Folded Stack Machine

Status: design accepted 2026-07-23; shipping in
`sf-nano-core/src/vm/interpreter/` behind the non-default `interp` feature,
on all six backends the JIT supports (§13). Handler generation moved to
BUILD time on 2026-07-25, so the interpreter needs no executable memory and
no longer depends on `sf_jit`. The interpreter shares no code with the JIT
pipeline; reuse remains a deliberate later refactor.

## 1. Goals and non-goals

Goals, in priority order:

1. **Peak interpreter performance on out-of-order cores.** The target is to
   beat the historical fast-interpreter peak (CoreMark 6,251 on Apple M4 with
   a 2.9 MB fusion library) without a pattern library.
2. **Run where the JIT cannot**: platforms that forbid runtime code
   generation (strict W^X, XIP-only deployments).
3. **Tier-0 startup for the JIT**: begin executing a module immediately
   while the JIT compiles it; switch at root-invocation granularity.
4. A whole-pipeline differential oracle as a side benefit (wasm-level
   reference execution, unlike the emulator which validates MachineIR only).

Non-goals for v1:

- No large discovered-fusion pattern library (size and app-dependence are
  the reasons the historical design is not being revived as-is;
  Appendix B.1).
- No register-resident local variables (see §8; the historical l0/l1/l2
  design multiplies handler variants and its measured standalone benefit
  was ~0%; Appendix B.2–B.4).
- No code sharing with the JIT's compile/execute tier during bring-up
  (fast iteration; the JIT must be unbreakable by interpreter work). The
  runtime substrate IS shared from day one: module parsing/decode, `Store`,
  instances, and the `runtime/` layer serve both engines — only
  `middle/`/`machine/`/`arch/` stay untouched.

## 2. Relation to the historical designs

The project's design tree (`mcts_mem/`, nodes under `compiler.alt/`) records
three interpreter generations and their post-mortems. The lessons this
design is built on (the full inherited measurement table, and why the
JIT-only decision was reopened at all, are in Appendix D):

- **Dispatch count is the dominant interpreter cost** on modern cores;
  memory access is second (the -rs final verdict).
- The old fast interpreter eliminated routing dispatches with a fusion
  pattern library: measured 189 M CoreMark dispatches at a 2.9 MB size cost,
  with app-dependent coverage.
- xir (compiler-technology-for-an-interpreter) died on register-permutation
  handler explosion, SSA-edge/parallel-move shuffle dispatches, and pipeline
  complexity. Any new design must show where it structurally avoids all
  three.
- The `preserve_none`/`musttail` C-handler substrate could not port to
  RISC-V/ARM32/MCU targets. The dispatch substrate must be owned by us.

## 3. Execution model overview

Wasm is predecoded, one pass per function, into a fixed-width threaded
instruction stream. The key idea is **operand folding**: pure routing
opcodes emit no instructions at all.

- `local.get x` pushes a compile-time descriptor `Local(x)` on the
  predecoder's symbolic stack. No instruction is emitted.
- `i32.const k` pushes `Const(k)`. No instruction is emitted.
- A semantic op (arithmetic, compare, load/store, …) pops its operand
  descriptors and emits **one** instruction whose operand fields encode
  where each input lives: a frame slot (locals and materialized temps are
  both frame slots) or an inline constant. The result becomes a `Temp`
  descriptor bound to a height-indexed temp slot.
- `local.set/tee x` retro-patches the producing instruction's destination
  field to write `x` directly (**dst-folding**) when sound (§4.2), else
  emits one `mov`.

Dispatch happens only for semantic operations, movs, and control flow.
Measured on the benchmark corpus (§11), this predicts ~150 M CoreMark
dispatches — 21% fewer than the historical full-fusion build — with a
closed, app-independent handler set.

Temps are addressed by wasm stack height. Wasm validation guarantees stack
heights agree at every merge point, so height-indexed temp slots are
control-flow-consistent **by construction**: no SSA, no phi elimination, no
parallel moves. This is the structural answer to xir's shuffle tax.

## 4. Predecoding

### 4.1 Symbolic stack and descriptors

The predecoder maintains a compile-time stack of descriptors:

- `Local(i)` — a pending, unexecuted `local.get i`.
- `Const(k)` — a pending constant.
- `Temp { height, def, region }` — a value produced by emitted instruction
  `def`, materialized at the temp slot for `height`.

`drop` of a pending descriptor is free. `select`, calls, and all semantic
ops consume descriptors positionally per wasm stack discipline.

### 4.2 dst-folding soundness rules

`local.set x` may patch the producer's destination field iff **all** hold:

1. The top of the symbolic stack is a `Temp` with a known producer
   (call results and block results have no patchable producer).
2. The producer was emitted in the **current control region** (§4.3): no
   branch or merge point was emitted since. A branch between producer and
   set may leave the taken path expecting the value at its canonical temp
   slot.
3. `x` was neither **read nor written** by any instruction emitted since
   the producer (patching would change what those accesses observe).
4. The hazard flush for this set (§4.4) emitted nothing. Flush movs execute
   after the producer, so the producer must not already write `x`.

A call does **not** end a region: the callee cannot observe caller locals,
so folding a pre-call producer across a call is sound.

These rules were adversarially reviewed (three independent hostile reviews
of the `foldsim` reference model); rule 3's read half and rule 4's ordering
were review findings.

### 4.3 Control regions and boundary materialization

A monotone region counter increments at every emitted branch (`br`,
`br_if`, `br_table`, `if`, `else`-jump) and at every merge point (`loop`
header, `end`, `else`).

At every boundary that can merge control (loop entry, `if`/`else`, all
branches, `return`, and conservatively every `end`), all pending
`Local`/`Const` descriptors are **materialized**: one mov each into their
canonical height-indexed temp slots. After materialization, every arriving
path agrees on where every stack value lives. Plain `block` entry is not a
boundary. Movs measured at ~9% of dispatches on the corpus, half of them
call-argument staging (§5).

The v1 rule set is deliberately conservative (materialize everything at
`end` even if no branch targets it); a later pass may relax it with
use-def information, but the measured headroom is ~0.1% of dispatches
(non-call boundary movs), so this is not a priority.

### 4.4 Hazard flush

Before `local.set/tee x` is processed, every pending `Local(x)` still on
the symbolic stack is materialized (it captured the OLD value of `x`).
Measured frequency: <1% of movs on the corpus; LLVM output rarely holds a
local's value across an overwrite.

## 5. Frame layout and call convention

```
frame: [ params | locals | temps by height ... ]
```

Locals and temps live in one flat frame; a folded `L` operand and a
materialized `T` operand are both just frame-slot indices at run time. The
distinction matters only to register-residency optimizations (§8) and
statistics.

Calls pass arguments on the stack: the top `|params|` temp slots ARE the
outgoing argument area, and the callee's frame base points at them
(callee params overlap caller arg slots). A call therefore needs:

- Materialization movs only for arguments that are still pending
  descriptors — the residue of folded `local.get`s, not a new cost (the
  historical design paid one full dispatch per argument push).
- One call dispatch.

Argument staging is a fixed-combo target (§9): `stage_args_N` batches 2–3
staging copies per dispatch using the immediate slots, and a list-driven
`flush_args` variant does any count in one dispatch (a data-driven copy
loop inside one handler — loads/stores, no per-arg dispatch). Call-arg
movs are ~4.4% of all dispatches on the corpus (7.7% on call-heavy lua),
so this combo family is worth roughly that much.

Results are written by the callee directly into the overlap area; the
return path emits no movs.

## 6. Instruction format

Fixed-width 32-byte cells (two per cache line):

```
stage A:  [ op:u16 | flags:u16 | pad:u32 | a:u64 | b:u64 | c:u64 ]
stage B:  [ handler ptr or idx*stride    | a:u64 | b:u64 | c:u64 ]
```

- `op` selects the semantic operation; `flags` carry the operand-class
  bits (src-A const?, src-B const?, dst kind) during stage A.
- `a`/`b` are source operands: frame-slot index or inline 64-bit constant
  per the class bits. `c` is the destination slot, branch target, or
  op-specific payload.
- In stage B the class bits move into handler identity (one handler per
  op × class combination — the measured live combinations are small, §11)
  and the first word becomes the dispatch word.

A denser 16-byte format (2-byte opcode index, `idx*stride + base`
dispatch, constants via extension slots) is an open option pending the
dispatch microbenchmark; it halves stream footprint and D-cache pressure
and removes the handler-table load.

## 7. Dispatch

Staged plan:

- **Stage A (correctness)**: safe-Rust dispatch loop over the `op` word.
  Validates predecoder semantics against spectest; performance-irrelevant.
- **Stage B (performance)**: handlers generated by our own encoder under a
  self-defined register contract (the same move the native backend made
  with its VM ABI). No `preserve_none`, no `musttail`, no cross-language
  LTO; portable to every ISA we already emit.

**As built (arm64 first 2026-07-23; all six backends, generated at BUILD
time, 2026-07-25)**: `interp_gen/` emits each target's handler set as
assembler source that `global_asm!` folds into the binary's `.text` (§13).
It is deliberately NOT the JIT's encoder, and it needs no executable-memory
substrate at all. Pointer-in-cell threaded code: the
cell's first word is rewritten to the handler address at link, slot
operands pre-scale to byte offsets, branch targets to cell byte offsets.
Tail is `add pc, #32; ldr; br` (a pre-index `ldr [pc, #32]!` form measured
no better: the writeback µop serializes against the load). Calls and
returns run natively on a contiguous overlapped-frame value stack — the
callee's frame base IS the caller's staged-argument slot, so argument and
result copies vanish; a native return stack holds `(ret_pc, frame,
code_base)` records, and Rust plants a sentinel record whose cell routes
`Return` back to the driver at activation boundaries. Everything without a
native handler (div/rem, float ops, conversions, bulk memory, table ops,
`call_indirect`, host calls) exits with a reason code; the driver maps pc
back to the function via a range map, executes the ORIGINAL instruction
with `exec_ins`, and re-enters — one uniform slow path covering rare ops,
imports, and rich trap messages. (The list of what lacks a native handler
is per target: arm64 and x86-64 cover div/rem, float, bulk memory and both
call flavours natively; the 32-bit backends do not — see §13.2.)

Stage-B design points, each an A/B axis in the trace-driven dispatch
microbenchmark rather than an assumption:

- What the dispatch consumes: handler pointer in-cell vs 2-byte index +
  table vs index×stride+base (LuaJIT-style; no table load; requires our
  generator to place handlers at a fixed stride — trivial for us).
- **Next-handler preload is a per-target knob, default off on big OoO
  cores.** With a good indirect predictor the branch does not wait for the
  loaded target (prediction decouples fetch from the load); preload only
  pays on weak-predictor cores and marginally on mispredict resolve.
- `br_if` uses dual preload + branchless select (`csel`) of the target so
  the indirect branch's source register is ready as early as the
  condition: interpreted-branch entropy is irreducible, resolve latency is
  the controllable half of the misprediction cost.
- Handlers keep zero shared mutable per-instruction state: pc is the only
  per-instruction register that advances; no runtime sp exists (heights
  are static); no flags thread between handlers; frame slots are
  full-width u64 (store-forwarding friendly).

## 8. Register roles

The governing law, learned from xir's post-mortem and re-derived in this
design round: **registers are free; statically addressing them is what
costs** — every register-resident *operand class* multiplies the handler
set. Appendix B.2–B.4 works the arithmetic and costs the three ways one
might try to escape it; Appendix C.6 states the resulting triangle.
Therefore:

- Fixed-role registers only: pc, frame base, ctx, mem0_base, mem0_size,
  dispatch base, plus the value windows below. These do not participate in
  operand encoding.
- **gp window: 2 slots; fp window: 3 slots** (corpus-measured: temp depth
  at semantic ops ≤2 covers 98.5–100% on integer code; c-ray's float
  expression trees need 3). Depth variants multiply only T-involving
  handlers (~16% of binop operands), so the multiplier is mild.
- Type-disjoint banks don't multiply: int ops touch only the gp window,
  float ops only the fp window. This also fixes the historical design's
  float-through-GPR defect.
- **No register-resident locals.** Adding l0..lN operand classes is the
  xir explosion. A fill/spill scheme instead trades a hidden L1 load for a
  full dispatch — measured backwards. The one bounded escalation knob is a
  single l0 class (~×1.5 variants) IF the trace microbenchmark shows the
  short-span local-read latency (§11) actually binds; undecided, default
  off.

## 9. Fixed ISA combos

A small, closed, app-independent set of super-instructions — ISA design,
not pattern discovery. Corpus-measured value:

| combo | mechanism | saves |
|---|---|---|
| `cmp_br` family | compare + `br_if`/`if` in one handler; resolves earliest | 10.3% of dispatches |
| `dec_and_branch` | `i±k; store; compare; branch` (loop control cluster); the local round-trip becomes handler-internal registers | part of induction's 7.1% |
| `stage_args_N` / `flush_args` | §5 | up to ~4.4% (7.7% call-heavy) |

**Implemented (2026-07-23)**: the `cmp_br` family — 20 fused ops
(`I32_BrEq` … `I64_BrGeU`), produced by the predecoder rewriting a
just-emitted, unfolded, same-region compare in place when its consumer is
`br_if`, `br_if_not`, or an `if` guard (guards use the inverted sense,
which is closed within the set). `i32.eqz` folds twice: over a compare it
inverts the compare; as a branch condition it flips the branch sense.
Measured on CoreMark: −8.9% dispatches (25.16G → 22.92G). `dec_and_branch`
and arg staging remain open.

Loads/stores fold their address arithmetic via operand modes (base slot +
static offset), which is encoding, not a combo.

## 10. Co-existence with the JIT

- The runtime architecture is common: one `Store`/instance/entity model
  and one `runtime/` layer host both engines; the engines differ only in
  how a function body is compiled and executed. The `interp` feature is
  non-default; JIT builds are byte-identical with it off.
- Tiering preserves the native backend's hard invariant (a root invocation
  never mixes execution engines): the interpreter runs whole root
  invocations while the JIT compiles in the background; the switch to
  native happens at the next root call after compilation completes. Both
  directions stay pure.
- The interpreter's call/loop-back handlers can count invocations nearly
  free, feeding lazy-compile priority later.

## 11. Measured basis (foldsim v4)

All quantitative claims above come from `tools/foldsim`, a simulator of
this predecoder run over the full `benchmarks/wasi` corpus (2,613
functions, 10 modules, 100% simulated under hard stack-height
invariants; zero bail-outs). The tool survived three independent hostile
reviews (wasm-semantics, accounting, model-validity); ~15 findings were
fixed across four data iterations, several of them headline-changing —
treat any regeneration of these numbers with the same skepticism.
Appendix E records the tool's method, which findings moved which numbers,
and the full result table; Appendix A explains how to weigh a corpus
measurement against a historical or modelled one.

Key v4 aggregates (weighted by the 10^loop-depth static heuristic):

- dispatch / old-interpreter-basis ratio: **0.445** (CoreMark 0.489 →
  ~150 M predicted dynamic dispatches vs the historical 307 M unfused /
  189 M full-fusion).
- `local.get` folded 95.8%; `local.set` dst-folded 75.1%; `tee` 98.9%;
  consts 97.7%.
- binop operand classes: LC 47.2%, LL 29.3%, LT 9.2%, TC 7.6%, TL 3.3%,
  TT 3.1% — the live variant set is small. Stores are their own family
  (addr 71% L), 13.5% of semantic dispatches.
- temp def→use span 1 = 94.9% (accumulator-friendly); >4 ≈ 0.1%.
- movs 9.0% of dispatches (half call-arg staging, §5).
- short-span (≤2 dispatches) local reads: 30.6% of consumed local reads;
  per-function top-3 locals cover ~70% of them (the open l0 question).

Known limits of the data: the 10^loop-depth weighting is blind to call
frequency and recursion (fib/lua dynamic projections are weak evidence;
loop-dominated workloads are robust — CoreMark's weighted and unweighted
ratios agree). Static-weighted ≈ dynamic is a heuristic; final arbitration
belongs to the real prototype.

## 12. Staging plan

- **A1 — DONE (2026-07-23)**: predecoder in `vm/interpreter/predecode.rs`.
  Coverage: i32/i64/f32/f64 ALU + compares + conversions (trapping and
  saturating), locals/consts (folded), memory loads/stores of every width,
  memory.size/grow/fill/copy/init + data.drop, globals, `select`,
  block/loop/if/else/br/br_if/br_table, call/call_indirect, return,
  unreachable, multi-value blocks. Unsupported ops fail with a clean error:
  reference-typed ops, table.* mutation, SIMD, EH/GC, return_call, and
  br_table forms needing per-target value moves.
- **A2 — DONE (2026-07-23)**: executor in `vm/interpreter/exec.rs`:
  explicit activation-stack trampoline (call depth is interpreter data,
  never host recursion — the classic-interpreter lesson), full trap
  semantics (div/rem edge cases, exact trunc boundaries, OOB, call-stack
  exhaustion), wasm float semantics (NaN-propagating min/max,
  ties-to-even nearest), and self-contained instantiation (globals, active
  data/element segments, funcref tables) directly from the parsed module.
  Validated by 23 unit tests plus 9 differential tests against the JIT on
  identical modules (`tests/interp_diff.rs`) — recursion, a memory sieve,
  float escape iteration, br_table, call_indirect, multi-value, bulk
  memory, and the real `benchmarks/wasi/fib/fib_min.wasm` binary all agree
  with the JIT bit-for-bit.
- **A3 — DONE (2026-07-23)**: reference/table ops (ref.null/func/is_null,
  table.get/set/size/grow/fill/copy/init, elem.drop) and the imports/host
  boundary (`InterpInstance::set_host`, a `(module, name, memory, args,
  results)` dispatcher). End-to-end proof (`tests/interp_wasi.rs`): the
  interpreter runs the real `benchmarks/wasi/coremark/coremark.wasm` binary
  through a minimal WASI shim — CoreMark completes and prints "Correct
  operation validated" — and the sha256 benchmark completes (ignored by
  default: ~5 s native, minutes on stage-A-only builds). Multiple memories
  were added during spectest hardening (memidx packs into the static
  offset's high bits). Deliberately deferred, with clean errors: SIMD (the
  v128 window is stage-B fp-file budget, as in the old design), exception
  handling / GC (3.0), tail calls, memory64.
- **B substrate — DONE (2026-07-23, arm64)**: native dispatch chain as
  described in §7 "as built", with native calls/returns on overlapped
  frames, native `select`/`br_table`, the `cmp_br` combo family (§9), and
  an in-chain dispatch counter + per-op slow-exit profile as permanent
  observability. The planned B1 trace microbenchmark was superseded by
  measuring the real thing (per the project rule: microbenches mislead,
  real workloads decide). Validation: the full battery — 333 unit tests,
  9 three-way differential tests (stage A vs stage B vs JIT), 21001
  spectest asserts with 0 failures, CoreMark + sha256 end-to-end — all
  through the native chain, plus stage-A parity legs.
  **Measured (M-series macOS, release, 5×/config, mean±σ)**: CoreMark
  stage B 4227.7±39.4 vs stage A 837.9±8.2 (5.05×) vs JIT 39314.3±505
  (stage B = 10.8% of JIT). Dispatch physics: 22.92G dispatches for the
  run, 0.73 ns ≈ 2.3 cycles each — the chain is throughput-bound on the
  dispatch count, not on handler memory traffic, so remaining headroom is
  in §9's open combos (dec_and_branch, arg staging) and §8's window
  knobs, each governed by how many dispatches they remove.
- **B verification against §11 (2026-07-23, dynamic, real pipeline)**:
  per-cell old-basis accounting (`PredecodedFunction::basis`, foldsim's
  denominator definition: block/loop/end/else/nop/drop are structural) and
  a stage-A dynamic profiler now measure the design KPIs on real runs.
  CoreMark, 11000 iterations: 3.904G dispatches / 9.387G old-basis ops =
  **dynamic ratio 0.416** vs the static-weighted prediction 0.489 —
  better than predicted (the fused branch family and eqz folding postdate
  the foldsim model). Movs 10.6% of dispatches (MovSlot 8.0 + MovConst
  2.6) vs predicted 9.0. Per iteration: 355K dispatches vs 853K old-basis
  ops. **The failed assumption**: the historical no-fusion interpreter ran
  those 853K old-basis dispatches per iteration at a comparable score —
  ≈0.94 cycles per dispatch effective, against our measured 2.3 — because
  its TOS + register-resident locals kept values out of memory while our
  every operand round-trips through the frame. Dispatch-count reduction
  delivered (0.416 < 0.489 ✓); the "similar per-dispatch cost" premise did
  not survive contact. Consequence: §8's parked register-residency knobs
  (acc window, l0 class) were promoted from "default off" to the next
  measured experiment. (The dynamic profiler that produced these numbers
  lived in the stage-A loop and was removed with it — see below; the
  ratio verification stands as recorded here.)
- **Stage A REMOVED (2026-07-23)**: after B validation the stage-A driver
  loop, its dispatch-mode toggle, and the three-way diff legs were
  deleted at the owner's direction — one high-performance interpreter,
  not two execution paths. What remains of "stage A" is `exec_ins`,
  which was never optional: it IS the native chain's slow path, so the
  semantic dual-maintenance (native handlers vs `exec_ins`) is inherent
  to the fast/slow split, not a removable duplicate. Consequences: hosts
  without executable memory and targets without a backend now fail
  instantiation with a clean error (they get an engine again when
  build-time handler generation lands); the correctness oracle is the
  JIT (differential tests) plus spectest.
- **acc window v1 — DONE (2026-07-23)**: single accumulator (x8), the §8
  gp window at width 1. Chosen over width 2 on the corpus data: temp
  depth ≤1 covers 95.1% aggregate vs ≤2's 98.5% — the second slot buys
  ~3.4pp of coverage for ~2.7× the variant space (TT binops are 0.1%).
  Mechanism: the predecoder retro-marks adjacent producer/consumer pairs
  (same guard as cmp_br fusion: producer at len-1, same region — sound
  because every merge bumps the region — dst unfolded) with three flag
  bits (`FLAG_A_ACC`/`FLAG_B_ACC`/`FLAG_DST_ACC`); slot fields stay
  valid, so the hints are droppable. The linker honors a pair only when
  both sides have native handlers and strips the hints otherwise, which
  keeps `exec_ins` acc-oblivious. Handler emission became class-driven
  ({slot, const, acc}² × dst {mem, acc}, key = flags & 0x1F, ~1350
  handlers ≈ 50 KB); the acc side of an operand or destination costs
  zero instructions. Composes with fusion: a fused compare-branch can
  consume the acc. Measured: CoreMark 4227.7±39.4 → **5454.6±48.9
  (+29%)**, 13.9% of the JIT. Follow-ups if data ever demands: window
  width 2, fp-side acc with the float handlers.
- **Write-through acc — DONE (2026-07-23)**: every native value handler
  now computes into the accumulator and mem-dst variants store from it,
  so adjacent local read-after-write consumers get the register for free
  (zero new variants). The local edges themselves measured +0.7%±1.5% on
  CoreMark — flat, kept because free. The measurement campaign surfaced
  two lessons that outweigh the feature: (1) sequential comparison under
  thermal drift faked a −8% regression (pair runs, and pair against the
  COMMITTED baseline — sibling-variant pairing was blind to a shared
  loss); (2) the refactor silently dropped the BrTable index acc edge
  (the one marking site that must run after boundary materialization) —
  restored, that single edge is worth ~8% of CoreMark: br_table is the
  dominant mispredicting branch and a register-resident index shortens
  every resolve. Final cold-machine formal: **5528.9±40.6** (vs 5454.6
  pre-write-through). The analysis also sharpened l0's case: adjacent
  local reads feed predicted branches (off the critical path); the
  remaining memory-carried chain is the LOOP-CARRIED local cycle, which
  only a function-scoped register-resident local (l0) can break — next
  experiment, at link time (hottest local per function pinned to a
  dedicated register, ~×2.7 variants, call/return reload via the packed
  return record).
- **l0 class — DONE (2026-07-24)**: one register-resident local per
  function, chosen at LINK time (most-referenced slot; no predecoder
  changes). Operand/dst classes become {slot, const, acc, l0}² ×
  {mem, acc, l0} — 48 dense variants per op, ~3,100 handlers ≈ 110 KB
  (vs the historical 2.9 MB pattern library). Write-through keeps the
  slot authoritative: the slow path is oblivious, the driver reloads the
  register at every chain entry, and calls/returns move it via the l0
  offsets packed into call cells (b high bits, callee; c bits 48-63,
  caller — stamped into the return record's code-base word). A Select
  whose packed dst is the l0 goes slow. Measured: +15.3% paired on
  CoreMark; formal **6485.3±100.5 = 16.5% of the JIT — above the
  historical full-fusion peak of 6,251**, which was goal #1 of §1.
  The loop-carried local dependency cycle analysis (previous entry)
  predicted this: l0 breaks the store→load cycle the adjacency schemes
  cannot reach.
- **Build-time generation and the other five backends — DONE
  (2026-07-25)**: see §13. The runtime micro-encoder is gone; handlers are
  generated per target by `build.rs` and linked into `.text`. arm64 is at
  parity through the port (CoreMark 8179 vs the 8143 baseline, engine
  334,820 bytes vs 334,828), and x86-64, RV64, RV32 and both arm32
  encodings each pass the full 21001-assert spec suite.
- **C**: revisit code reuse with the JIT front half (decoder/validator are
  already shared via `op_decoder`; anything deeper waits until the
  backends settle).

## 13. Build-time generation and the multi-target backends

Handlers are generated at BUILD time and folded into the binary's own
`.text` through `global_asm!`. Nothing is emitted, mapped, or made
executable at run time. Three consequences, in order of importance:

1. The interpreter runs where runtime code generation is forbidden or
   impossible — strict W^X, XIP-only deployments, MCUs — which was goal #2
   of §1 and the reason the tier exists at all.
2. `interp` no longer implies `jit`. The dispatch chain used to be gated on
   the JIT's executable-memory substrate; now `sf_interp_engine` is set
   purely by whether the target has a backend.
3. Engine size becomes a link-time budget rather than an allocation. On an
   MCU it is flash.

### 13.1 What the generator emits, and why it emits text

`interp_gen/` is compiled by `build.rs`, never by the crate. It walks the
handler variant space in `vm/interpreter/layout.rs` and asks the selected
backend to emit each variant, producing one assembler source file plus a
packed table of handler offsets.

It emits **assembly text**, not machine code, for two reasons. The
assembler resolves branch labels — hand-counted branch deltas were this
emitter's most frequent bring-up defect, recorded twice in the design tree
in a single day — and the result lands in `.text` through `global_asm!`
with no executable-memory substrate underneath. The cost is one dependency
on the assembler's mnemonic set, which every target already has.

`layout.rs` is compiled TWICE, once into the crate and once into
`build.rs`. The generator enumerates the variant space and the linker
classifies cells into it from the same source, which is what makes them
agree; a divergence would not fail the build, it would silently demote
cells to the slow path or point one at a handler that reads its operands
from somewhere else. Handler slots are packed per variant FAMILY rather
than as a dense `op x 200` matrix: most ops vary only one or two of the
three operand positions, and the dense form costs ~160 KB of table for
~10.5 k live handlers.

### 13.2 The backends, and what each one gives up

| | arm64 | x86-64 | rv64 | rv32 | arm32 (A32 / Thumb-2) |
|---|---|---|---|---|---|
| operand classes | 5 x 5 x 4 | 5 x 5 x 4 | 5 x 5 x 4 | 4 x 4 x 3 | 4 x 4 x 3 |
| pinned locals | l0 + l1 | l0 + l1 | l0 + l1 | l0 | l0 |
| float registers | NEON | xmm | F/D | — | — |
| native calls | yes | yes | yes | no | no |
| bulk memory block | 64 B NEON | 64 B SSE2 | machine word | machine word | machine word |
| engine size | 327 KB | 370 KB | 362 KB | 93 KB | 118 / 87 KB |
| spectest | 21001 / 0 | 21001 / 0 | 21001 / 0 | 21001 / 0 | 21001 / 0 each |

Engine size is the number that makes the reduced class sets worth having:
the 32-bit backends land at 87–118 KB against the 64-bit ones' 327–370 KB,
which is the difference between fitting a flash budget and not. It is also
the number to watch — every added operand class multiplies it.

Everything switched off above simply has no handler: the cell links to the
slow stub and the shared executor runs it, so a reduced backend is slower,
never wrong. The reductions are each forced by one property of the target:

- **Class-set width follows the register budget.** Dropping `l1` takes a
  three-operand op from 100 variants to 48, which roughly halves the
  emitted blob. That is the right trade where the blob is flash.
- **A 32-bit host makes a wasm value a register PAIR.** The accumulator and
  the pinned local cost two registers each. The 64-bit ops that stay native
  there are the ones a pair handles in a couple of instructions — moves,
  loads and stores, add and sub with carry, the bitwise family, equality.
  Multiply, divide, the variable shifts, the rotates and the ordered 64-bit
  compares go to the shared executor.
- **Calls are slow on the 32-bit backends.** The call protocol packs its
  operands into the high halves of 64-bit cell fields, and threading six
  half-registers through the shared activation path buys less than it costs
  on an MCU profile. `Return` stays native everywhere — it must, because a
  slow return would desync the native return stack — and the two compose
  because the driver plants a sentinel record per activation.
- **x86-64 declines what its BASELINE lacks**: `roundsd` (SSE4.1) for
  ceil/floor/trunc/nearest, and the saturating float-to-int family and
  unsigned-64 conversions, whose fixups cost more than they buy. Every one
  is a full decline, never a runtime bail (§13.3).
- **RISC-V declines what needs Zbb** (clz/ctz/popcnt) and float-to-float
  rounding, which the ISA does not have in any form.

### 13.3 Two invariants a backend must not break

**A handler may bail only on a path that traps.** The accumulator pairing
rests on "producer bails ⟹ execution never reaches the consumer": the slow
path is accumulator-oblivious and writes the frame slot, so a producer that
bailed and then SUCCEEDED would leave its consumer reading a stale
register. This is why an op a backend cannot do fully is declined outright
— a decline lets the linker strip the pair's hints, which a runtime bail
cannot.

**An i32 value is zero-extended in its 8-byte slot.** That convention is
shared with the single-instruction executor, the host boundary, globals and
invocation results, so it is not forkable per target. arm64 and x86-64 get
it free from their 32-bit ops; RV64 pays two instructions to re-establish
it after the arithmetic that sign-extends; the 32-bit backends write a zero
high word. Reading a memory address as 32 bits rather than trusting the
convention is what keeps a bounds check independent of it.

### 13.4 Portability of the measured optimizations

Everything the arm64 engine was tuned into is either structural (and
carried everywhere) or target-specific (and gated):

| lever | ported | note |
|---|---|---|
| pointer-in-cell threading, 32-byte cells | all | structural |
| absolute branch targets, target handler loaded at entry | all | -4.80% of CoreMark cycles on arm64 |
| emission ORDER (cold families last) | all | driven by the shared `emit_order` |
| accumulator, write-through | all | |
| pinned locals | all, width per budget | |
| `cmp_br` fusion, address-add fusion, `MovPair` | all | predecoder-level, target-independent |
| dispatch counter off by default | all | -3.69% of CoreMark cycles |
| `ldp` operand-pair loads | arm64 | pairs the CELL payload; pairing two FRAME reads is unsound on this core |
| wide bulk-memory blocks | arm64, x86-64 | RISC-V and arm32 have no wider move in their baselines |

## 14. Open questions

0. **An interpreter-only build still compiles the JIT's front and middle
   end.** `arch/`, `machine/`, `build/` and `template/` are gated on
   `sf_jit`; `middle/` and `wasm/` are not, so
   `--no-default-features --features interp` links a working interpreter
   but drags the SSA pipeline in as dead code. Nothing is wrong with the
   engine — this is binary size, not correctness — but it is the remaining
   work before "ship only the interpreter" is a real configuration, and it
   is why CI builds the interpreter as `jit,interp` rather than alone.
1. 16-byte vs 32-byte cells (dispatch microbenchmark, B1).
2. index×stride vs pointer-in-cell dispatch (B1).
3. l0 single register-local class: default off, reopen only on B1 latency
   evidence.
4. Relaxed end-materialization: measured headroom ~0.1%, parked.
5. In-order/MCU targets change the cost hierarchy entirely (dispatch
   misprediction cheap, XIP fetch expensive); out of scope for v1, revisit
   if the RP2350 port wants an interpreter tier.

---

# Appendices: supporting data

The sections above state the design. The appendices below record what it was
chosen *against*, the cost model that justifies it, the historical
measurements it inherits, and the full evidence record behind §11.

## Appendix A — How to read the numbers

Four kinds of claim appear in this document. They are not equally strong, and
mixing them is how a design gets steered by bad data. Every quantitative claim
below carries one of these tags:

| tag | meaning | how to falsify it |
|---|---|---|
| **(corpus)** | measured by `tools/foldsim` v4 over `benchmarks/wasi` | re-run the tool |
| **(historical)** | measured on the deleted fast interpreter or xir; recorded in the design tree | not re-runnable — those engines no longer exist. Treat as a dated hypothesis, not a standing result |
| **(model)** | arithmetic from the cost model in Appendix C, using illustrative micro-architectural constants | check the arithmetic; the constants themselves are unmeasured |
| **(open)** | an assumption scheduled for measurement in B1 | the microbenchmark |

Two standing cautions that apply to everything tagged **(historical)**:

- All of it was measured on Apple M4 — one wide out-of-order core with a large
  BTB and a strong indirect predictor. None of it has been reproduced on
  another micro-architecture.
- No test campaign is guaranteed to have been exhaustive. The author's position
  on record: historical data is a hypothesis list, not a conclusion list.

Nothing tagged **(model)** has been measured. The cycle counts in Appendix C
are order-of-magnitude placeholders chosen to make the *direction* of an
argument visible; where a decision depends on their magnitude rather than their
direction, that decision is listed in §14 as open.

## Appendix B — Rejected alternatives

Each entry records what the alternative would have bought, and the reason it
lost. Alternatives are kept rather than deleted: if the reason one lost stops
holding, it can be reclaimed.

### B.1 A large discovered-fusion pattern library

**What it buys.** The historical champion's central mechanism, and the reason
it was the fastest measured wasm interpreter. Fusion has no per-semantic-op
dispatch floor — one pattern can absorb several semantic operations, where
folding always pays one dispatch each. `get/const + get/const + op + set` is
6 wasm instructions and 1 dispatch under fusion, 2 under folding.

**Why it lost.** Two author objections, neither of which the measured data
contradicts:

1. *Application dependence.* Patterns are discovered from a target binary and
   generalize across applications only partially.
2. *Size.* The full 1,500-pattern library is 2.9 MB **(historical)** — an
   order of magnitude larger than the JIT it is supposed to substitute for on
   size-constrained targets. An interpreter that is bigger than a JIT has
   defeated its own purpose.

**The counter-evidence, recorded honestly.** The tree's measurements do not
fully support the generality objection: top-10 `local.get` successors covered
88–92% across two unrelated workloads (CoreMark and Lua fib), and a curated
~100 KB subset recovered 80% of the full library's benefit **(historical)**.
That sample was only four workloads, all LLVM-compiled C. The size argument
stands on its own regardless, so this was not contested further.

**The decisive measurement.** Fusion in practice does not reach its own
theoretical floor. The full 2.9 MB library measured 189 M CoreMark dispatches
**(historical)**; structural folding predicts ~150 M **(corpus)** with no
library at all. Pattern hit rate, not the mechanism's ceiling, is what bounds
fusion. This inverted the design's standing: folding is the *lower* baseline,
and fusion drops from load-bearing wall to an optional increment above it.

**What survived.** Bounded, application-independent, ISA-level combos (§9) —
`cmp_br`, `dec_and_branch`, argument staging. These are instruction-set
design, not pattern discovery: the set is closed and known at build time.

### B.2 Register-resident locals as operand classes (l0/l1/l2)

**What it buys.** The historical design's hot-local cache, and the obvious way
to spend an AArch64 register file that otherwise sits mostly idle. Removes the
frame load from every hot local access.

**Why it lost — handler count is a product, not a sum.** Operand class count
multiplies across operand positions. With locals in the frame bank, every
local is one class (`fp[imm]`, where `imm` is *data*, not handler identity).
Adding l0/l1/l2 makes each register its own class:

| operand classes | binop variants | whole ISA |
|---|---|---|
| `{T, L, C}` (this design) | ~5 commutative / ~8 non-commutative, × dst ~2–3, × depth ~2 | **2,000–4,000 handlers, 150–300 KB (model)** |
| `+ l0/l1/l2` → src 6, dst 5 | 6×6×5 = 180, ~100+ after normalization, × depth | **10,000–30,000 handlers, 1–2 MB (model)** |
| xir's 8 addressable registers | held to ~15 k only by a 2-address constraint | **died here (historical)** |

The second row lands back in exactly the two places this design exists to
avoid: xir's handler count and the fusion library's footprint. Every added
addressable register class is a multiplication, and there is no way around it.

**The second reason: the benefit is not what it was.** In the historical
design a hot local was valuable because each local access carried a dispatch,
and l0 let fusion delete the dispatch *and* the instruction together. Under
folding, local operands generate no dispatch at all — what remains is one
`ldr fp[imm]` inside the handler, whose address is ready the moment the
handler enters the out-of-order window. This also resolves a contradiction the
tree recorded but never explained: **hot-local caching measured ~0% benefit
standalone, and +10% only in combination with fusion (historical)**. The
consistent reading is that residency was never removing latency — the latency
was already hidden — and the +10% came from fusion using l0 to cut instruction
*count*.

**The bounded escalation knob that remains.** A single `l0` class (~×1.5
variants, no runtime movement) if B1 shows short-span local read latency
actually binds. Default off. Three classes is not on the table; the historical
coverage curve has a sharp knee (l0+l1 covered 41.1% of accesses, leaving
58.9% on the generic path **(historical)**), which is likely why the old
design stopped at three rather than by coincidence.

### B.3 Register-resident locals reached via spill/fill instead of operand classes

**What it buys.** The author's proposal to keep locals in registers without
paying the permutation cost: handlers recognize only `{T, L, C}`, and explicit
`fill_lN` / `spill_lN` instructions move values in and out. Trades runtime
work for handler count — normally a good trade.

**Why it lost — the exchange rate is inverted here.** A fill or spill is a
*full dispatch* (~4 scaffold instructions plus a taken branch). What it
replaces is one L1 load whose address is ready at issue. Scaling by the
historical access profile: locals are ~38% of the instruction stream, and
l0+l1+l2 covers roughly half of those accesses **(historical)** — so ~19% of
the stream moves from zero dispatches to one. On CoreMark that is
**150 M → ~213 M dispatches (model)**, worse than the 2.9 MB fusion library's
189 M.

The historical spill/fill mechanism worked because it was *rare* — 3.1% of
dispatches **(historical)**. The same mechanism applied to a ~40% frequency
event changes character entirely.

**What survived.** Spill/fill stays as the escape valve for window overflow
and control-boundary materialization — a few percent of dispatches, its
original role — not as the routine path to a local.

### B.4 An index-addressed register file

**What it buys.** The most ambitious version of the idea: keep ~20 locals
permanently in registers, with the local index arriving in a register, and
find some way to select a register by runtime index. If it worked, local
access would involve no memory operation at all.

**Why it lost.** Three implementation paths exist. All were costed against the
thing they replace — a single `ldr x8, [fp, x9, lsl 3]`: one instruction, one
issue slot, no branch, address ready at window entry, frame line resident in
L1, ~4 cycle latency usually hidden in pipeline depth.

| path | cost per operand **(model)** |
|---|---|
| branch tree (`if idx<8 …`) | 4–5 levels for 20 targets; ~4–5 instructions, ~2 taken branches |
| jump table + `mov`/`b` stubs | 1 table load + 1 indirect branch + 2 stub instructions; ~4 instructions, 2 taken branches |
| `TBL` over vector registers | 16 locals in 8 `v` registers; 3–5 instructions, 5–8 cycle latency chain; writes require dynamic lane insert, much worse |

None wins on instruction count, bandwidth, or latency. The compact statement:
**L1 is the register file this machine provides with an indexed read port.**
The architectural register file has no index port, and every software
emulation of one costs more than the load it replaces.

**Why "the branches will predict well" does not rescue it.** The premise is
partly right — a given program point's local index is a compile-time constant,
perfectly correlated with path history, so an ITTAGE-class predictor should
learn it. Two problems remain:

1. **A correctly predicted branch still costs fetch bandwidth.** The entire
   design is built around the ~1 taken branch per cycle fetch limit (Appendix
   C.1). These paths turn one load slot into 2–5 taken branches per operand.
2. **History pollution is a global cost.** Dispatch prediction depends on the
   global history holding the identity of the last several dozen handlers.
   Injecting 1–5 selection branches per operand dilutes the effective dispatch
   depth in that history, degrading prediction on the *main* dispatch chain —
   a cost paid on every dispatch, not only on local accesses. The jump-table
   variant additionally doubles the BTB working set. This second-order effect
   is the more serious of the two and is **(open)** — B1 should measure main
   dispatch prediction rate with and without selection branches mixed in.

**The underlying law.** This and B.2 are the same wall seen twice:

> A shared handler requires every register's meaning to be constant across the
> entire dispatch chain.

Making a register's meaning vary by program point has exactly three
implementations: generate code per program point (that is a JIT), generate a
handler variant per meaning (multiplicative explosion), or select at runtime
(the three paths above). There is no fourth.

The design tree contains the inverse of this statement in micro-JIT's origin
record: *TOS + l0/l1/l2 had already pinned every hot value to a fixed physical
register, so the JIT needed no register allocator — it was a template
assembler* **(historical)**. The complete form of "use all the registers" is
the JIT itself. The interpreter/JIT boundary falls exactly on whether register
meaning varies with program point, and building an interpreter means choosing
this side of that line deliberately.

Residual honesty: after the fixed roles in §8 are assigned, many GPRs will
still be idle during most handler executions. That is a structural property of
a shared-handler interpreter, not an oversight.

### B.5 Pure 3-address with an accumulator

**What it buys.** The design that preceded the folded stack machine, and still
the control group for B1. Operands are position-independent frame slots, which
buys one thing the folded stack machine structurally cannot have:
**predecode-time list scheduling**. A stack machine's operation order *is*
stack order — moving an instruction requires inserting a stack shuffle, and
each shuffle is a dispatch. With position-independent operands the predecoder
could interleave independent dependency chains, hoist linear-memory loads
early, and (later) software-pipeline hot loops.

It is also much smaller: a few hundred handlers versus 2,000–4,000.

**Why it lost.** Every intermediate value edge travels through a frame slot —
store, store-to-load forward, load — where the folded stack machine's TOS
window keeps short-span edges in registers. The accumulator recovers span-1
edges but not span 2–4. On the worked serial example (Appendix C.3) that is
the difference between ~9 and ~17 cycles per iteration **(model)**.

**What tipped it.** The author's observation that a 3-address form generated
*solely for local elimination* keeps stack discipline: if both operands are on
the stack they are necessarily adjacent, so their position is implied by
height rather than encoded. The class space collapses to roughly ×12 instead
of exploding — the folded stack machine gets the register edges *and* a
bounded handler set, giving up only scheduling freedom.

**What was given up, and why that is acceptable.** The value of predecode-time
scheduling is unproven: LLVM has already scheduled its wasm output, and this
predecoder must stay fast enough to serve as tier-0 startup (§B.6). Reopen if
B1 shows frame-slot edge latency dominating on real traces.

### B.6 SSA-based predecoding

**What it buys.** Cross-block value propagation and stronger folding than a
single-pass symbolic stack can achieve.

**Why it lost.** The cost is precisely xir's cause of death: merge points need
phi elimination → parallel moves → **one dispatch per move**, plus dominance
and liveness analysis inside the predecoder. Height-indexed temps get
control-flow consistency for free from wasm's own validation rules, which is
the entire structural advantage.

A second constraint reinforces it: this interpreter doubles as tier-0 startup
(§10), so predecoding must be dramatically cheaper than JIT compilation or the
startup advantage evaporates. Single-pass symbolic-stack throughput is roughly
"read the bytecode once." That is a feature of the design, not a shortcut.
Workloads that genuinely need cross-block optimization have the JIT pipeline
available.

### B.7 Interpreting MachineIR / reusing the middle end

**Why it lost.** All three of xir's causes of death apply unchanged to today's
middle end — parallel moves are still one dispatch each on an interpreter, the
register model still permutes, and the pipeline complexity is still there. The
`emulator` is already a MachineIR interpreter and is explicitly positioned as
a debugging oracle that will never be a production engine.

### B.8 What `wasm3` does and does not already do

Recorded because "isn't this just wasm3?" is the obvious question, and half
the answer is yes.

**Shared.** wasm3 maintains a compile-time symbolic stack whose entries may be
constants, slot references, or "in r0". `local.get` pushes a slot alias and
emits nothing, so local reads are already free. A `local.set` onto a local
with unconsumed aliases emits a preserve/copy first — the same hazard flush as
§4.4.

**Not shared, and this is the delta that matters.** wasm3 is not 3-address.
Its execution model is anchored on a single result register (`r0`/`fp0`), so
a computation's result location is fixed by the handler, not chosen by the
instruction. Consequently `local.set` is a real `SetSlot` dispatch and **there
is no dst-patching** — the producing operation cannot be retargeted to write
the local directly. Temp slots come from a dynamic allocator rather than being
height-indexed, so merge-point agreement relies on allocator state matching
rather than on structural alignment.

Dst-folding is what the 3-address form unlocks, and §11 measures it at 75.1%
of `local.set` **(corpus)**. This also explains the author's earlier
experimental result that `local.set` elimination rates were disappointing:
without a patchable destination field, every set costs a mov.

This subsection is from recollection of `m3_compile.c` **(open)** — worth
confirming against the source before it is cited anywhere load-bearing.

## Appendix C — The cost model

The model out-of-order reasoning in this document rests on. All constants are
illustrative **(model)**; B1 replaces them with measurements.

### C.1 What bounds interpreter throughput

Assume a modern out-of-order core with a trace/path-correlated indirect
predictor, so dispatch branches predict well. Prediction hitting means the
control-flow scaffolding largely leaves the critical path: the predictor
supplies the target at fetch, and the loaded handler pointer is consumed only
at branch resolve. Scaffolding cost is therefore **bandwidth, not latency**.

Throughput is then the minimum of three quantities:

1. **Taken-branch fetch limit.** Each dispatch is a taken indirect branch,
   which breaks the fetch stream. Most cores sustain ~1 taken branch per cycle
   (a few newer ones, 2). This caps dispatch rate at roughly 1/cycle and gives
   the design its central consequence: **the useful work carried per dispatch
   determines how wide a machine you can feed.** A handler with 1 useful
   instruction cannot exceed ~5 IPC on a 10-wide machine, most of it
   scaffolding; 4 useful instructions is what it takes to approach ~8. This —
   not the raw dispatch saving — is the real argument for dispatching on
   semantic operations rather than routing operations.
2. **Total instruction bandwidth.** Scaffolding inflates instruction count
   4–5×; rarely the binding constraint on a wide core.
3. **The interpreted program's own dataflow critical path**, whose edge
   latency depends on where values live. This is the one that decides
   architecture, below.

### C.2 Value transfer latency by carrier

| carrier | latency **(model)** |
|---|---|
| register edge (TOS window, acc) | 0–1 cycles — renaming removes false dependencies; the real chain is the program's own |
| frame slot edge | ~4–6 cycles — store → store-to-load forward → load |

With a dispatch initiation interval of ~3 cycles, a fully serial dependency
chain costs ~5 cycles/step through frame slots (edge latency dominates) versus
~3 cycles/step through registers (dispatch interval dominates) — roughly 1.7×.
In ILP-rich code, independent chains overlap, edge latency hides, and both
degrade to a contest over dispatch count and bandwidth, which the folded form
wins. **Each form wins a different workload class; the dividing line is the
interpreted program's serialization.**

### C.3 Worked example — serial chain

```c
x = ((x ^ k) * 33) + 7;   // single accumulator, every step dependent
```

LLVM emits 8 wasm instructions: `local.get $x`, `local.get $k`, `i32.xor`,
`i32.const 33`, `i32.mul`, `i32.const 7`, `i32.add`, `local.set $x`.

Constants used: store→load forward 4, `mul` 3, `add`/`xor` 1, ≤1 taken branch
per cycle. All **(model)**.

**Pure TOS, no fusion — 8 dispatches.** Intermediates stay in registers, but
`x` crosses iterations through the frame:

```
get x    : t0 = load fp[x]      ← previous iteration's store, forward 4cy
get k    : t1 = load fp[k]      (loop-invariant, off the chain)
xor      : t0 ^= t1             1cy
const 33 : t1 = 33              (off the chain)
mul      : t0 *= t1             3cy
const 7  : t1 = 7               (off the chain)
add      : t0 += t1             1cy
set x    : fp[x] = t0
```

Chain 4+1+3+1 ≈ **9 cy**; dispatch bound ≈ 8 cy. Actual ≈ **9 cy/iteration** —
both constraints near-binding.

**Pure 3-address — 3 dispatches**, every intermediate edge through a frame slot:

```
xor  t1, x, k     : load fp[x] (fwd 4) → xor 1 → store fp[t1]
mul  t1, t1, #33  : load fp[t1](fwd 4) → mul 3 → store fp[t1]
add  x,  t1, #7   : load fp[t1](fwd 4) → add 1 → store fp[x]
```

Chain (4+1)+(4+3)+(4+1) ≈ **17 cy**; dispatch bound 3 cy, entirely buried.
Actual ≈ **17 cy/iteration** — 62% fewer dispatches and nearly 2× slower. The
saved dispatches idle while the machine waits on forwarding.

**3-address + acc — 3 dispatches**, the two single-consumer edges in registers:

```
xor  acc, x, k     : load fp[x] (fwd 4) → xor 1     (no frame store)
mul  acc, acc, #33 : register edge → mul 3
add  x,  acc, #7   : register edge → add 1 → store fp[x]
```

Chain 4+1+3+1 ≈ **9 cy** — level with TOS at 3 dispatches instead of 8.
Carrying `acc` across the back edge would remove the frame round trip too,
giving 1+3+1 ≈ **5 cy**; that is where remaining headroom lives. The
historical full-fusion + l0 build reached the same level on this shape, but
via a 2.9 MB pattern library rather than an ISA-level rule.

### C.4 Worked example — ILP-rich

```c
s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3];   // four independent chains
```

Five wasm instructions per chain, 20 total.

**Pure TOS — 20 dispatches.** The chains overlap freely in the out-of-order
window, but fetch crosses only ~1 taken branch per cycle:

- dispatch bound ≈ **20 cy** ← binding
- instruction bandwidth: 20 handlers × ~5 instructions ≈ 100, ~10 cy at 10-wide
- dependency: load 4 + add 1 per chain, fully hidden across four chains

**Folded — 8 dispatches** (addresses folded into loads, locals folded into
operands): dispatch bound ≈ **8 cy**, bandwidth ≈ 6–7 cy, dependencies hidden.
Roughly **2.5× faster**, and the accumulator neither helps nor hurts here.

| | C.3 serial | C.4 parallel |
|---|---|---|
| pure TOS | ~9 cy | ~20 cy |
| pure 3-address | ~17 cy | ~8 cy |
| 3-address + acc | ~9 cy (≈5 with loop-carried acc) | ~8 cy |
| folded stack machine | ~9 cy | ~8 cy |

Serial code is bound by **edge latency** — whoever keeps values in registers
wins. Parallel code is bound by **fetch bandwidth across taken branches** —
whoever dispatches less wins. The folded stack machine takes the better side
of both.

### C.5 The span argument

Why a small window recovers most of what full register residency would offer.
Define an edge's **span** as the number of dispatches between definition and
use:

| span | exposure | covered by |
|---|---|---|
| 1 | full edge latency exposed | TOS window / acc — register edge, no loss |
| ≥2 | ≥1 intervening handler ≈ ≥3–6 cycles of other work; 4-cycle forwarding already hidden | the out-of-order core, for free |
| loop-carried | spans the whole loop body, tens of cycles | invisible |

Corpus data supports the shape: temp def→use span 1 is 94.9%, span >4 ≈ 0.1%
**(corpus)**. So a 2–3 slot window plus the machine's own latency hiding
captures nearly all of the value that a full register file would.

The residual is genuinely open: short-span (≤2 dispatch) local reads are 30.6%
of consumed local reads **(corpus)** — higher than expected, and the reason
the l0 knob is still listed in §14 rather than closed. Concentration is low
(per-function top-1 local ≈ 30%, top-3 ≈ 70% of short reads), which bounds any
single-register scheme's upside. `dec_and_branch` absorbs the densest cluster
(loop control: `i++; i<n; br` — written then immediately read) inside one
handler, which is the cheapest available answer.

### C.6 The bounded-handler triangle

Three properties, any two:

1. **Explicit operand addressing** — the precondition for eliminating routing
   dispatches.
2. **Operand register residency** — the precondition for eliminating edge
   latency.
3. **Bounded handler count** — the precondition for acceptable size.

| design | picks | pays |
|---|---|---|
| historical `fast` | 2 + 3 | routing elimination only via a fusion library |
| pure 3-address | 1 + 3 | every edge through memory |
| xir | 1 + 2 | ~15 k handlers — died here **(historical)** |
| **folded stack machine** | 1 + 3, **plus** implicit-position stack operands | scheduling freedom; ~2–4 k handlers |

The fourth corner exists only because stack discipline makes the TOS
operands' positions *implicit* — they are the top of the stack by
construction, so they cost no encoding and create no operand class. That is
the whole trick, and it is why depth variants grow linearly (×2–4) where
addressable registers grow multiplicatively.

## Appendix D — Inherited measurements

### D.1 Why the JIT-only decision was reopened

The interpreter lineage was deleted 2026-04-07 for two recorded reasons. Both
were re-examined; each fails under today's constraints for a specific,
nameable reason.

| recorded reason | why it no longer binds |
|---|---|
| **Ceiling.** Interpreter peak ≈ Winch; it can never catch an optimizing JIT, so maintaining a permanently-losing tier is not worth it. | The argument holds only where interpreter and JIT *compete on the same platform*. The target here is platforms where the JIT cannot run at all (strict W^X, XIP-only) plus tier-0 startup. Non-competing tiers are not subject to it. |
| **Portability wall.** Dispatch depended on clang's `musttail` + `preserve_none`, which exist only on x86-64/AArch64 — RISC-V, ARM32 and MCU targets were unreachable. | The premise was that handlers must be compiled by a C compiler. The project now owns encoders for six ISAs and has shipped a custom VM ABI with global-asm trampolines in the native backend. Generating handlers ourselves removes the dependency entirely. |

Unchanged: the Rust toolchain is pinned at stable 1.97 and `become` (explicit
tail calls) is still unavailable. No part of this design may depend on it.

### D.2 The historical record this design builds on

All **(historical)**, all Apple M4.

| measurement | value | what it decides here |
|---|---|---|
| `fast` peak | CoreMark **6,251** — fastest measured wasm interpreter | the bar to beat (§1) |
| vs. wasm3 | 1.7–2.5×, geometric mean 2.0× | — |
| vs. JIT tiers | 67% of Winch; 38–62% of optimizing Cranelift | the ceiling argument in D.1 |
| CoreMark dispatches | **307 M** unfused → **189 M** full fusion | the baseline §11 compares against |
| fusion ablation | fusion-only 1.70–2.04× faster than hot-local-only | fusion was the load-bearing mechanism, not l0 |
| hot-local standalone | **≈ 0%**; +10% only combined with fusion | B.2 — residency was never removing latency |
| l0+l1 coverage | 41.1% of accesses; 58.9% always generic | B.2 — the coverage curve's knee |
| TOS window | 4 registers, mod-4 variants; spill/fill **3.1%** of dispatches | B.3 — why spill/fill worked there and not as a routine path |
| nh + guard classification | 89.1% of handlers always-linear | §7 — and why nh looked good on this core |
| locals share of stream | ~38% | B.3 arithmetic |
| sizes | core ~230 KB (no fusion, 40% slower ≈ wasm3); +100 KB recovers 80%; full 1,500 patterns **2.9 MB** | B.1 — the size objection |
| top-10 `local.get` successors | 88–92% coverage on two unrelated workloads | B.1 — the counter-evidence, recorded |
| xir handler count | ~15 k at 8 abstract registers, held down only by a 2-address constraint; final score ~90% of wasm3 | B.2, C.6 — the explosion boundary |
| runtime register selection microbench | 8-element array index **1.7–17 ns**; nested ternary **3.4 ns** | B.4 — runtime selection was already measured and rejected once |
| profiler observation | Instruments/VTune showed dispatch rate far below theoretical ceiling, never explained | §11 — the unexplained headroom this design is aimed at |

The last row is the most interesting inheritance: the historical design pushed
all three classical interpreter levers (fewer dispatches, less memory traffic,
smaller handlers) to their limits and still did not saturate the machine. The
hypothesis this design pursues is that the remaining headroom is in
cross-handler ILP and fetch-bandwidth effects — quantities that only become
controllable once handler generation is owned rather than delegated to a C
compiler. That hypothesis is **(open)**; B1 is its first test.

### D.3 A deferred lever: handler replication

Recorded because it was raised, deliberately deferred, and does not affect any
encoding or handler-set decision.

Each handler has one dispatch site, so every "X follows add" pair in the whole
program shares a single branch site and the predictor must separate targets by
history alone. Replicating hot handlers K ways and distributing copies by code
position shrinks each copy's target set. Ertl & Gregg described this; it is
rarely implemented because it requires controlling handler generation and
layout — which build-time generation provides for free.

Deferred on the author's assessment that trace-based predictors on modern
out-of-order cores already capture most of this. Cheap to test later; it
changes nothing structural.

## Appendix E — foldsim: method and evidence record

### E.1 What the tool is

`tools/foldsim` (workspace crate `sf-nano-foldsim`) reuses `sf-nano-core`'s
`Module` parser and streaming `op_decoder` to simulate this exact predecoder —
symbolic stack, `Local`/`Const`/`Temp` descriptors, dst-patching, hazard
flush, boundary materialization — over real wasm, and reports dispatch
prediction, fold rates, operand class distribution, window depth, and span
histograms.

```
cargo run --release -p sf-nano-foldsim -- <wasm files>
```

Coverage: 2,613 functions across 10 modules — coremark, lua (778 functions),
sqlite/speedtest1 (1,423 functions), c-ray, mandelbrot, sha256, stream, lz4,
bzip2, fib. **0 bailed, 0 desynced**, under hard stack-height invariants
checked at every block end and function end.

Weighting is the project's inherited 10^loop-depth static heuristic.

### E.2 Two methodology commitments this tool encodes

Both from the author, both a response to a specific past failure where
benchmark artifacts drove a design off course:

1. **Real-program static analysis before microbenchmarks.** Static profiles
   measure actual code shapes, so there is no simulation-fidelity gap to argue
   about.
2. **B1 traces must come from real predecoded wasm**, loop-replayed — never
   synthetic loops or random jump streams. Successor patterns and target
   entropy must be real. Results are used as bounds, not predictions.

### E.3 The hostile review record

The author required adversarial review before any of these numbers were
allowed to drive design. Three independent reviews ran concurrently on
distinct briefs — wasm semantics, counting/report arithmetic, and model
validity. Roughly 15 findings across four data iterations (v1→v4). The
findings that **changed a design-relevant number** are recorded here, because
anyone regenerating this data needs to know which mistakes were easy to make:

**Changed a conclusion:**

| finding | effect |
|---|---|
| The 307 M conversion denominator was wrong — raw counts included structural instructions, the historical figure did not | Introduced a separate old-basis ratio. Corrected prediction ~150 M. The earlier 155 M estimate had been right *by coincidence*, from wrong arithmetic |
| Depth histogram sampled *after* popping operands, understating window need by one slot | **Window sizing flipped**: from "1 slot covers 99.8%" to 1 slot 94%, 2 slots 99.8%, and c-ray's float expression trees needing 3 (2 slots only 88.4%). This is the origin of §8's gp 2 / fp 3 |
| Loop-carried local read spans measured too long — static order makes back edges invisible | Added a back-edge write proxy. Short-span local reads 25% → **30.6%**. Moved the l0 knob from "close it" back to "B1 decides" |
| `induction` counted `(local, const)` add/sub without checking the local matched the store target | 12.1% → **7.1%** |
| Stores were counted into the binop class matrix (15–25% contamination) | Split into a separate store family; the §11 class distribution is post-split |
| Loop `END` raw weight used pre-pop 10^d while the same end's emitted movs used post-pop 10^(d−1) | A half-applied fix from an earlier round, biasing the headline ratio in the *flattering* direction. Found only by the third review |

**Corrected in the model's favor:**

| finding | effect |
|---|---|
| Dead code was resurrected at the first inner `end` after a multi-level `br` | Replaced with real merge-reachability tracking. Low measured impact on LLVM -O2 output; would bite `br_table` switch code and non-LLVM producers |
| The region-based dst-fold predicate was unsound when the target local was read between a buried producer and the set | Became §4.2 rule 3's read half |
| A region bump on calls blocked dst-folding across void calls, though folding across a call is sound | Removed — a callee cannot observe caller locals. This is §4.2's closing note |
| Weighted percentages printed beside unweighted counts; short-read denominator included dropped gets; concentration aggregated by summing per-function top-k | Reporting honesty; no design impact |

**Verified clean** by the reviews: operand arity ranges against `opcodes.rs`;
control-frame bookkeeping including multi-value and if-with-params; the merge
plumbing (no dropped counter); `em_idx`/`def_em` accounting; the fold gate
after fixes (no over-folding — actively attacked); mov breakdown summing to
100%; report denominators.

Two lessons worth carrying: the most dangerous finding was a **half-applied
fix** — one side of a correction landed and the other did not, producing a
self-consistent-looking but biased number. And it biased in the flattering
direction, which is exactly the direction least likely to be questioned.

### E.4 v4 results in full

The verified dataset. §11 summarizes; this is the complete table.

| metric | value | decides |
|---|---|---|
| old-basis dispatch ratio | aggregate **0.445**, CoreMark **0.489** | ~150 M predicted vs 189 M full-fusion — **21% fewer**, with no pattern library |
| `local.get` folded | 95.8% | folding works |
| `local.set` dst-folded | 75.1% | the dst-patching payoff (B.8) |
| `local.tee` folded | 98.9% | |
| `const` folded | 97.7% | |
| binop operand classes | LC 47.2%, LL 29.3%, LT 9.2%, TC 7.6%, TL 3.3%, **TT 3.1%** | live variant set is small — the ×12 estimate holds. TT at 3% is why the TOS window is nearly vestigial: folding turns expression leaves into operand descriptors |
| store family (separate) | address operand 71% L; 13.5% of semantic dispatches | store handler family sizing |
| temp def→use span 1 | 94.9% (>4 ≈ 0.1%) | C.5 — acc/window coverage |
| materialized depth at semantic ops | integer ≤2 covers 98.5–100%; **c-ray needs 3** (≤2 only 88.4%) | §8 window widths |
| movs | 9.0% of dispatches | |
| — call-arg staging | ~half of movs ≈ **4.4%** of all dispatches (**7.7%** on lua) | §5 — the `stage_args`/`flush_args` family's ceiling |
| — boundary movs | 0–10% of movs | §4.3's conservatism costs ~0.1% of dispatches — parked |
| — hazard flush | <1% of movs | §4.4 — LLVM rarely holds a local across its overwrite |
| `cmp_br` fusable | **10.3%** of dispatches | §9 |
| induction update | **7.1%** of dispatches | §9 `dec_and_branch` |
| short-span (≤2) local reads | **30.6%** of consumed local reads (CoreMark 37.8%, sha256 44.2%); per-function top-1 ≈ 30%, top-3 ≈ 70% | the l0 knob stays open for B1 |

**Stated limits of this data:**

- The 10^loop-depth weighting is blind to call frequency and recursion.
  fib/lua dynamic projections are therefore **weak evidence**. Loop-dominated
  workloads are robust — CoreMark's weighted and unweighted ratios agree.
- Static-weighted ≈ dynamic frequency is a heuristic, not a measurement.
- Boundary and call-argument materialization both use deliberately
  conservative models. Measured boundary movs at 0–10% of movs suggest the
  conservatism does not distort the aggregate, but it is conservatism, not
  accuracy.
- Final arbitration belongs to a real prototype. This data sizes the design;
  it does not validate its performance.

### E.5 The extrapolation, labelled as such

Scaling the historical peak by dispatch count: 6,251 × 189/150 ≈ **7,800**
CoreMark. This is **(model)** — an extrapolation, not a prediction. It assumes
per-dispatch work density is comparable between the two designs, which is
precisely the assumption B1 exists to test, and it inherits every caveat in
Appendix A about single-core historical data. It is recorded because it sets
the order of magnitude the design is aiming at, not because it is expected to
be accurate.
