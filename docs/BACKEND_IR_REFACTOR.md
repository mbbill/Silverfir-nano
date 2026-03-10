# Backend IR Refactor

This document records the intended architecture for the next native/backend
refactor.

This is an initial design, not a strict file-by-file requirement. The purpose
of this document is to keep the important boundaries and intentions stable
while the code is being moved and split. Exact filenames and folder shapes may
change if a cleaner structure emerges during the refactor.

## Current Refactor State

This document is the source of truth for the refactor. The old implementation
was moved under `sf-nano-core/src/vm_bak/` for reference only.

Important:

- `vm_bak/` must never be wired back into the live build or runtime
- borrowing code from `vm_bak/` is allowed only if the borrowed code still
  respects the design in this document
- do not preserve old patterns just to make things compile
- do not run `cargo fmt`

Current implementation priority:

1. restore and stabilize the top-level `vm/*.rs` surface
2. refactor planning/grouping/LIR into CFG + SSA block-param form
3. make the base interpreter consume that new LIR and pass spectest
4. bring up native second on top of the proven LIR
5. bring up fusion last

Reason:

- the new LIR is the main semantic boundary and must be proven correct first
- base is the easiest consumer to use as that proof
- once base is green on the new LIR, it becomes the ground truth for native
  debugging
- fusion changes significantly under this design because grouping moves before
  LIR and helper/cold-path structure also changes, so it should be rebuilt last

## Scope Discipline

The refactor should focus on the compilation pipeline and backend folders:

- `vm/wasm/`
- `vm/plan/`
- `vm/lir/`
- `vm/interp/`
- `vm/native/`
- `vm/abi/`
- `vm/debug/`

Top-level `vm/*.rs` files should be treated as stable unless there is a strong
architectural reason to change them:

- `backend.rs`
- `entities.rs`
- `expr_eval.rs`
- `instance.rs`
- `runtime.rs`
- `store.rs`
- `value.rs`

Changing those files for naming or style reasons is noise and should be
avoided.

The main goal is to stop leaking stack-machine concepts into the backend.
The current code still carries interpreter-era assumptions such as
`pre_height`, generic `variant`, helper entry families like `read_t0`, and
continuation logic that reasons about top-of-stack state after lowering.
That direction is wrong.

This note is meant to be read before touching the backend. The most important
design intentions are listed first.

## Hard Invariants

### 1. After backend-facing IR, the backend is not a stack machine anymore

The backend works with only:

- registers
- `fp[...]` memory slots
- immediates / targets

It must not reason about:

- logical stack height
- spill depth
- how many stack values are cached
- generic TOS state reconstruction

If backend code needs those things, that is a frontend/planning bug or a bad
backend boundary.

### 2. TOS becomes SSA before or at LIR construction

The stack/TOS model survives only long enough to support:

- planning
- grouping
- spill/fill planning
- backend-budget-aware lowering

By the time LIR exists, TOS must no longer be represented as:

- stack height
- rotating window
- `window` / `variant`
- implicit top/next register selection

Instead, LIR must represent TOS through:

- block parameters
- successor arguments
- SSA values

This is the key change in the current design direction.

### 3. TOS values and hot locals are different

TOS-derived values are transient SSA values inside the CFG.

Hot locals are not just more TOS values:

- hot locals are persistent VM state carried across blocks
- each hot-local slot has function-static identity
- hot locals are part of the target-tuned VM ABI
- ordinary frame locals and memory remain explicit stateful effects

So the IR must distinguish:

- transient SSA values
- named hot-local state
- frame/memory effects

This is also the reason the old "TOS + local cache is already the register
allocator" idea can still survive in the new design:

- TOS lanes are not durable machine state; they are an entry/edge interface
- hot locals are durable cached machine state
- native lowering only has to place values into those fixed VM locations
- there is still no general-purpose register allocator pass

### 4. Hot and cold paths obey the same lowered IR contract

Cold wrappers do not get a special exemption.

A cold helper wrapper must obey the same rule as a hot native group:

- inputs are explicit
- outputs are explicit
- no runtime reconstruction of stack/TOS state

If a cold op needs data in memory, frontend/planning must make that explicit
before the backend sees it.

## Root Cause Of The Current Design Problem

The current lowered IR still leaks stack-machine state through:

- `pre_height`
- `variant`
- `window`
- helper entry selection based on `read_t0`, `read_top2_d1`, `write_t1`, etc.

This makes the backend deduce register behavior from stack-machine metadata.
That is the wrong direction.

What `variant` really means today is not "stack depth". It is only a compact
way to encode which physical rotating-cache register mapping applies.

What `pre_height` is mostly used for today is also wrong:

- selecting register layout
- inferring which T registers are live
- helper wrapper specialization

Those should not be backend responsibilities.

## Target Pipeline

The intended pipeline is now:

1. Decode to semantic / stack-aware ops
2. Planning
3. Grouping
4. Convert TOS flow into CFG + SSA LIR
5. Execute that LIR in the base interpreter
6. Lower LIR into one target-independent native IR
7. Translate native IR mechanically into ISA code

### 1. Semantic / stack-aware layer

This layer is allowed to think in stack-machine terms.

It handles:

- Wasm semantics
- control flow
- stack effect
- locals
- structured blocks

### 2. Planning

Planning is still allowed to be stack-aware.

It handles:

- hot local policy
- spill/fill insertion
- frame layout
- any frontend-only artifacts needed to make the backend simple

### 3. Grouping

Grouping should happen before backend-facing IR.

Grouping gets its policy from the backend:

- `fusion`: conservative, pattern-constrained grouping
- `native`: maximal grouping policy
- `base`: can ignore grouping results later

Grouping is the only stage where the rotating TOS cache semantics should still
be a first-class concept.

### 4. CFG + SSA LIR

This is where the stack-machine abstraction must end.

LIR should be a CFG with:

- blocks
- block parameters
- SSA values
- explicit successor arguments
- explicit hot-local state operations
- explicit frame/memory effects
- explicit control-flow terminators

It must not require any backend to deduce behavior from:

- stack height
- rotation/window
- implicit top-of-stack location

TOS values should arrive at a block through explicit lane parameters, for
example conceptually:

- `t0`
- `t1`
- `t2`
- `t3`

but as block parameters and successor arguments, not as hidden stack state.

Hot locals should also be explicit in LIR, but not necessarily as block
parameters or successor arguments. If hot-local slot identity is static across
the function, LIR may model hot locals as named VM state with explicit
read/write operations plus explicit entry initialization.

### 5. Base interpreter over LIR

The base interpreter should consume this new LIR directly.

This is intentional:

- it proves the new LIR semantics before native work begins
- it separates LIR bugs from native lowering bugs
- spectest on base becomes the correctness gate for the new boundary

### 6. Native lowering

Native should not lower directly from stack-shaped concepts.

It should lower from CFG + SSA LIR into one target-independent pseudo-register
IR.

### 7. ISA translation

ISA-specific lowering should be simple:

- no optimization
- no stack reasoning
- no Wasm semantic reinterpretation
- just translation, legalization, and patching

## Explicit Design Decision: Only One IR After LIR

We considered a deeper native stack such as:

- `LIR -> Entry IR -> Role IR -> ISA`

and rejected it for this project.

Reason:

- this engine is intentionally small and embedded-friendly
- the whole point of the TOS/local-cache model is to avoid a traditional
  register allocator and a large optimizer stack
- adding multiple native-only semantic layers starts to recreate a general JIT
  compiler pipeline
- if users want a large optimizing JIT, they can use something like Cranelift

The chosen direction is:

- `wasm -> planning/grouping + SSA -> LIR`
- `LIR -> virtual-register assignment + simple optimization -> native IR`
- `native IR -> ISA`

So there is exactly one IR after LIR.

The meaning of that choice is:

- all semantic restructuring happens before or at LIR
- all "real" optimization happens in `LIR -> native IR`
- ISA lowering is intentionally dumb and mechanical

## Why Grouping Must Stay Before LIR

If T registers are treated as fully generic unconstrained registers too early,
grouping becomes much harder or incorrect.

Grouping depends on the fact that:

- T-register updates follow stack-machine discipline
- internal T-register churn inside a group can be omitted or fused
- spill/fill guarantees the rotating cache remains valid

So the stack/TOS model should survive long enough to drive grouping.
After grouping, it should collapse into:

- CFG structure
- block parameters
- SSA values
- explicit state effects

## Backend Roles

### Base backend

- consumes CFG + SSA LIR directly
- is the first semantic validator of the new LIR

### Fusion backend

- should later consume the same CFG/LIR plus grouping-derived policy
- should be rebuilt only after base and native are stable again

### Native backend

- consumes CFG + SSA LIR
- lowers it into one target-independent pseudo-register IR
- emits native code from that pseudo-register IR

The important points are:

- grouping remains shared frontend work
- LIR is the main semantic handoff
- native IR is not another semantic IR, only a machine-shaped lowering step

## Red Flags

These are signs that the design is drifting back toward the old bad boundary.

- backend code using `pre_height` to infer register validity
- backend code reasoning about spill depth
- runtime metadata that says how many top values are live
- helper entry families tied to stack-machine notions like:
  - `read_t0`
  - `read_top2_d1`
  - `write_t1`
- wrappers that reconstruct TOS state at runtime
- continuations/resume entries that exist only to adapt generic TOS metadata

If any new code needs those things, stop and revisit the boundary.

## Immediate Refactor Direction

The next large refactor should aim for:

1. Move grouping before backend-facing IR
2. Convert TOS flow into block parameters and SSA values in LIR
3. Make base interpreter consume that LIR and pass spectest
4. Lower that LIR into one target-independent pseudo-register IR
5. Keep ISA lowering mechanical only

## LIR As CFG + SSA

This is the current agreed direction and supersedes the earlier idea of
keeping a rotation/window-based LIR plus a second semantic native IR layer.

### Core shape

LIR should be a CFG made of blocks.

Each block should have:

- block parameters for incoming TOS lanes
- explicit hot-local state operations
- SSA values for transient results
- explicit stateful frame/memory operations
- an explicit terminator

Successor edges should carry explicit arguments.

This means:

- no `window`
- no `variant`
- no implicit top-of-stack convention inside LIR
- no backend-side stack reconstruction

LIR is the semantic handoff. It is not "almost machine code". It still carries:

- SSA values
- control-flow structure
- explicit successor arguments
- explicit stateful effects

but it must no longer carry stack-machine deduction hints.

### TOS as SSA

The important insight is:

- TOS values are not really machine registers
- inside a grouped/native region they behave like SSA values
- the 4 TOS lanes are primarily a boundary convention between blocks

So the TOS model should become:

- block input parameters
- successor arguments
- SSA values inside the block body

Conceptually:

- a `pop1 push1` op becomes one input SSA value and one output SSA value
- a `pop2 push1` op becomes two input SSA values and one output SSA value
- the terminator chooses which SSA values occupy which outgoing TOS lanes

Important:

- slot identity belongs to the edge/block-parameter contract
- slot identity does not belong to the SSA value itself

The same SSA value may later be copied into a different outgoing TOS lane.

That means LIR should think in terms of:

- block parameters for incoming TOS lanes
- SSA values within the block
- successor arguments for outgoing TOS lanes

not "value `vN` permanently lives in `t0`".

### Block contract

Every block should expose the incoming state it needs.

Conceptually a block may look like:

```text
block B(
  t0 = v0,
  t1 = v1,
)
```

where:

- `t*` are transient incoming TOS-lane values
- hot locals are named VM state with function-static slot identity

The block body computes new SSA values, reads/writes hot-local state if needed,
and ends in a terminator that chooses successor arguments for TOS lanes.

Conceptually:

```text
block B(t0 = v0)
  h0 = read_hot_local L0
  v1 = i32.add h0, v0
  write_hot_local L0, v1
  jump C(t0 = v1)
```

This is how the TOS window disappears from LIR.

### Hot locals are different

Hot locals are not just more TOS SSA values.

Hot locals are:

- persistent cached VM state
- part of the target-tuned VM ABI
- explicit named state carried across blocks

So LIR must keep the distinction between:

- transient SSA values
- hot-local state
- frame/memory effects

Hot locals should be thought of as named persistent VM state values that are
available throughout the function. They are not ordinary frame loads/stores,
and they are not just another transient TOS SSA value.

Because hot-local slot identity is static across the function, they do not need
to be threaded through block parameters / successor arguments the way TOS lanes
are. The important requirements are:

- explicit `read_hot_local` / `write_hot_local` style operations in LIR
- function-level metadata describing the hot-local mapping
- explicit entry initialization for those hot locals

Ordinary locals that are not hot remain ordinary frame-home state addressed via
explicit effects.

### Frame state is still state

One subtle but important rule:

- TOS/SSA values can be forwarded like pure transient values
- frame-local and memory writes are still real effects and must not disappear
  just because a later read can be forwarded

So this is valid:

```text
v0 = load_frame slot7
store_frame slot8, v0
v1 = load_frame slot8
v2 = add v0, v1
```

may simplify to:

```text
v0 = load_frame slot7
store_frame slot8, v0
v2 = add v0, v0
```

but not to:

```text
v0 = load_frame slot7
v2 = add v0, v0
```

The write is still part of the entry/block state.

### Why this is the right semantic boundary

Once TOS has become SSA in LIR:

- intra-block optimization becomes straightforward
- compare/branch fusion becomes straightforward
- branch targets use explicit edge arguments instead of resume metadata
- base can execute the same LIR the native backend lowers from

That is the main architectural goal of the current refactor.

## Native IR

After LIR, there should be exactly one more IR before ISA lowering.

This IR is not another semantic IR. It is a machine-shaped pseudo-register IR.

The intended pipeline is:

1. `wasm -> planning/grouping + SSA -> LIR`
2. `LIR -> virtual register assignment + simple optimization -> native IR`
3. `native IR -> simple machine translation`

### Purpose

Native IR exists to:

- minimize ISA porting cost
- isolate all optimization before ISA lowering
- make the final emitter simple and mechanical

The ISA backend should not optimize. It should only:

- translate pseudo registers to physical registers
- legalize for ISA constraints
- encode instructions
- patch addresses/literals

This IR should look like platform-independent pseudo assembly, not another
semantic SSA IR.

### Register model

Native IR should use platform-independent VM register names.

Conceptually:

- `T0..Tn` for TOS lane registers
- `L0..Lm` for hot-local registers
- `Tmp0..Tmpk` for scratch temporaries
- `Ctx`
- `Fp`

This is still not general register allocation.
The TOS/local-cache design remains the allocator.

The main job of `LIR -> native IR` is just:

- respect block input/output contracts
- place incoming/outgoing TOS values into `T*`
- keep hot locals in `L*`
- use a small number of `Tmp*` values for emission convenience
- insert edge shuffles only when a successor needs a different lane layout

That is a deterministic placement problem, not a full RA problem.

### Platform tuning

The backend should provide a register-budget configuration.

At minimum this includes:

- number of TOS registers
- number of hot-local registers

and conceptually also:

- available scratch temporaries
- reserved special registers like `ctx` and `fp`

The frontend should only need this backend configuration.

Then different targets can tune:

- how many TOS lanes to keep live
- how many hot locals to cache
- how much temporary capacity remains

without changing the frontend architecture.

This means the VM ABI becomes backend-parameterized rather than hardcoded.

For example, one target may choose:

- 4 TOS lanes
- 3 hot locals

while another target may choose:

- 3 TOS lanes
- 2 hot locals

and a larger-register target may choose:

- 6 TOS lanes
- 5 hot locals

The frontend should only consume these counts and plan accordingly.

### Optimization boundary

Optimization should happen in `LIR -> native IR`, not in `native IR -> ISA`.

Examples of the intended pre-ISA optimizations:

- constant folding
- copy propagation
- dead SSA value elimination
- compare + branch fusion
- redundant frame reload elimination
- edge shuffle elimination
- trivial direct-chaining cleanup

This is intentionally a small/simple optimization set, not a full JIT optimizer.

Non-goals here:

- no global register allocation
- no large scheduling framework
- no attempt to become a general optimizing compiler
- no ISA-specific optimization logic hidden inside backend folders

### Cold helpers

The helper ABI is intentionally deferred to the final platform-emission stage.

Reason:

- helper ABI is highly platform-dependent
- it should not distort the LIR or native IR design
- later, CFG/block-argument structure should allow cold helpers to be grouped or
  integrated with hot paths if desired

For now, the important rule is:

- LIR and native IR must model helper inputs/outputs explicitly
- exact helper call ABI details belong to the final emitter/runtime boundary

This also leaves room for future work where some currently-cold helpers may be
fused or grouped with hot blocks. The CFG/block-argument design should not
prevent that.

## Old Native Code: What Must Move Above ISA

The old `vm_bak/native/` implementation mixed together three different jobs:

- target-neutral optimization
- cross-entry state adaptation
- ISA-specific encoding

Examples from the old code:

- alias/state tracking such as `slots`, `local_fp_aliases`, and
  `frame_aliases`
- target-neutral peepholes such as compare/branch tail fusion
- resume/adaptation logic such as packed resume-slot handling

Those concerns must not stay inside `arm64/` or any future ISA folder.

The new split should be:

- LIR owns semantic/control-flow structure
- `LIR -> native IR` owns value placement, simple optimization, and edge
  adaptation
- `native IR -> ISA` owns only platform lowering/encoding

## Validation Sequence

The new sequencing is deliberate:

1. refactor planning/grouping/LIR into CFG + SSA form
2. make the base interpreter run that LIR correctly
3. use spectest on base as the proof that LIR semantics are correct
4. only then lower LIR into native IR and build platform emitters

This gives a clean debugging split:

- if base fails, the bug is in planning/grouping/LIR/interpreter semantics
- if base passes and native fails, the bug is in LIR-to-native lowering,
  native optimization, helper ABI, or ISA emission

Once native bring-up starts, dump/trace comparison should use the base
interpreter as the oracle. That is the intended debugging flow for narrowing
future native bugs.

## Proposed `vm/` Structure

This is the intended folder/file structure under `vm/`. Again, this is an
initial design, not a rigid requirement. The important part is the ownership
boundary, not the exact path spelling.

```text
vm/
  mod.rs
  backend.rs
  runtime.rs
  instance.rs
  entities.rs
  store.rs
  value.rs
  expr_eval.rs

  wasm/
  plan/
  lir/
  interp/
  native/
  debug/
  abi/
```

### Top-Level Files

- `mod.rs`
  VM module wiring only.
- `backend.rs`
  Backend selection and capability reporting.
- `runtime.rs`
  High-level backend dispatch.
- `instance.rs`
  Instance creation and compilation entry.
- `entities.rs`
  Runtime-owned compiled artifacts and function/module state.
- `store.rs`
  Store/runtime state.
- `value.rs`
  Runtime value representation.
- `expr_eval.rs`
  Constant-expression evaluation.

### `wasm/`

Semantic frontend only. This layer is still allowed to think in structured
Wasm / stack-machine terms.

```text
vm/wasm/
  mod.rs
  common.rs
  core_op.rs
  decode.rs
  semantic_ir.rs
  control.rs
```

- `common.rs`
  Shared semantic ids and small types.
- `core_op.rs`
  Canonical semantic op set.
- `decode.rs`
  Decode Wasm into semantic IR.
- `semantic_ir.rs`
  Semantic IR containers and op records.
- `control.rs`
  Block/control-stack analysis helpers.

This layer should not own:

- hot-local policy
- spill/fill
- frame layout
- grouping policy
- backend register mapping

### `plan/`

Planning is the last place where stack-machine and rotating-cache concepts are
allowed to be first-class.

```text
vm/plan/
  mod.rs
  config.rs
  hot_local.rs
  frame.rs
  spill.rs
  tos.rs
  group.rs
  policy.rs
  plan.rs
```

- `config.rs`
  Backend-provided planning configuration.
- `hot_local.rs`
  Hot-local policy.
- `frame.rs`
  Frame-slot / operand-slot layout planning.
- `spill.rs`
  Spill/fill planning logic.
- `tos.rs`
  Rotating T-cache planning logic.
- `group.rs`
  Group formation over planned ops.
- `policy.rs`
  Backend grouping policy interface.
- `plan.rs`
  Planned stream produced for later lowering.

This is the only stage where:

- rotating T-cache semantics
- spill/fill placement
- hot-local planning
- grouping legality

should still be direct concerns.

### `lir/`

Backend-facing semantic CFG boundary. This is where stack-machine deduction
must stop.

```text
vm/lir/
  mod.rs
  ir.rs
  lower.rs
  slot.rs
  target.rs
  dump.rs
```

- `ir.rs`
  CFG, block, SSA value, and terminator records.
- `lower.rs`
  Lower planned ops/groups into CFG + SSA LIR.
- `slot.rs`
  `fp[...]` slot references.
- `target.rs`
  Block/edge target representation.
- `dump.rs`
  Shared LIR dump.

Important:

- LIR should not carry `window`, `variant`, or implicit stack-top state
- LIR should represent TOS through block params, successor args, and SSA values
- hot locals should remain explicit state, not be collapsed into ordinary TOS
  SSA values
- frame/memory/global effects should remain explicit operations

This makes LIR the semantic proof boundary shared by base and native.

### `interp/`

Handler-based interpreter family only.

```text
vm/interp/
  mod.rs
  raw_value.rs
  stack.rs

  fast/
    mod.rs
    instruction.rs
    resolved.rs
    encoding.rs
    frame_layout.rs
    precompile.rs
    runtime.rs
    resolve.rs
    finalizer.rs
    dump.rs

    fusion/
    handlers/
    handlers_c/
    trampoline/
```

- `raw_value.rs`
  Interpreter raw-value helpers.
- `stack.rs`
  Plain interpreter stack support if needed.

`fast/` remains handler/trampoline-oriented and is allowed to keep that model.
It should not define the native backend model anymore.

### `native/`

Standalone native backend facade and shared runtime/helper pieces.

```text
vm/native/
  mod.rs
  build.rs
  ir.rs
  lower.rs
  code.rs
  code_buf.rs
  runtime.rs
  precompile.rs
  context.rs

  helper.rs
  helper_meta.rs
  bridge.rs

  dump.rs
  map.rs
  jitdump.rs

  arch/
```

- `build.rs`
  Shared decode -> plan -> group -> LIR native build entry.
- `ir.rs`
  Target-independent pseudo-register native IR.
- `lower.rs`
  Lower CFG + SSA LIR into native IR with simple optimization and virtual
  register placement.
- `code.rs`
  Native compiled code object.
- `code_buf.rs`
  Executable-memory ownership and write/exec transitions.
- `runtime.rs`
  Native runtime entry/launch.
- `precompile.rs`
  Native precompile entry.
- `context.rs`
  Native runtime context ABI.
- `helper.rs`
  Rust helper implementations that remain cold/complex.
- `helper_meta.rs`
  Typed helper metadata records.
- `bridge.rs`
  Native wrapper generation and helper-call ABI glue.
- `dump.rs`
  Native dump appendix.
- `map.rs`
  Address map support.
- `jitdump.rs`
  Profiler symbol emission.
- `arch/`
  ISA-specific lowering and encoding only.

This folder must not drift back toward:

- interpreter-style `Instruction`
- `NativeInst`
- generic `imm0/imm1/imm2`
- generic `tos_slots`
- helper entry families like `read_t0`

The goal is to minimize porting cost. Adding a new ISA later should mainly mean
adding a new `arch/<isa>/` lowering/encoding implementation, not re-implementing
semantic optimization or value-state reasoning.

### `native/arch/`

ISA-specific native backend only.

```text
vm/native/arch/
  mod.rs

  arm64/
    mod.rs
    reg.rs
    enc.rs
    emit.rs
    entry.rs
    lower.rs

  x64/
    mod.rs
    reg.rs
    enc.rs
    emit.rs
    entry.rs
    lower.rs
```

- `reg.rs`
  Physical register assignment for that ISA.
- `enc.rs`
  Raw instruction encoders.
- `emit.rs`
  Small ISA emission helpers and helper-ABI glue.
- `entry.rs`
  ISA-specific entry/term glue.
- `lower.rs`
  Lower native IR to encoded ISA bytes.

This layer should not know about:

- Wasm decode
- planning policy
- LIR semantics
- high-level optimization policy
- generic dump layout

Anything that interprets CFG/LIR semantics or performs optimization should live
above `arch/`, not inside one ISA folder.

### `debug/`

Shared optional debug infrastructure.

```text
vm/debug/
  mod.rs
  dump_layout.rs
  function_trace.rs
  trace_compare.rs
```

- `dump_layout.rs`
  Shared dump directory layout helpers.
- `function_trace.rs`
  Sparse trace recording.
- `trace_compare.rs`
  Trace comparison helpers.

### `abi/`

Shared low-level ABI/data-layout utilities.

```text
vm/abi/
  mod.rs
  operand_encoding.rs
  compaction.rs
```

- `operand_encoding.rs`
  Shared immediate/operand encoding helpers.
- `compaction.rs`
  Shared compaction rules if still shared.

## Review Question

When reviewing a backend change, ask this first:

- is this code operating on:
  - CFG structure
  - block parameters / successor arguments
  - SSA values
  - explicit hot-local state
  - `fp[...]` effects
  - pseudo registers / machine instructions

or is it secretly trying to reconstruct a stack machine again?

If it is reconstructing stack state after planning/grouping, the boundary is
wrong.

## Practical Review Question

When reviewing any backend change, ask:

> Is this code just selecting registers and addressing `fp[...]`, or is it
> secretly reasoning about a stack machine again?

If the answer is the latter, the design has already drifted.
