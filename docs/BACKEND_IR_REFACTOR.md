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
2. bring up the base fast interpreter first
3. bring up native second
4. bring up fusion last

Reason:

- base is the easiest backend to make correct first
- once base is working again, it becomes the ground truth for native debugging
- fusion changes significantly under this design because grouping moves before
  backend IR, so it should be rebuilt last

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

### 2. TOS is only a rotating register cache

The frontend uses a 4-register rotating cache for the operand stack top.
That cache is always valid because frontend/planning inserts `Spill` / `Fill`
as needed.

The important point:

- backend must not ask whether `T0..T3` are valid
- backend must assume the rotating cache is valid
- backend only needs to know the current rotation

Example:

- top starts at `T1`
- `drop`
- top becomes `T0`
- `drop`
- top becomes `T3`

`T3` must already contain the correct value. The backend does not get to
reason about whether it is live.

### 3. TOS semantics only matter in grouping/planning

The only place where the stack-machine / TOS model should matter is:

- grouping
- spill/fill planning
- hot-local planning
- stack-aware lowering decisions

Once backend-facing IR is produced, the backend should only see explicit
register and memory behavior.

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

The intended pipeline is:

1. Decode to semantic / stack-aware ops
2. Planning
3. Grouping
4. Lower to backend-facing IR
5. Backend codegen

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

### 4. Backend-facing IR

This is where the stack-machine abstraction must end.

Backend-facing IR should describe:

- which register classes are read
- which register classes are written
- which `fp[...]` slots are read
- which `fp[...]` slots are written
- immediates / targets

It must not require the backend to deduce any of that from `pre_height`.

Important clarification:

- backend-facing IR should not carry explicit `T0` / `T1` / `T2` / `T3`
  operands
- it is enough to carry the current rotating-window top / rotation offset
- each instruction already has stack effect, so the backend can derive which
  concrete T register is used from:
  - the current rotation
  - the opcode stack effect

So the backend-facing IR still carries the rotating-cache convention, but it
does not carry explicit stack-machine liveness or explicit per-op T-register
lists.

### 5. Backend codegen

Backends consume:

- groups
- explicit register/memory IR

They do not redo grouping or stack reasoning.

## What The Backend Still Needs To Know

The backend still needs one stack-related concept:

- the current rotation of the 4-register T cache

This is not the same thing as stack height or number of cached values.

The backend may need a compact field describing:

- which T register is currently the logical top

That field should mean only:

- how to map top / next / next-next to concrete registers

It should not mean:

- how many values are live
- how many values are cached
- whether a given T register is valid

Possible names:

- `rotation`
- `tos_head`
- `top_rotation`

Avoid vague names like `variant` unless the meaning is explicitly limited to
rotating-cache register selection.

## Why Grouping Must Stay Before Backend IR

If T registers are treated as fully generic unconstrained registers too early,
grouping becomes much harder or incorrect.

Grouping depends on the fact that:

- T-register updates follow stack-machine discipline
- internal T-register churn inside a group can be omitted or fused
- spill/fill guarantees the rotating cache remains valid

So the stack/TOS model should survive long enough to drive grouping.
After grouping, it should collapse into explicit register/memory behavior.

## Backend Roles

### Base backend

- ignores grouping information
- emits handler stream instruction-by-instruction

### Fusion backend

- consumes grouping result
- emits handler stream mapped to groups / fusion patterns

### Native backend

- consumes grouping result and group contents
- emits native code for groups and wrappers

The important point is that grouping should be shared policy/planning work, not
reimplemented inside each backend.

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
2. Replace `pre_height`-driven backend logic with backend-facing IR driven by:
   - rotation
   - stack effect
   - register classes
   - `fp[...]`

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

Backend-facing IR boundary. This is where stack-machine deduction must stop.

```text
vm/lir/
  mod.rs
  ir.rs
  lower.rs
  reg.rs
  slot.rs
  target.rs
  dump.rs
```

- `ir.rs`
  Backend-facing IR records.
- `lower.rs`
  Lower planned ops/groups into LIR.
- `reg.rs`
  Register classes and rotation helpers.
- `slot.rs`
  `fp[...]` slot references.
- `target.rs`
  Branch/call target representation.
- `dump.rs`
  Shared LIR dump.

Important:

- LIR should not carry explicit `T0..T3` operands
- it should carry only the rotating-window top / offset
- backend derives the concrete T register from:
  - rotation
  - stack effect

That keeps the backend simple without leaking stack height, spill depth, or
cached-value count.

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
  code.rs
  code_buf.rs
  runtime.rs
  precompile.rs

  lower.rs
  resolve.rs
  finalizer.rs

  helper.rs
  helper_meta.rs
  bridge.rs
  context.rs

  dump.rs
  map.rs
  jitdump.rs

  arm64/
```

- `code.rs`
  Native compiled code object.
- `code_buf.rs`
  Executable memory ownership.
- `runtime.rs`
  Native runtime entry/launch.
- `precompile.rs`
  Native precompile entry.
- `lower.rs`
  LIR to native-lowered representation if needed.
- `resolve.rs`
  Native resolved op/group representation.
- `finalizer.rs`
  Native target patching and final assembly orchestration.
- `helper.rs`
  Rust helper implementations that remain cold/complex.
- `helper_meta.rs`
  Typed helper metadata records.
- `bridge.rs`
  Native wrapper generation and helper-call ABI glue.
- `context.rs`
  Native runtime context ABI.
- `dump.rs`
  Native dump appendix.
- `map.rs`
  Address map support.
- `jitdump.rs`
  Profiler symbol emission.

This folder must not drift back toward:

- interpreter-style `Instruction`
- `NativeInst`
- generic `imm0/imm1/imm2`
- generic `tos_slots`
- helper entry families like `read_t0`

### `native/arm64/`

ISA-specific native backend only.

```text
vm/native/arm64/
  mod.rs
  reg.rs
  enc.rs
  emit.rs
  op_meta.rs
  codegen.rs
  group.rs
  semantics.rs
```

- `reg.rs`
  ARM64 register assignment for native ABI.
- `enc.rs`
  Raw ARM64 encoders.
- `emit.rs`
  Small emission helpers/templates.
- `op_meta.rs`
  Native codegen metadata per op.
- `codegen.rs`
  Emit native code from native-lowered ops.
- `group.rs`
  Group code shape assembly for ARM64.
- `semantics.rs`
  Structured emission helpers.

This layer should not know about Wasm decode, planning policy, or generic dump
layout.

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

- is this code using only:
  - rotation
  - stack effect
  - registers
  - `fp[...]`
  - immediates / targets

or is it trying to reconstruct stack state after backend-facing IR?

If it is reconstructing stack state, the boundary is wrong.
3. Replace generic `variant` with an explicit rotating-cache descriptor
4. Remove native helper entry specialization based on TOS reads/writes
5. Make cold helpers use explicit memory/register contracts only

## Practical Review Question

When reviewing any backend change, ask:

> Is this code just selecting registers and addressing `fp[...]`, or is it
> secretly reasoning about a stack machine again?

If the answer is the latter, the design has already drifted.
