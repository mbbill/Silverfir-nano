# Early 32-Bit Legalization Design

This document captures the new design direction for 32-bit native backends.

The key change is to legalize true GP `i64` values before planning, instead of
repairing them later after lowering has already collapsed SSA values into a
finite machine-register bank.

The purpose of this document is not only to describe the new stage placement,
but to define the rules that every future implementation must follow.

## Goal

For 32-bit targets, everything below legalization should behave like a genuine
32-bit pipeline:

- no scalar GP `i64` values
- transient GP lane count equals number of 32-bit GP values
- fixed GP registers are word-sized
- lowering, MachineIR, and backends do not need to recover split 64-bit state
  after the fact

FP values are not part of this legalization. `f32` and `f64` remain as-is.

## Pipeline

64-bit targets keep the existing shape:

`wasm -> semantic IR -> planning -> LIR -> lowering -> MachineIR -> binary`

32-bit targets instead operate on a legalized semantic form before planning:

`wasm -> legalized semantic IR -> planning -> LIR -> lowering -> MachineIR -> binary`

This legalization must happen before planning so the planner sees the real
32-bit GP shape and does not need mixed `i32`/`i64` transient accounting.

Implementation detail:

- the 32-bit path may decode directly into legalized semantic IR, or
- decode into a raw semantic form and rewrite it immediately

Both are acceptable, but downstream stages should only see the legalized
semantic form on 32-bit targets.

## Stage Ownership

Each stage should own only one kind of responsibility:

- semantic IR / legalized semantic IR:
  - explicit typed value flow
  - structured control
  - call / branch / block signatures
- early 32-bit legalization:
  - eliminate scalar GP `i64`
  - rewrite value flow into word-sized GP form
  - preserve semantic correctness and canonical slot ABI
  - produce the semantic form that planning will consume on 32-bit
- planning:
  - budget and transient-lane reasoning over already-legalized values
  - local-cache preference selection
- LIR:
  - explicit SSA values plus slot publication / reload
- lowering:
  - map legal SSA values into machine operations
  - never rediscover how to split a GP `i64`
- MachineIR / backend:
  - consume already-legal 32-bit GP programs
  - reject leftover raw GP `i64`

This separation is intentional. Planning should not be responsible for `i64`
repair, and lowering should not be responsible for reconstructing split state.

## Core Contract

After 32-bit legalization:

- there is no scalar GP `i64`
- every Wasm `i64` GP value is represented as two word values: `(lo, hi)`
- integer transient lanes count word values, not original Wasm values
- 32-bit backends must reject any raw GP `i64` that survives below this point

This is the central invariant for the rest of the pipeline.

## Required Invariants

The following invariants should hold if the implementation is correct.

### Before Early Legalization

- semantic IR may contain ordinary Wasm `i64`
- params / results / branch payloads / block signatures must carry enough type
  information to identify which values are true GP `i64`

### After Early Legalization On 32-Bit Targets

- no scalar GP `i64` remains
- every legalized GP value has one of these shapes:
  - one word
  - two-word pair representing a Wasm `i64`
  - unchanged FP value
- every block param and edge payload arity is already expanded to word count
- every transient-lane count is already word count
- planning cannot be surprised later by a value splitting into more GP lanes

### At LIR / Lowering / MachineIR Boundaries

- slot-based boundaries still refer to canonical 8-byte Wasm slots
- value-based boundaries already use legalized word arity
- a 32-bit backend must not receive a raw scalar GP `i64`
- `emu32` must consume the same legal 32-bit GP shape as a real 32-bit backend

## Design Principles

### Legalize Values, Not Frame Layout

Do not solve the GP `i64` problem by changing canonical frame layout.

The correct rule is:

- Wasm storage ABI remains canonical and 8-byte slotted
- transient/register ABI becomes word-sized for 32-bit targets

This keeps runtime layout stable while fixing the real architectural issue.

### Split As Early As Needed, But No Earlier

Legalization must happen before planning, because planning needs correct
word-sized transient counts.

It is acceptable for the 32-bit semantic builder to produce legalized semantic
IR directly. What matters is that planning does not see unresolved scalar GP
`i64`.

### Preserve Target Choice For Hard Ops

Do not eagerly turn every hard `i64` operation into a helper call in the
legalizer.

The legalizer should expose a target-legal shape, not force a single codegen
strategy. Some 32-bit backends can inline operations that others may lower
through helpers.

### Fail Fast Below Legalization

A 32-bit backend receiving illegal raw GP `i64` is not a recoverable situation.

Do not add backend-local fallback splitting or hidden repair logic. That would
recreate the same class of late-stage complexity this design is trying to
eliminate.

## Frame And Boundary Contract

Early legalization changes GP SSA values, not the canonical frame ABI.

The frame contract stays the same:

- one canonical Wasm value still occupies one 8-byte frame slot
- slot-based boundaries such as calls, returns, and runtime helper regions stay
  canonical and 8-byte based

What changes is how GP `i64` values are published to and reloaded from those
slots:

- a legal `i64` SSA value is carried as `(lo, hi)`
- publishing or reloading that value uses paired word traffic against the same
  canonical 8-byte slot

This keeps the frame layout stable while making the transient/register world
truly 32-bit.

## What A Proper Implementation Must Do

### Strengthen Semantic Signatures

Implementation must make the semantic layer explicit enough to split `i64`
value flow before planning:

- typed function params and results
- typed block params and block results
- typed branch payloads
- typed call params and results

Simple counts are not sufficient for early legalization.

### Rewrite Control-Flow Payloads Consistently

When one value becomes two words:

- block params must double
- edge bindings must double
- branch payload arity must double
- call argument/result value arity must double where value-based
- value-based `select` must operate on both halves under the same condition

This must be done consistently across the entire CFG. Any mismatch here will
create invalid SSA that later stages cannot repair cleanly.

### Rewrite Slot Traffic But Keep Canonical Slots

`LoadSlot` / `StoreSlot`-style publication and reload must be rewritten so that
an `i64` pair reads/writes the same canonical 8-byte slot as two word
transfers.

The implementation must not:

- split one canonical `i64` slot into two independent canonical slots
- change call/result frame layout
- redefine local-slot numbering based on legalization

### Keep Pair Identity Explicit

Once a Wasm `i64` is split into `(lo, hi)`, later stages must still know those
two word values belong to the same logical value.

The concrete rule is:

- the `lo` and `hi` halves of one legalized `i64` always appear adjacently
- `lo` always comes first

This should hold anywhere the legalized form exposes ordered value lists:

- result lists
- block params
- edge bindings
- value-based call payloads, if any remain

That does not require backend-facing pair instructions, but it does require the
legalized representation to preserve pairing in a mechanically checkable way
from position alone, without side tables.

Examples:

- paired results of one legalized op
- paired block params introduced from one original `i64`
- paired slot load/store traffic

### Keep Planning Simple

After legalization, planning should only need to count word-sized GP values.

Planning must not:

- rediscover which two words form a Wasm `i64`
- reason about hidden later value expansion
- reserve speculative extra transient lanes for future backend repair

### Make Lowering Strictly One-Way

Lowering should only translate an already-legalized value graph into machine
operations.

Lowering must not:

- reintroduce scalar GP `i64`
- guess value pairing from slot width
- reinterpret pointer-sized GP values as 64-bit integers just because they live
  in 8-byte slots

## Semantic IR Requirements

This design does not require semantic IR to stay Wasm-shaped.

For 32-bit targets, the semantic layer consumed by planning must already be in
legalized form. That legalized semantic form may be produced directly during
semantic construction or by an immediate rewrite before planning. To make that
possible, the semantic layer must carry enough type information to split value
flow correctly:

- function params/results
- block params/results
- branch payload arity and types
- call params/results

Simple counts are not enough once one Wasm `i64` may become two legalized word
values.

## Legalized Value Model

On 32-bit targets:

- `i32`, refs, pointers, booleans, indices: one GP word value
- `i64`: two GP word values `(lo, hi)`
- `f32`, `f64`: unchanged

This implies:

- block params carrying `i64` are doubled
- edge bindings carrying `i64` are doubled
- stack effects for legalized GP `i64` operations are expressed in word values
- planning sees the true transient lane demand directly

## Operation Strategy

Not every original Wasm `i64` op needs to survive as a later-stage special
instruction.

### Expand Immediately To Word Ops

These can be rewritten mechanically during early legalization:

- `i64.const`
- `i64.and`, `i64.or`, `i64.xor`
- `i64.eq`, `i64.ne`
- `i32.wrap_i64`
- `i64.extend_i32_s`, `i64.extend_i32_u`
- `i64.extend8_s`, `i64.extend16_s`, `i64.extend32_s`
- `i64.eqz`
- `i64.clz`, `i64.ctz`, `i64.popcnt` with legalized result `(res32, 0)`
- legalized `select` on `i64` as two word selects sharing one condition
- full-width linear-memory `i64.load` / `i64.store`
- sign/zero-extending `i64.load8/16/32_*`
- truncating `i64.store8/16/32`

For these operations, prefer full rewrite during legalization rather than
carrying special downstream forms.

### Requires A Small New Semantic Primitive Set

The legalizer should introduce only a very small number of new 32-bit semantic
primitives where plain word ops are not enough:

- add with carry
- sub with borrow
- wide multiply

This should be treated as the core new primitive surface. In particular:

- equality and inequality do not need a special pair-compare op
- the main new requirement is explicit carry / borrow production and use

### Keep Abstract When Backend Choice Matters

These should remain as explicit legalized operations until lowering or backend
selection, because some 32-bit targets can inline them while others may prefer
helpers or more target-specific sequences:

- signed/unsigned ordered pair compare (`lt/le/gt/ge`)
- variable shifts and rotates
- `i64 <-> float` conversions
- `f64 <-> (lo, hi)` bit reinterpret

These are not late MachineIR pair ops. They are target-aware legalized ops in
the semantic/LIR world.

The important rule is: keep only the minimum abstract surface needed to defer a
backend choice. Do not recreate a large pair-instruction family downstream.

### Helper-Backed By Default, But Not Hardwired Too Early

These are good candidates for backend-selected helper lowering:

- `i64.div_s`, `i64.div_u`
- `i64.rem_s`, `i64.rem_u`
- trapping and saturating float-to-`i64` conversions

The legalizer should preserve enough structure that a backend may still inline
them later if it wants to.

## What A Proper Implementation Must Not Do

- Do not reintroduce post-lowering GP `i64` splitting.
- Do not build a second late legalizer under MachineIR "just in case".
- Do not couple canonical 8-byte frame slots to 64-bit GP register semantics.
- Do not encode backend-only ABI quirks into shared legalization.
- Do not hide illegal 32-bit GP state inside backend-local repair paths.
- Do not make `emu32` more permissive than real 32-bit backends.
- Do not let helper-call decisions erase opportunities for targets that can
  lower an op natively.
- Do not let value pairing become implicit or reconstructable only by dataflow
  guesswork.

## Why This Is Better Than Late MachineIR Legalization

Late legalization had to recover information that was already lost:

- SSA value identity had collapsed into reused machine registers
- 64-bit GP meaning had to be rediscovered per program point
- high-half tracking had to be reconstructed after register reuse
- legalization could inflate the machine-register graph after planning

Early legalization avoids that entire class of problems:

- split happens while values are still explicit
- planning sees the real 32-bit GP demand
- no post-hoc GP bank compaction is needed just to get back under budget
- MachineIR and backends no longer need a large family of 32-bit bridge ops

This is the main success criterion for the design: complexity should move
upward into an explicit typed rewrite, not downward into backend repair.

## Backend Expectations

For 32-bit backends, the contract below legalization should be strict:

- no raw scalar GP `i64`
- no backend-side guesswork about how to split values
- no hidden repair logic for mixed-width GP state

If a 32-bit backend receives illegal raw GP `i64` state, it should fail fast.

Backend code should be allowed to assume:

- all GP values are already word-sized unless explicitly represented as a
  legalized pair/abstract 32-bit op
- frame slots remain canonical 8-byte Wasm slots
- helper-time slot ABI is stable across targets

## Emulator Rule

`emu32` must expose shared 32-bit pipeline problems.

That means `emu32` should execute the same legalized 32-bit GP shape that a
real 32-bit backend consumes. If a bug is in shared 32-bit legalization,
planning, or lowering, `emu32` should expose it before a real backend does.

`emu32` is part of the architecture, not just a debugging convenience.

If a shared 32-bit bug can reach a real backend but cannot reach `emu32`, that
is a design failure in the pipeline contract.

## Implementation Outline

1. Strengthen semantic IR type/signature metadata so `i64` flow can be split
   before planning.
2. Make the 32-bit path produce legalized semantic IR before planning, either
   directly during semantic construction or by an immediate rewrite.
3. Make planning consume only that legalized semantic form on 32-bit targets.
4. Keep canonical frame slots and slot-based boundaries unchanged.
5. Remove backend-facing raw GP `i64` assumptions from lowering and MachineIR.
6. Make 32-bit backends reject leftover raw GP `i64`.

## Validation Rules

During implementation, each stage should validate the invariants it owns.

Recommended checks:

- semantic-legalization validator:
  - no unresolved scalar GP `i64`
  - all split block params / edges are consistent
  - all rewritten arities match legalized signatures
  - all legalized `i64` halves appear adjacently with `lo` first
- planning validator:
  - transient GP lane counts are already word counts
- lowering validator:
  - 32-bit lowering never constructs raw GP `i64`
- backend / `emu32` validator:
  - reject any illegal leftover scalar GP `i64`

The goal is to make illegal mixed-width state impossible to carry quietly into
later stages.

## Non-Goals

This design does not try to:

- change canonical 8-byte frame slots
- split FP values as part of the GP `i64` legalization work
- force every hard legalized op into a helper call immediately

The goal is early GP `i64` legalization, not a general rewrite of the runtime
or frame ABI.
