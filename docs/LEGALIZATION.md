# 32-Bit GP Legalization Design

This document captures the design for 32-bit native backends.

The key principle: **pair-aware MachineIR is the primary 32-bit native IR,
emitted directly from lowering.** Semantic IR, planning, LIR, locals, params,
returns, call slots, and frame layout stay Wasm-shaped and scalar. The lowerer
is the single place that knows "this i64 LirValue is actually two machine
registers."

## Goal

For 32-bit targets, the native lowering path should produce MachineIR that
uses only word-sized GP values, with a small vocabulary of pair instructions
for operations that span two words.

Everything above the lowerer — semantic IR, planning, LIR — stays
Wasm-shaped. Everything below — MachineIR, emulator, backends — sees
explicitly paired 32-bit GP values.

FP values (`f32`, `f64`) are not affected. They remain as-is at all stages.

## Pipeline

64-bit targets:

```
semantic IR -> planning -> LIR -> lowering -> MachineIR -> backend
                                    |
                              (i64 = 1 reg)
```

32-bit targets:

```
semantic IR -> planning -> LIR -> lowering -> MachineIR -> backend
                                    |
                              (i64 = 2 regs,
                               pair MachineIR)
```

The difference is entirely in the lowerer's code path. No separate legalization
pass. No post-hoc repair.

## What Stays Wasm-Shaped

- **Semantic IR**: unchanged. `i64.add` is `i64.add`. No pair ops, no split
  values, no rewriting.
- **Planning**: budgets i64 as 2 GP units on 32-bit targets. Spill/fill works
  on scalar i64 LirValues. The planner never splits anything.
- **LIR**: scalar. One `LirValue` per Wasm value, including i64. `LoadSlot` /
  `StoreSlot` refer to canonical 8-byte frame slots.
- **Locals**: `local_count` and `params` stay Wasm-original. One i64 local =
  one 8-byte frame slot. No slot inflation, no index remapping.
- **Call boundaries**: both external and internal calls are slot-based through
  canonical `FrameSpan` arg/result regions. The lowerer handles pair
  marshalling at the slot/register boundary — packing `(lo, hi)` into one
  8-byte arg slot before a call and unpacking from one 8-byte result slot
  after.
- **Frame layout**: canonical. One Wasm value = one 8-byte slot. No
  `compiler_locals_start`, no scratch locals, no inflated slot counts.

## What Changes At The Lowerer

The lowerer's 32-bit code path:

1. Maps each i64 `LirValue` to two machine registers `(lo, hi)`.
2. Expands `LoadSlot` for i64 into two half-slot loads (offset +0 and +4
   within the same canonical 8-byte slot).
3. Expands `StoreSlot` for i64 into two half-slot stores.
4. Emits pair MachineIR instructions for i64 arithmetic, comparisons, shifts,
   conversions, etc.
5. Emits paired `Select` for i64 select (two word-selects sharing one
   condition).
6. Never produces `GpI64`-tagged scalar instructions — the 32-bit MachineIR
   vocabulary uses `GpWord` for all GP traffic plus explicit pair ops.

### The (lo, hi) Register Mapping

The lowerer maintains a per-block mapping from each i64 `LirValue` to its
companion high-half machine register:

```rust
fn use_i64_value(&mut self, val: LirValue) -> Result<(MachineReg, MachineReg), WasmError> {
    let lo = self.use_value(val)?;       // existing: allocates/finds the GP reg
    let hi = self.alloc_hi_for(val)?;    // new: allocates the companion reg
    Ok((lo, hi))
}
```

This is a local per-block table. The planner already ensured the transient
budget fits (2 GP lanes per i64), so the companion allocation always succeeds.
No whole-program dataflow analysis is needed.

### Half-Slot Access

An i64 local occupies one canonical 8-byte frame slot. The lowerer accesses it
as two 32-bit halves:

```
// LoadSlot for i64 on 32-bit:
Load { ty: GpWord, dst: lo, addr: FP + slot*8 + 0, width: U32 }
Load { ty: GpWord, dst: hi, addr: FP + slot*8 + 4, width: U32 }

// StoreSlot for i64 on 32-bit:
Store { ty: GpWord, addr: FP + slot*8 + 0, width: U32, src: lo }
Store { ty: GpWord, addr: FP + slot*8 + 4, width: U32, src: hi }
```

This introduces paired word accesses to a single canonical 8-byte slot.
Downstream passes must recognize this pattern:

- **Validators**: two word-sized accesses to the same slot at offsets +0 and
  +4 are a legal i64 pair, not an aliasing conflict.
- **Peephole optimizers**: must not reorder or eliminate one half of a paired
  store/load without the other. The two accesses are semantically atomic at
  the i64 value level.
- **Frame-effect analysis**: both half-slot accesses alias the same canonical
  slot. A store to `slot+0` and a store to `slot+4` together constitute one
  complete i64 store; neither alone is sufficient.

The `frame_addr` helper needs a sub-slot offset parameter for the 32-bit path.

## Pair MachineIR Vocabulary

The pair instructions already exist in `MachineInstKind`. They are the primary
32-bit native IR for i64 operations:

### Binary Arithmetic

```rust
Int64PairBinary {
    op: MachineIntBinaryOp,  // Add, Sub, Mul
    dst_lo, dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi,
}
```

Covers add, sub, mul. Bitwise ops (and, or, xor) don't need a pair
instruction — the lowerer emits two independent `IntBinary { width: I32 }`
instructions.

### Division / Remainder

```rust
Int64PairDivRem {
    sign, rem,
    dst_lo, dst_hi, lhs_lo, lhs_hi, rhs_lo, rhs_hi,
}
```

### Shifts / Rotates

```rust
Int64PairShift {
    op: MachineIntBinaryOp,  // Shl, ShrS, ShrU, Rotl, Rotr
    dst_lo, dst_hi, lhs_lo, lhs_hi, rhs,  // rhs is word-sized count
}
```

### Comparisons

```rust
Int64PairCompare {
    kind, sign, dst,
    lhs_lo, lhs_hi, rhs_lo, rhs_hi,
}
```

Equality/inequality don't need the pair compare — the lowerer can emit
`(lo_eq & hi_eq)` or `(lo_ne | hi_ne)` using word-sized compares.

### Unary

```rust
Int64PairUnary {
    op: MachineIntUnaryOp,  // Clz, Ctz, Popcnt, Extend8S, etc.
    dst_lo, dst_hi, src_lo, src_hi,
}
```

### Conversions

```rust
ConvertI64PairToFloat { width, sign, dst, src_lo, src_hi }
ConvertFloatToI64Pair { op, dst_lo, dst_hi, src }
ReinterpretF64ToI64Pair { dst_lo, dst_hi, src }
ReinterpretI64PairToF64 { dst, src_lo, src_hi }
```

## Backend Scratch Registers (Implementation Guidance)

Some pair ops need internal temporaries during backend code emission. This
section is implementation guidance for backend authors, not part of the formal
design contract. Each backend owns its scratch register policy.

Current ARMv7 scratch budget as reference:

| Pair op | Typical scratch needed |
|---|---|
| PairBinary (Add/Sub) | 1 (carry/borrow) |
| PairBinary (Mul) | 1 (cross product) |
| PairCompare | 1 (intermediate boolean) |
| PairShift | 2 (count_mod32 + temp) |
| PairDivRem | 0 (helper-backed) |
| PairUnary (Clz/Ctz) | 1 |
| Bitwise (And/Or/Xor) | 0 (two independent i32 ops) |
| Conversions | 0 (helper-backed) |
| Load/Store (half-slot) | 0 |

ARMv7 has R12 (SCRATCH0) and R14 (SCRATCH1) reserved outside the register
allocator. The emulator doesn't need scratch — it interprets semantically.
Other backends define their own scratch pools.

The design-level rule is only: scratch registers are backend-private and must
not appear in the LIR value graph or the planner's transient budget.

## Planning: Budget i64 As 2 GP Units

The planner counts each i64 value as consuming 2 GP transient lanes on 32-bit
targets. This is already implemented via `gp_value_budget_units()` in the
local-cache analysis and the transient-budget spill logic.

The planner does not split values. It just counts correctly:

- An i64 value live in the transient window costs 2 GP lanes
- Spilling an i64 value frees 2 GP lanes
- The frame slot for an i64 is still one canonical 8-byte slot

The lowerer then maps the planned i64 LirValue to two machine registers,
confident that the budget was pre-approved by the planner.

### Cached i64 Locals

A cached local is pinned to a machine register for the entire function,
bypassing spill/fill. On 32-bit targets, a cached i64 local consumes 2 GP
cache registers (one for lo, one for hi). The cache analysis must account for
this — `gp_value_budget_units()` returns 2 for i64 when `gp_unit_bytes == 4`.

The design rule: a cached i64 local costs 2 GP cache slots. The cache
analyzer's knapsack selection already handles this via budget-aware selection
(`select_with_budget`). If the GP cache budget is too small for any i64 local
(e.g., only 1 slot available), the i64 local is not cached and falls back to
slot traffic.

This is a hard requirement, not optional. If the cache analysis selects an i64
local for caching but only reserves 1 GP register, the lowerer will fail to
allocate the companion hi register. The planner must guarantee that every
cached i64 local has 2 GP cache registers reserved.

## What The Old Late Legalizer Did (And Why It Was Complex)

The previous design ran legalization as a post-lowering repair pass on
already-allocated MachineIR. That required:

1. **Storage-flow analysis** (~400 lines): a fixed-point dataflow pass to
   rediscover which registers hold i64 vs i32 vs float values, because type
   information was lost after lowering.

2. **Hi-half register tracking** (~200 lines): `persistent_hi` / `current_hi`
   tables to allocate and track companion registers for each i64 register,
   with per-block binding management across the CFG.

3. **GP bank compaction** (~400 lines): graph-coloring-like pass to pack the
   inflated register set (original + hi-halves + temps) back into the
   backend's physical register budget.

These three systems — ~1000 lines of infrastructure — exist solely because
the legalizer had to recover information that was already available during
lowering. The actual rewrite logic (matching i64 ops and emitting pair
instructions) was the same work either way.

The new design eliminates all three systems:

- The lowerer knows the LIR value types → no storage-flow analysis
- Pair instructions carry their operands explicitly → no hi-half tracking
- The planner budgets 2 GP lanes per i64 → no compaction

## Stage Ownership

- **Semantic IR**: Wasm-shaped. No pair ops, no split values.
- **Planning**: budgets i64 as 2 GP units. Does not split or rewrite.
- **LIR**: scalar. One LirValue per Wasm value.
- **Lowering**: the single place that maps i64 LirValues to (lo, hi) machine
  register pairs and emits pair MachineIR.
- **MachineIR**: the first stage where 32-bit legalization is explicit.
  Contains pair instructions, half-slot loads/stores, word-sized GP traffic.
- **Backend**: consumes pair MachineIR. Uses backend-owned scratch registers
  for pair-op expansion. Rejects any `GpI64`-tagged scalar instruction.

## Core Contract

On 32-bit targets, the MachineIR produced by the lowerer must satisfy:

- No `GpI64`-typed scalar instructions (Move, Load, Store, IntBinary, etc.)
- Every i64 value flow uses explicit pair instructions or paired half-slot
  traffic
- Block parameters for i64 values are doubled (two GP-word params per i64)
- Edge bindings for i64 values are doubled
- The GP register count stays within the planner's budget (no inflation)
- Frame layout is canonical (no slot inflation, no local remapping)

## Backend Expectations

32-bit backends should:

- Implement codegen for all pair MachineIR instructions
- Use backend-owned scratch registers (not LIR-visible) for internal temps
- Reject any `GpI64`-typed scalar instruction that leaks through
- Assume frame slots are canonical 8-byte Wasm slots

## Emulator Rule

`emu32` consumes the same pair-aware MachineIR as real 32-bit backends. If a
bug is in the shared lowering path, `emu32` should expose it before a real
backend does.

## What This Design Must Not Do

- Do not run a separate legalization pass after lowering.
- Do not introduce storage-flow analysis to rediscover register types.
- Do not allocate companion hi-half registers post-hoc.
- Do not compact the GP bank after inflation.
- Do not change semantic IR, LIR, or frame layout for 32-bit targets.
- Do not introduce compiler scratch locals.
- Do not inflate `local_count`, `params`, or slot counts.
- Do not change call boundary `FrameSpan` layout.
- Do not make `emu32` more permissive than real 32-bit backends.

## Implementation Outline

1. Add `gp_unit_bytes` awareness to the lowerer context so it knows when to
   emit the 32-bit code path.
2. Implement the `(lo, hi)` register mapping in the lowerer: `use_i64_value`
   / `alloc_hi_for` as a per-block companion-register table.
3. Implement half-slot load/store expansion for i64 `LoadSlot` / `StoreSlot`.
4. Route each i64 `LirLeafOp` through the pair MachineIR emission path on
   32-bit targets.
5. Handle i64 block parameters: double them in the MachineIR block params and
   edge bindings.
6. Handle i64 select: emit two word-selects sharing one condition.
7. Handle i64 call boundaries: marshal between canonical 8-byte slot layout
   and paired GP-word register form.
8. Ensure all pair instructions are implemented in the emulator and ARMv7
   backend.
9. Add 32-bit backend validation: reject any `GpI64`-tagged scalar
   instruction.
10. Remove the old post-lowering legalizer (`legalize.rs`), storage-flow
    analysis, and GP bank compaction.
