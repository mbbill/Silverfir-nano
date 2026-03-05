# Neutral IR Layer for Unified Interpreter/Fusion/JIT Pipeline

## Context

Stack state management is duplicated three times across the interpreter builder (`dispatch.rs` + `stack.rs`), static fusion (`gen_fusion_*.rs` + `StackSim`), and micro-JIT (`codegen.rs` + `group.rs`). Each independently tracks TOS height, computes D1-D4 variants, classifies ops for stack effects, and manages spill/fill. Adding a new opcode requires updating 4-5 places. The current `TempInst` is too tightly coupled to the interpreter's handler-based instruction format — it carries handler function pointers and encoding-focused `PatternData`, making it unsuitable as a neutral representation for fusion or JIT.

The goal is a clean IR that resolves all stack management once, then serves as input to three unified backends: 1:1 interpreter handlers, static fusion, and dynamic JIT — all sharing the same pipeline with graceful degradation.

## Design Overview

```
Wasm bytecode
    │
    ▼
[Wasm→IR Lowering] ← StackTracker (single source of stack truth)
    │  Resolves: TOS variants, spill/fill, hot local mapping
    │  Output: Vec<IrOp> — neutral, backend-agnostic
    │
    ▼
[Unified Backend Pass]
    │  For each group of consecutive IR ops:
    │    1. If micro-jit enabled → try JIT compilation → ARM64 code
    │    2. If fusion enabled → check fusion table → pre-compiled C handler
    │    3. Fallback → 1:1 handler mapping
    │  Output: Vec<TempInst> (handler-resolved, ready for finalizer)
    │
    ▼
[Finalizer] (needs minor adaptation — see §4)
    │  Compact, patch branches, encode immediates
    │  Output: Box<[Instruction]>
```

The IR doesn't know or care about fusion, JIT, or the interpreter. Whether the unified backend enables JIT, fusion, both, or neither is a backend decision — the IR is the common input regardless.

## 1. The IR Type (`ir.rs`)

### `IrOp` — one per resolved instruction (including spill/fill)

```rust
pub struct IrOp {
    pub kind: IrOpKind,       // What this op does (semantic)
    pub variant: u8,          // D1-D4 (1-4), resolved by lowering. 0 = N/A
    pub pre_height: u16,      // Stack height before this op
    pub fallthrough: Option<usize>,  // Next IR index (linear)
    pub alt_target: Option<usize>,   // Branch/else target (IR index)
    pub has_target: bool,     // Needs pointer patching in finalizer
}
```

The `variant` field's meaning is "which D-variant handler to select." Different op categories compute it differently during lowering:

| Op category | Variant computation |
|---|---|
| Push ops (const, local_get, global_get) | Post-push depth: `((height + 1 - 1) % 4) + 1` |
| Pop ops (binop, unop, local_set, drop) | Pre-pop depth: `((height - 1) % 4) + 1` |
| Spill | `(spill_depth % 4) + 1` |
| Fill | `((spill_depth - 1) % 4) + 1` |

This is computed once during lowering. Downstream backends just read the value.

### `IrOpKind` — purely semantic, no handler pointers, no encoding concerns

```rust
pub enum IrOpKind {
    // Arithmetic (no immediates — operands are on TOS)
    I32Add, I32Sub, I32Mul, I32And, I32Or, I32Xor,
    I32Shl, I32ShrS, I32ShrU, I32Rotl, I32Rotr,
    I64Add, I64Sub, I64Mul, /* ... all i64 binops ... */
    F32Add, F32Sub, F32Mul, F32Div, /* ... all float ops ... */

    // Comparisons
    I32Eq, I32Ne, I32LtS, I32LtU, I32GtS, I32GtU, I32LeS, I32LeU, I32GeS, I32GeU,
    I64Eq, I64Ne, /* ... */

    // Unary
    I32Eqz, I32Clz, I32Ctz, I64Eqz, I64Clz, I64Ctz, /* ... */

    // Conversions
    I32WrapI64, I64ExtendI32S, I64ExtendI32U, /* ... all conversions ... */

    // Constants
    I32Const { value: u32 },
    I64Const { value: u64 },

    // Locals — hot vs frame already resolved during lowering
    LocalGetHot { reg: u8 },    // l0=0, l1=1, l2=2
    LocalSetHot { reg: u8 },
    LocalTeeHot { reg: u8 },
    LocalGetFrame { idx: u16 }, // remapped frame index
    LocalSetFrame { idx: u16 },
    LocalTeeFrame { idx: u16 },

    // TOS management — explicit, inserted during lowering
    Spill { slot: u16, count: u8 },
    Fill  { slot: u16, count: u8 },

    // Memory
    I32Load { offset: u32, memidx: u32 },
    I32Store { offset: u32, memidx: u32 },
    /* ... all load/store variants ... */

    // Globals
    GlobalGet { idx: u32 },
    GlobalSet { idx: u32 },

    // Control flow
    If, Else,
    Br { stack_drop: u32, arity: u16, height: u16, operand_base_offset: u32 },
    BrIfSimple,
    BrIf { stack_drop: u32, arity: u16, height: u16, operand_base_offset: u32 },
    BrTable { entries: Vec<BrTableEntry>, height: u16, operand_base_offset: u32 },

    // Calls
    CallExternal { func_idx: u32, delta: u16 },
    CallInternal { callee: u64, delta: u16 },
    CallIndirect { type_idx: u32, table_idx: u32, delta: u16, operand_base_offset: u32, height: u16 },

    // Returns
    ReturnVoid { frame_size: u16 },
    ReturnOne { frame_size: u16, operand_base_offset: u32, height: u16 },
    Return { arity: u16, frame_size: u16, operand_base_offset: u32, height: u16 },
    Unreachable,

    // Misc
    Drop, Select, RefNull, RefIsNull, RefFunc { func_idx: u32 },
    MemorySize { mem_idx: u32 }, MemoryGrow { mem_idx: u32 },
    TableGet { table_idx: u32 }, TableSet { table_idx: u32 },
    /* ... remaining misc ops ... */

    // Prologue
    InitLocals { k0: u16, k1: u16, k2: u16 },

    // Structural (removed during finalization)
    Nop, Block, Loop, End,

    // Terminal / pseudo
    Term,
    Data { imm0: u64, imm1: u64, imm2: u64 },
}
```

### Shared stack effect table — single source of truth

```rust
/// Canonical (pops, pushes) for every IrOpKind.
/// Used by: lowering (validation), JIT grouper, fusion matcher.
pub fn stack_effect(kind: &IrOpKind) -> (u8, u8) { /* one big match */ }
```

This replaces: `op_stack_effect()` in `group.rs`, implicit knowledge in `dispatch.rs`, `get_pop_push()` in `op_classify.rs`, and `compute_tos_pattern()` in `fusion_discovery.rs`.

### Key design principles

- **No handler pointers** — the IR is backend-agnostic
- **Variant is explicit** — computed once during lowering, never recomputed
- **Hot locals resolved** — `LocalGetHot { reg: 0 }` vs `LocalGetFrame { idx: 5 }` — downstream never checks hot_local mapping
- **Spill/fill are first-class fusible ops** — simple register↔memory transfers, simpler than memory loads (no bounds check). Fully fusible, not group boundaries.
- **No PatternData** — encoding is the backend's concern, not the IR's

## 2. Wasm→IR Lowering (`ir_lower.rs`)

New file: `sf-nano-core/src/vm/interp/fast/builder/ir_lower.rs`

This replaces `dispatch.rs` as the primary Wasm decode driver. It:
1. Decodes Wasm opcodes (reuses existing `Decoder`/`OpStream`)
2. Tracks stack via `StackTracker` (reuses existing `stack.rs`)
3. Inserts `Spill`/`Fill` IR ops when TOS overflows
4. Computes `variant` for each op based on current height (per-category formula)
5. Resolves hot local indices (`remap_local` + `has_l0/l1/l2`)
6. Handles control flow (block/loop/if/else/end with forward branch fixups)
7. Outputs `Vec<IrOp>`

### Control flow subtleties the lowering must preserve

**LOOP label placement**: LOOP emits spill_all → LOOP marker → fill. The loop target is placed BETWEEN spill and fill, so backward branches arrive at a normalized TOS state:
```
Spill_all   ← IR op
Loop        ← structural (branch target here)
Fill        ← IR op
... loop body ...
```

**END conditional spill/fill**: Whether to spill before END depends on whether any forward branch has `stack_offset > 0`. Whether to fill depends on whether forward branches exist or it's an IF block. The branch target index is captured between the conditional spill and the fill.

**Implicit RETURN**: `on_decode_end()` synthesizes `Spill_all + Return` at the end of the function body if not unreachable.

**OPERAND_BASE placeholder**: Call delta uses `OPERAND_BASE = 16384` as a placeholder resolved by the finalizer's `fixup_slot()`. The IR preserves this convention — the finalizer transforms `OPERAND_BASE + x` to `operand_base + x`.

**BR_TABLE entries**: Stored as a Vec on the IR op. The finalizer expands them into inline data pseudo-instructions, preserving the `pc[1]`, `pc[2]` layout the runtime handler expects.

**Unreachable code**: After BR, RETURN, UNREACHABLE, BR_TABLE, the stack sets `unreachable = true`. All subsequent stack/spill operations are no-ops until END resets it.

## 3. Unified Backend Pass (`ir_backend.rs`)

New file: `sf-nano-core/src/vm/interp/fast/builder/ir_backend.rs`

This is where the three strategies (interpreter, fusion, JIT) are unified into one pass.

```rust
pub fn ir_to_temps(
    ir: &[IrOp],
    #[cfg(feature = "micro-jit")] jit_buf: &mut Option<CodeBuffer>,
) -> Vec<TempInst> {
    let mut temps = Vec::with_capacity(ir.len());
    let mut i = 0;
    while i < ir.len() {
        // Try fusion: match concrete IR sequence against table
        #[cfg(feature = "micro-jit")]
        if let Some(ref mut buf) = jit_buf {
            if let Some((len, handler)) = try_jit_group(&ir[i..], buf) {
                emit_jit_group(&mut temps, &ir[i..i+len], handler);
                i += len;
                continue;
            }
        }

        #[cfg(feature = "fusion")]
        if let Some((len, handler, data)) = FUSION_TABLE.try_match(&ir[i..]) {
            temps.push(TempInst { handler, data, ... });
            i += len;
            continue;
        }

        // Fallback — 1:1 handler mapping
        temps.push(ir_op_to_temp(&ir[i]));
        i += 1;
    }
    temps
}
```

### 3a. 1:1 Handler Mapping

```rust
fn ir_op_to_temp(op: &IrOp) -> TempInst {
    let v = op.variant.saturating_sub(1) as usize;
    let (handler, data) = match &op.kind {
        IrOpKind::I32Add => (handler_lookup::I32_ADD[v], PatternData::Raw { .. }),
        IrOpKind::I32Const { value } => (handler_lookup::I32_CONST[v], PatternData::Const { value: *value as u64 }),
        IrOpKind::LocalGetHot { reg } => {
            let lookup = [&handler_lookup::LOCAL_GET_L0, &handler_lookup::LOCAL_GET_L1, &handler_lookup::LOCAL_GET_L2];
            (lookup[*reg as usize][v], PatternData::LocalGet { idx: *reg as u16 })
        },
        IrOpKind::Spill { slot, count } => {
            let handlers = [&handler_lookup::SPILL_1, &handler_lookup::SPILL_2, &handler_lookup::SPILL_3, &handler_lookup::SPILL_4];
            (handlers[*count as usize - 1][v], PatternData::Spill1 { slot: *slot })
        },
        // ... all other ops ...
    };
    TempInst { handler, data, fallthrough_idx: op.fallthrough, alt_idx: op.alt_target, has_target: op.has_target, .. }
}
```

### 3b. Static Fusion on IR

In the IR world, there is no "variant" concept. Each IR instruction is a fully concrete, self-contained operation. `i32_add_d1`, `i32_add_d2`, `i32_add_d3`, `i32_add_d4` are **four separate instructions**, each with its own handler that operates on specific registers. They are not "variants of i32_add" — they are different instructions that happen to perform the same arithmetic.

**Profiling**: Record a frequency map of concrete IR instruction sequences. Since the IR has variants, spill/fill, and hot locals already resolved, the profiler sees the actual instruction stream the interpreter would execute.

**Discovery**: Find hot patterns from the frequency map. These patterns are concrete IR sequences — for example `[i32_const_d2, i32_add_d2]` is one pattern, `[i32_const_d3, i32_add_d3]` is a different pattern. Patterns can include spill/fill ops (they're simple register↔memory transfers, simpler than memory loads). The discovery system doesn't need to reason about variants, spill/fill, or local mapping — it simply records what sequences occur frequently.

**Handler generation**: Each discovered pattern gets a pre-compiled C handler. The handler knows the exact registers because the pattern is concrete. It directly reads/writes the specific TOS registers using `SEM_*` macros — no `impl_*` + wrapper split, no pointer indirection.

**Wasm compile time**: Match the IR stream against the fusion table. Match → emit fused handler. No match → emit individual handlers (`i32_add_d1`, `local_get_l0_d2`, etc.).

**What this eliminates**:
- `OpFuser` (Wasm-level pattern matching with lookahead) — replaced by simple IR sequence scan
- `gen_fusion_emit.rs` spill/fill helpers — spill/fill are already in the IR from lowering
- Variant computation during fusion emission — variants are part of the IR instruction identity
- The conceptual gap between "pattern" and "variant wrapper" — each pattern IS a concrete instruction sequence, each handler IS for specific registers

**TOML format**: Contains concrete IR patterns, not variant-agnostic Wasm patterns. The fusion system lives entirely behind the IR boundary and has no concept of variants.

```toml
# Current (variant-agnostic, Wasm-level):
[[fused]]
op = "const_add"
pattern = ["i32_const", "i32_add"]
tos_pattern = { pop = 1, push = 1 }

# New (concrete IR-level):
[[fused]]
op = "const_d2_add_d2"
pattern = ["i32_const_d2", "i32_add_d2"]
# No tos_pattern needed — registers are implicit in the instruction names
```

Discovery profiles at the IR level. If `[i32_const_d2, i32_add_d2]` is hot, it gets an entry and a handler. If `[i32_const_d4, i32_add_d4]` is cold, it doesn't — no wasted code for unused combinations. This is an improvement over the current system which blindly generates 4 handlers for every pattern regardless of frequency.

### 3c. JIT on IR

The JIT grouper and emitter consume `IrOp` directly. No height recomputation needed.

```rust
fn try_jit_group(ir_slice: &[IrOp], buf: &mut CodeBuffer) -> Option<(usize, Handler)> {
    // Find maximal group of JIT-able ops
    let len = ir_slice.iter().take_while(|op| is_jit_able(op)).count();
    if len < 2 { return None; }

    let mut emitter = JitEmitter::new(buf); // No initial_height!
    for op in &ir_slice[..len] {
        emit_ir_op(&mut emitter, op); // variant read from op, not tracked
    }
    let start = emitter.finish();
    Some((len, unsafe { buf.fn_ptr(start) }))
}

fn emit_ir_op(e: &mut JitEmitter, op: &IrOp) {
    let d = op.variant;
    match &op.kind {
        IrOpKind::I32Add => {
            let lhs = tos_reg(d, 2);
            let rhs = tos_reg(d, 1);
            e.buf.emit(arm64_enc::add_reg_32(lhs, lhs, rhs));
        }
        IrOpKind::I32Const { value } => {
            let dst = tos_reg(d, 1);
            materialize_u32(e.buf, dst, *value);
        }
        IrOpKind::LocalGetHot { reg } => {
            let dst = tos_reg(d, 1);
            e.buf.emit(arm64_enc::mov_reg_64(dst, LOCAL_REGS[*reg as usize]));
        }
        // ... all ops — no self.height tracking anywhere
    }
}
```

`JitEmitter` becomes stateless — no `height` field, no `dv()` method. The variant comes from the IR op. `op_stack_effect()` and `is_jit_able()` in `group.rs` (~130 lines) are replaced by shared functions in `ir.rs`.

## 4. Finalizer Adaptation

The finalizer (`finalizer.rs`) currently uses handler function pointers and `wasm_op` for structural checks. These must be updated:

- **Keep-mask** (line 183-195): Currently checks `h == op_nop as usize`. Must check `IrOpKind` instead (or the `ir_to_temps` backend sets a compatible `wasm_op` on TempInst for incremental migration).
- **Route terminals** (line 84): Checks `wasm_op` for RETURN and UNREACHABLE. Must check `IrOpKind`.
- **Expand BR_TABLE** (line 112): Checks `wasm_op` for BR_TABLE. Must check `IrOpKind`.

The simplest incremental approach: `ir_to_temps` sets `wasm_op` on TempInst to the correct Wasm opcode equivalent, preserving finalizer compatibility. In a later cleanup phase, the finalizer is updated to use `IrOpKind` directly.

Everything else in the finalizer (compaction, branch patching, encoding, two-pass target pointer resolution) is unchanged.

## 5. Build-Time Codegen Changes

| File | Change |
|------|--------|
| `gen_c_wrappers.rs` | Fused handlers: generate 1 handler per concrete IR pattern (direct register access). Base handlers: unchanged (4 D-variant wrappers). |
| `gen_fusion_c.rs` | StackSim adapts to emit code using named registers instead of pointer params. SEM_* macros still used (they're pure value macros, compatible with direct register access). |
| `gen_fusion_match.rs` | Generate IR-level pattern matching (match on concrete IR instruction names). |
| `gen_fusion_emit.rs` | Largely replaced by `ir_backend.rs`. |
| `gen_handler_lookup.rs` | Base handlers: unchanged. Fused handlers may not need lookup tables (direct resolution from concrete pattern). |
| `gen_encoding.rs` | Fused handlers: simpler encoding (no tos_pattern logic). Base handlers: unchanged. |
| `op_classify.rs` | Simplified — references shared `stack_effect()`. |
| `gen_extern_decl.rs` | Trivially extensible — same PARAMS signature for concrete-pattern handlers. |

## 6. Migration Strategy

Each phase passes all 88 spectests + 6 WASI benchmarks before proceeding.

### Phase 0: IR Types (non-breaking)
- Create `builder/ir.rs` with `IrOp`, `IrOpKind`, `stack_effect()`
- Unit tests for types
- **Risk**: Zero — no existing code changes

### Phase 1: Dual-Path Lowering
- Create `ir_lower.rs` — port `dispatch.rs` logic to produce `Vec<IrOp>`
- Create `ir_backend.rs` — convert `Vec<IrOp>` → `Vec<TempInst>` (1:1 only, no fusion/JIT yet)
- Wire both paths in `mod.rs`, compare output in debug mode
- **Risk**: Medium — extensive porting but verifiable by comparison

### Phase 2: JIT on IR
- Create `jit/ir_jit.rs` — port grouper to use `IrOp`
- Modify `JitEmitter` to accept variant from `IrOp` instead of tracking height
- Remove `group.rs::op_stack_effect()`, `is_jit_able()` — use shared `ir.rs` functions
- **Risk**: Low — straightforward rewrite with clear 1:1 mapping

### Phase 3: Fusion on IR
- Update discovery to profile at IR level, generate concrete IR patterns in TOML
- Update `gen_fusion_c.rs` + `gen_c_wrappers.rs` for concrete-pattern handler generation
- Move fusion matching to IR-level in `ir_backend.rs`
- **Risk**: Medium-high — most complex change, touches both Rust and C codegen

### Phase 4: Cleanup
- Remove old `dispatch.rs`, `emitter.rs` (or reduce to internal helpers)
- Remove JIT-specific fields from `TempInst`
- Update finalizer to use `IrOpKind` directly instead of `wasm_op` compatibility shim
- Remove duplicate classification functions

### Phase 5: Fusion Discovery on IR (future)
- Update `fusion_discovery.rs` to profile/discover patterns on IR ops
- Update `handlers_fused.toml` format for concrete IR patterns
- Re-run discovery on benchmarks

## 7. What Gets Simpler

- **Adding a new opcode**: Update `IrOpKind` enum + `stack_effect()` + lowering logic. One place, not 4-5.
- **JitEmitter**: Becomes stateless (~200 lines shorter). No height tracking.
- **group.rs**: `op_stack_effect` (~50 lines) and `is_jit_able` (~80 lines) deleted.
- **Fusion emission**: `gen_fusion_emit.rs` spill/fill helpers (~100 lines) deleted — lowering handles it.
- **Understanding the pipeline**: Clear separation: "IR has everything resolved" → backends just map.

## 8. What Stays the Same

- `Instruction` struct (32-byte layout)
- `preserve_none` ABI and handler signature (uniform PARAMS for all handlers)
- `StackTracker` (used by lowering pass)
- `SEM_*` macros (pure value macros, compatible with direct register access)
- Handler function pointers in final instruction stream
- Threaded-code dispatch at runtime (guard-check / nonlinear)
- Encoding field packing algorithm (3×64-bit budget)
- Dispatch epilogue (register-agnostic, works unchanged)
- Extern declaration generation (same signature, different names)

## 9. What Needs Adaptation

- `finalizer.rs` — keep-mask and structural checks: `wasm_op` shim initially, then `IrOpKind` directly
- `gen_c_wrappers.rs` — new code path for concrete-pattern fused handlers
- `gen_fusion_c.rs` — StackSim emits direct-register code instead of pointer params
- `gen_fusion_match.rs` — IR-level matching code
- `gen_fusion_emit.rs` — largely replaced by `ir_backend.rs`
- `gen_handler_lookup.rs` — fused handlers resolved directly, no lookup arrays
- `fusion_discovery.rs` — profile and discover at IR level
- `handlers_fused.toml` — concrete IR patterns

## 10. Verification

- `cargo run --bin sf-nano-spectest` — all 88 spec tests pass at each phase
- WASI benchmarks: `coremark`, `lua`, `brotli`, `c-ray`, `mandelbrot`, `stream` all pass
- CoreMark performance: no regression (target ≥ 8900)
- JIT stats: same number of groups compiled
- Fusion stats: same number of patterns matched
- Debug-mode dual-path comparison in Phase 1 catches any divergence
