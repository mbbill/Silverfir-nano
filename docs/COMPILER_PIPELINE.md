# Micro-JIT Compiler Pipeline

Entry point: `vm/build.rs` → `ensure_module_compiled()`

## Phase 1: Wasm Decode → Semantic IR

`wasm/decode.rs` → `decode_to_semantic_ir()`

Decodes Wasm bytecode into `SemanticProgram`: a list of `SemanticOp`s with structured control flow (Block, Loop, If/Else), abstract local access (LocalGet/Set/Tee), and semantic calls (CallInternal/External/Indirect). No frame layout or register decisions yet. Tracks `max_stack_height`, local types, and result types.

## Phase 2: Function Inlining

`wasm/inline.rs` → `inline_calls_in_function()`

Replaces `CallInternal` ops with the callee's body when the callee is a small leaf function (no calls, ≤200 ops, ≤8 params). Iterates to fixed-point so transitive chains (A→B→C) are fully resolved regardless of index ordering.

## Phase 3: Semantic IR → SSA-IR

`middle/mod.rs` → `prepare_function()`

Transforms semantic IR into a flat basic-block SSA-IR with explicit frame slots, spill/fill, and register budgets. This is a multi-step process:

### 3a. Semantic IR Validation

`semantic_ir.rs` → `SemanticProgram::validate()`

Checks structural invariants of the decoded semantic program before lowering.

### 3b. Frame Layout Planning

`middle/frame.rs` → `plan_frame_layout()`

Allocates frame slots for all locals and stack values. Produces `FrameLayoutPlan` with canonical slot assignments for params, non-param locals, stack temporaries, call-scratch region, and total frame size.

### 3c. Local Cache Preference Analysis

`middle/local_cache.rs` → `analyze_local_cache_prefs()`

Determines which locals should be pinned to dedicated registers (cached locals). Produces separate GP and FP preference lists based on access frequency. Also computes continuation-skip-reload info (which cached locals don't need reload after a call returns).

### 3d. Spill/Fill Planning

`middle/spill_plan.rs` → `prepare_semantic_ops()`

Walks the semantic stream tracking conceptual value-stack height. When height exceeds the transient register budget, inserts explicit spill/fill prefix actions. Produces `PreparedStream` with `PreparedOp`s (semantic op + prefix spill/fill actions) and per-block `entry_states` (what values are live at each block entry point).

### 3e. Control-Flow Graph Construction

`middle/lower_cfg.rs` → `build_block_ranges()` + `retain_reachable_blocks()`

Converts structured control flow (nested blocks, loops, if/else) into flat basic-block ranges with explicit edges. Eliminates unreachable blocks. Builds the `semantic_to_block` index mapping.

### 3f. Block-Level Lowering

`middle/lower_block.rs` → `lower_block_range()`

For each basic block, lowers semantic ops to SSA instructions:
- LocalGet/Set → LoadSlot/StoreSlot
- Arithmetic/conversion ops → `SsaInstKind::Value` with `SsaLeafOp`
- Memory/table/call ops → `SsaInstKind::Boundary`
- Control flow edges → SSA terminators (Goto, Branch, BrTable, Return)

Edge lowering (`lower_edge.rs`) and terminator lowering (`lower_term.rs`) handle block parameter passing and multi-way branches.

### 3g. SSA-IR Optimization

`middle/optimize.rs` → `optimize_ssa()`

Two intra-block passes:

1. **Slot-value forwarding** (`forward_slot_values`): Tracks what value was last stored to each frame slot. When a LoadSlot reads a slot whose value is still live in a register, replaces the load with an alias. Eliminates redundant load/store round-trips.

2. **Constant folding into operands** (`fold_constants_into_operands`): Two rewrites in one pass:
   - *Full evaluation*: When all args of a pure op are known constants, evaluates at compile time and replaces the op with a const definition. Results cascade through the block.
   - *Operand folding*: Replaces `SsaOperand::Value(v)` with `SsaOperand::Const(bits)` when v is a single-use constant. Backends that can't encode the immediate natively use the pre-budgeted transient register.
   - *Dead const elimination*: Removes constant definitions whose results are no longer referenced after folding.

### 3h. SSA-IR Validation

`middle/ssa_ir/validate.rs` → `validate_program()`

Checks block structure consistency, value use/def chains, and type correctness of the produced SSA-IR.

## Phase 4: SSA-IR → Machine IR

`machine/lower_module.rs` → `lower_module()`

Lowers all functions from SSA-IR to MachineIR — a target-neutral register-based IR ready for architecture backends.

### 4a. Register File Partitioning

`machine/lower_regalloc.rs` → `MachineRegFile::new()`

Partitions the virtual register file into fixed regions: `[fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache]`

Fixed registers: context pointer, frame pointer, memory base, memory size. Local cache registers are pinned to high-frequency locals. Transient registers serve SSA temporaries within a bounded budget.

### 4b. Per-Function Lowering

`machine/lower_module.rs` → `lower_function()`

For each SSA block, creates a `BlockLowerContext` and processes each instruction:

- **LoadSlot** → frame load into transient or cached register
- **StoreSlot** → register store to frame slot
- **Value ops** → dispatched through `lower_inst.rs` → `lower_leaf_arith.rs` (arithmetic, compare, convert, select) and `lower_leaf_special.rs` (div, rem, memory load/store, atomics, traps)
- **Boundary ops** → `lower_boundary.rs` for memory/table runtime calls, internal/external/indirect calls
- **Terminators** → Goto, Branch (with condition), BrTable, Return, Unreachable

Register allocation is one-pass: transient registers are allocated/freed as values are produced/consumed. Dead-value reuse prefers recycling registers whose values just died.

### 4c. I64 Legalization (32-bit targets only)

`machine/gp32/lower_leaf.rs` → `Gp32Lowering`

On 32-bit GP targets, all 64-bit operations are legalized into register-pair instructions: `Int64PairBinary`, `Int64PairUnary`, `Int64PairDivRem`, `Int64PairShiftRotate`. Each expands to multiple 32-bit ops with carry/borrow handling in the backend.

64-bit targets use `Gp64Lowering` (`lower_i64_gp64.rs`) which emits native 64-bit instructions directly.

### 4d. Sidecar Metadata Collection

`machine/lower_sidecar.rs` → `SidecarBuilder`

Collects constants and external binding metadata for runtime helper calls (memory.grow, table.init, etc.). Stored as `MachineConstData` and `MachineExternBinding` in the MachineProgram.

## Phase 5: Machine IR Optimization

`machine/peephole.rs` → `optimize()`

Six intra-block peephole passes, run in sequence:

1. **Constant deduplication** (`deduplicate_constants`): When the same non-zero constant is materialized multiple times (via `Move { Imm64 }` or `FloatConst`), replaces duplicates with register copies from the first.

2. **Copy propagation** (`copy_propagate`): Rewrites uses of `move rTmp <- rSrc` to reference `rSrc` directly. Also folds single-use `move rX <- Imm64(C)` into consumer operands as inline immediates. Run twice (before and after store-to-load forwarding).

3. **Store-to-load forwarding** (`forward_stored_values`): Pattern: `store.u64 [addr] <- X; ...; load.u64 rY <- [addr]` → `move rY <- X` when no intervening op can invalidate the address or source.

4. **Load-to-load reuse** (`reuse_loaded_values`): When the same address is loaded twice with no intervening store that could alias, replaces the second load with a register copy from the first.

5. **Indexed memory fusion** (`fuse_indexed_memory`): Two patterns:
   - Pattern A: `cvt.I64ExtendI32U + [offset_add] + base_add + load/store` → `IndexedLoad/Store { base, index, index_extend: UXTW, offset }`
   - Pattern B: `base_add + load/store` → `IndexedLoad/Store { base, index, offset }`

6. **Compare-and-branch fusion** (`fuse_compare_branch`): Fuses `IntCompare { dst } + Branch { Reg(dst) }` into `Branch { IntCompare { ... } }` when the compare result is dead in both successors. Eliminates boolean materialization (CSET/SETCC). FloatCompare is NOT fused due to x86_64 NaN handling complexity.

## Phase 5.5: Machine IR Validation

`machine/validate.rs` → `MachineModule::validate()`

Debug-mode checks: register bounds, storage type vs register bank consistency, block parameter validity, terminator target validity.

On 32-bit targets, `validate_32bit_gp_target()` verifies no raw 64-bit instructions remain — all must use `Int64Pair*` variants.

## Phase 6: Code Emission

`arch/common/pipeline.rs` → `compile_function<A: ArchBackend>()`

Emits native machine code from MachineIR through an architecture-specific backend (ARM64, x86_64, ARMv7a).

### 6a. Prologue

Allocates stack frame, saves callee-saved registers, loads fixed registers (context pointer, frame pointer, memory base/size from the runtime context).

### 6b. Block Emission

Iterates blocks in layout order. For each block: binds the block label, emits each `MachineInst` via the backend's instruction selector, emits the terminator. The backend handles:
- **Operand preparation**: Maps virtual MachineRegs to physical registers. Immediates are materialized into scratch registers when they can't be encoded inline.
- **Immediate selection** (`arch/arm64/fusion.rs`, `arch/x86_64/fusion.rs`): Tries reg+imm forms (add/sub imm12, logical bitmask, shift immediate, mul by power-of-two via shift) before falling back to scratch materialization.
- **Float compare-branch fusion** (arch-specific): ARM64 emits `fcmp + b.cond`. x86_64 emits `ucomisd/ucomiss + jp/jcc` multi-instruction sequences.
- **Zero-store pair fusion** (ARM64): Consecutive stores of zero to adjacent addresses fused into `stp xzr, xzr`.

### 6c. Edge Stubs

Emits parameter-moving glue between blocks. Uses a parallel-move algorithm with cycle detection: finds ready moves (destination not used as source), emits them, breaks cycles by spilling to scratch registers.

### 6d. Tail

Emits the return-ok path (epilogue: restore callee-saved regs, deallocate frame, ret), stack-overflow trap, return-error path, and all deferred trap stubs.

### 6e. Fixup Patching

Resolves all forward branch references: patches branch instruction offsets now that all label positions are known.

### 6f. Output

Produces `FunctionArtifact`: native code bytes, entry point address, root-return offset, and debug regions for diagnostics. The artifacts are installed into `NativeCode` objects on each function spec, backed by a shared `CompiledNativeModule`.

## Data Flow Summary

```
Wasm bytecode
  → SemanticProgram          (structured control, abstract locals)
  → SemanticProgram           (with inlined callees)
  → SsaProgram               (flat blocks, frame slots, spill/fill, SSA values)
  → SsaProgram               (optimized: slot forwarding, const fold, DCE)
  → MachineModule             (virtual regs, target-neutral MachineInsts)
  → MachineModule             (peephole-optimized: dedup, copy prop, forwarding, fusion)
  → FunctionArtifact[]        (native code bytes per function)
```
