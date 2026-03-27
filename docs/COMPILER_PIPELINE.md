# Micro-JIT Compiler Pipeline

Entry point: `vm/build.rs` → `ensure_module_compiled()`

Wasm Bytecode → **Semantic IR** → **SSA-IR** → **Machine IR** → **Binary**

- Wasm Bytecode → Semantic IR
  - Decode
  - Function Inlining
- Semantic IR → SSA-IR
  - Semantic IR Validation
  - Frame Layout Planning
  - Local Cache Preference Analysis
  - Spill/Fill Planning
  - Control-Flow Graph Construction
  - Block-Level Lowering
  - SSA-IR Optimization
  - Sink Planning
  - SSA-IR Validation
- SSA-IR → Machine IR
  - Register File Partitioning
  - Per-Function Lowering
  - I64 Legalization
  - Sidecar Metadata Collection
  - Peephole Optimization
  - Machine IR Validation
- Machine IR → Binary
  - Prologue
  - Block Emission
  - Edge Stubs
  - Tail
  - Fixup Patching
  - Output

## Wasm Bytecode → Semantic IR

### Decode

`wasm/decode.rs` → `decode_to_semantic_ir()`

Decodes Wasm bytecode into `SemanticProgram`: a list of `SemanticOp`s with structured control flow (Block, Loop, If/Else), abstract local access (LocalGet/Set/Tee), and semantic calls (CallInternal/External/Indirect). No frame layout or register decisions yet. Tracks `max_stack_height`, local types, and result types.

### Function Inlining

`wasm/inline.rs` → `inline_calls_in_function()`

Replaces `CallInternal` ops with the callee's body when the callee is a small leaf function (no calls, ≤200 ops, ≤8 params). Iterates to fixed-point so transitive chains (A→B→C) are fully resolved regardless of index ordering.

## Semantic IR → SSA-IR

`middle/mod.rs` → `prepare_function()`

Transforms semantic IR into a flat basic-block SSA-IR with explicit frame slots, spill/fill, and register budgets. This is a multi-step process:

### Semantic IR Validation

`semantic_ir.rs` → `SemanticProgram::validate()`

Checks structural invariants of the decoded semantic program before lowering.

### Frame Layout Planning

`middle/frame.rs` → `plan_frame_layout()`

Allocates frame slots for all locals and stack values. Produces `FrameLayoutPlan` with canonical slot assignments for params, non-param locals, stack temporaries, call-scratch region, and total frame size.

### Local Cache Preference Analysis

`middle/local_cache.rs` → `analyze_local_cache_prefs()`

Determines which locals should be pinned to dedicated registers (cached locals). Produces separate GP and FP preference lists based on access frequency. Also computes continuation-skip-reload info (which cached locals don't need reload after a call returns).

### Spill/Fill Planning

`middle/spill_plan.rs` → `prepare_semantic_ops()`

Walks the semantic stream tracking conceptual value-stack height. When height exceeds the transient register budget, inserts explicit spill/fill prefix actions. Produces `PreparedStream` with `PreparedOp`s (semantic op + prefix spill/fill actions) and per-block `entry_states` (what values are live at each block entry point).

### Control-Flow Graph Construction

`middle/lower_cfg.rs` → `build_block_ranges()` + `retain_reachable_blocks()`

Converts structured control flow (nested blocks, loops, if/else) into flat basic-block ranges with explicit edges. Eliminates unreachable blocks. Builds the `semantic_to_block` index mapping.

### Block-Level Lowering

`middle/lower_block.rs` → `lower_block_range()`

For each basic block, lowers semantic ops to SSA instructions:
- LocalGet/Set/Tee → `LocalGet`/`LocalSet`/(`LocalSet` + `LocalGet`) with local version tracking
- Spill/fill of stack temporaries → `Spill`/`Fill` (distinct from local access)
- Arithmetic/conversion ops → `SsaInstKind::Value` with `SsaLeafOp`
- Memory/table/call ops → `SsaInstKind::Boundary`
- Control flow edges → SSA terminators (Goto, Branch, BrTable, Return)

`LocalSet` carries a `version` field — a monotonically increasing counter per local slot. Each `LocalGet` records a `ValueHome::LocalVersion { slot, version }` on its destination value, enabling the sink planner to reason about local liveness.

Edge lowering (`lower_edge.rs`) and terminator lowering (`lower_term.rs`) handle block parameter passing and multi-way branches.

### SSA-IR Optimization

`middle/optimize.rs` → `optimize_ssa()`

Two intra-block passes:

1. **Slot-value forwarding** (`forward_slot_values`): Tracks what value was last stored to each local slot. When a `LocalGet` reads a slot whose value is still live in a register, replaces the load with an alias. Eliminates redundant LocalSet/LocalGet round-trips. Only applies to local slots — `Spill`/`Fill` for stack temporaries are left unmodified.

2. **Constant folding into operands** (`fold_constants_into_operands`): Two rewrites in one pass:
   - *Full evaluation*: When all args of a pure op are known constants, evaluates at compile time and replaces the op with a const definition. Results cascade through the block.
   - *Operand folding*: Replaces `SsaOperand::Value(v)` with `SsaOperand::Const(bits)` when v is a single-use constant. Backends that can't encode the immediate natively use the pre-budgeted transient register.
   - *Dead const elimination*: Removes constant definitions whose results are no longer referenced after folding.

### Sink Planning

`middle/sink_plan.rs` → `plan_sinks()`

Runs after SSA optimization. For each `LocalSet { slot, src, version }`, determines whether the producer of `src` can write its result directly into the local's home register instead of a transient. When legal, annotates the value in `value_sink_local` so the machine lowering can elide the `LocalSet` move.

A sink is legal when:
- `src` is produced by a single-result `Value` instruction in the same block
- No `Boundary` (call/runtime op) exists between the producer and the `LocalSet`
- The previous version of the local is dead at the producer — no `LocalGet` reads the old version between the producer and the `LocalSet`

### SSA-IR Validation

`middle/ssa_ir/validate.rs` → `validate_program()`

Checks block structure consistency, value use/def chains, and type correctness of the produced SSA-IR.

## SSA-IR → Machine IR

`machine/lower_module.rs` → `lower_module()`

Lowers all functions from SSA-IR to MachineIR — a target-neutral register-based IR ready for architecture backends.

### Register File Partitioning

`machine/lower_regalloc.rs` → `MachineRegFile::new()`

Partitions the virtual register file into fixed regions: `[fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache]`

Fixed registers: context pointer, frame pointer, memory base, memory size. Local cache registers are pinned to high-frequency locals. Transient registers serve SSA temporaries within a bounded budget.

### Per-Function Lowering

`machine/lower_module.rs` → `lower_function()`

For each SSA block, creates a `BlockLowerContext` and processes each instruction:

- **LocalGet** → if the local is cached, **source-aliases** the SSA value to the cache register (zero-copy, no instruction emitted); otherwise loads from the frame slot into a transient
- **LocalSet** → if the value is already in the target cache register (via sink annotation or same-local tee), elided; otherwise moves the value into the cache register or stores to the frame slot. Before overwriting a cache register, `materialize_cache_aliases()` spills any other live values aliased to it into transients.
- **Fill** → always loads from frame slot into a transient (stack temporary reload)
- **Spill** → always stores from register to frame slot (stack temporary publish)
- **Value ops** → dispatched through `lower_inst.rs` → `lower_leaf_arith.rs` (arithmetic, compare, convert, select) and `lower_leaf_special.rs` (div, rem, memory load/store, atomics, traps). When a sink annotation targets a cached local, the result register is pre-mapped to the cache register so the instruction writes directly there.
- **Boundary ops** → `lower_boundary.rs` for memory/table runtime calls, internal/external/indirect calls
- **Terminators** → Goto, Branch (with condition), BrTable, Return, Unreachable

Register allocation is one-pass: transient registers are allocated/freed as values are produced/consumed. Dead-value reuse prefers recycling registers whose values just died. Values source-aliased to cache registers are tracked but their registers are not freed — they belong to the cached local.

### I64 Legalization (32-bit targets only)

`machine/gp32/lower_leaf.rs` → `Gp32Lowering`

On 32-bit GP targets, all 64-bit operations are legalized into register-pair instructions: `Int64PairBinary`, `Int64PairUnary`, `Int64PairDivRem`, `Int64PairShiftRotate`. Each expands to multiple 32-bit ops with carry/borrow handling in the backend.

64-bit targets use `Gp64Lowering` (`lower_i64_gp64.rs`) which emits native 64-bit instructions directly.

### Sidecar Metadata Collection

`machine/lower_sidecar.rs` → `SidecarBuilder`

Collects constants and external binding metadata for runtime helper calls (memory.grow, table.init, etc.). Stored as `MachineConstData` and `MachineExternBinding` in the MachineProgram.

### Peephole Optimization

`machine/peephole.rs` → `optimize()`

Six intra-block peephole passes, run in sequence:

1. **Constant deduplication** (`deduplicate_constants`): When the same non-zero constant is materialized multiple times (via `Move { Imm64 }` or `FloatConst`), replaces duplicates with register copies from the first.

2. **Copy propagation** (`copy_propagate`): Rewrites uses of `move rTmp <- rSrc` to reference `rSrc` directly. Also folds single-use `move rX <- Imm64(C)` into consumer operands as inline immediates. Run twice (before and after store-to-load forwarding).

3. **Store-to-load forwarding** (`forward_stored_values`): Pattern: `store.u64 [addr] <- X; ...; load.u64 rY <- [addr]` → `move rY <- X` when no intervening op can invalidate the address or source.

4. **Load-to-load reuse** (`reuse_loaded_values`): When the same address is loaded twice with no intervening store that could alias, replaces the second load with a register copy from the first.

5. **Indexed memory fusion** (`fuse_indexed_memory`): Two patterns:
   - Pattern A: `cvt.I64ExtendI32U + [offset_add] + base_add + load/store` → `IndexedLoad/Store { base, index, index_extend: UXTW, offset }`
   - Pattern B: `base_add + load/store` → `IndexedLoad/Store { base, index, offset }`

6. **Compare-and-branch fusion** (`fuse_compare_branch`): Fuses `IntCompare { dst } + Branch { Reg(dst) }` into `Branch { IntCompare { ... } }` when the compare result register is a transient and is dead in both successors. Cached-local and fixed registers are implicitly live across block boundaries, so fusion is only safe for transients. Eliminates boolean materialization (CSET/SETCC). FloatCompare is NOT fused at this level due to x86_64 NaN handling complexity; ARM64 performs float compare-branch fusion in the arch backend (`arch/arm64/fusion.rs`) with the same transient-register guard.

### Machine IR Validation

`machine/validate.rs` → `MachineModule::validate()`

Debug-mode checks: register bounds, storage type vs register bank consistency, block parameter validity, terminator target validity.

On 32-bit targets, `validate_32bit_gp_target()` verifies no raw 64-bit instructions remain — all must use `Int64Pair*` variants.

## Machine IR → Binary

`arch/common/pipeline.rs` → `compile_function<A: ArchBackend>()`

Emits native machine code from MachineIR through an architecture-specific backend (ARM64, x86_64, ARMv7a).

### Prologue

Allocates stack frame, saves callee-saved registers, loads fixed registers (context pointer, frame pointer, memory base/size from the runtime context).

### Block Emission

Iterates blocks in layout order. For each block: binds the block label, emits each `MachineInst` via the backend's instruction selector, emits the terminator. The backend handles:
- **Operand preparation**: Maps virtual MachineRegs to physical registers. Immediates are materialized into scratch registers when they can't be encoded inline.
- **Immediate selection** (`arch/arm64/fusion.rs`, `arch/x86_64/fusion.rs`): Tries reg+imm forms (add/sub imm12, logical bitmask, shift immediate, mul by power-of-two via shift) before falling back to scratch materialization.
- **Float compare-branch fusion** (arch-specific): ARM64 emits `fcmp + b.cond`. x86_64 emits `ucomisd/ucomiss + jp/jcc` multi-instruction sequences.
- **Zero-store pair fusion** (ARM64): Consecutive stores of zero to adjacent addresses fused into `stp xzr, xzr`.

### Edge Stubs

Emits parameter-moving glue between blocks. Uses a parallel-move algorithm with cycle detection: finds ready moves (destination not used as source), emits them, breaks cycles by spilling to scratch registers.

### Tail

Emits the return-ok path (epilogue: restore callee-saved regs, deallocate frame, ret), stack-overflow trap, return-error path, and all deferred trap stubs.

### Fixup Patching

Resolves all forward branch references: patches branch instruction offsets now that all label positions are known.

### Output

Produces `FunctionArtifact`: native code bytes, entry point address, root-return offset, and debug regions for diagnostics. The artifacts are installed into `NativeCode` objects on each function spec, backed by a shared `CompiledNativeModule`.

