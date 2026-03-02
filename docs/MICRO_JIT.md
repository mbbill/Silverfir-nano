# Micro-JIT: Runtime Assembly of Fused Basic Blocks

## Core Idea

The TOS register window + L0/L1/L2 hot local cache already define a fixed physical register assignment for all hot values. This means a "JIT compiler" for fused blocks doesn't need SSA, register allocation, or an optimizing compiler — it just needs a **micro-assembler** that emits 1-2 ARM64 instructions per Wasm opcode using the known register mapping, plus a trivial **alias/constant tracker** that eliminates redundant moves and folds immediates.

The approach: for each basic block, walk the instruction sequence, emit machine code for each operation using the pre-assigned registers, and output to an mmap'd executable buffer. A single forward pass with ~4 simple optimizations (alias tracking, constant folding, destination forwarding, compare+branch fusion) produces code quality close to clang -O2 for the common cases.

No external compiler dependency. Microsecond assembly time. Self-contained.

## ARM64 Register Map (from preserve_none ABI)

```
ctx=x20  pc=x21  fp=x22
l0=x23   l1=x24  l2=x25
t0=x26   t1=x27  t2=x28  t3=x0
nh=x1
```

## Key Design Decisions

### 1. The assembler operates at the SEM_* abstraction level

Each Wasm opcode maps to a known machine code template (1-3 instructions) with register operands determined by TOS depth. The assembler doesn't see "handlers" — it sees `add reg, reg, reg` with specific registers filled in based on the compile-time-known TOS state.

### 2. Four optimizations (all single forward pass)

- **Alias tracking**: `local_get_lN` records that the TOS slot aliases lN's register. No mov emitted. When a subsequent instruction reads that TOS slot, the assembler substitutes lN's physical register directly.
- **Constant tracking**: `i32_const(K)` records K as a pending constant. When the next instruction consumes it, K is folded into the instruction's immediate field (ARM64 12-bit for add/sub/cmp, bitmask encoding for logical ops).
- **Destination forwarding**: When an operation's result flows immediately into `local_set_lN`, emit the operation with lN's register as the destination, eliminating the mov.
- **Compare+branch fusion**: `i32_lt_s + br_if` → `cmp + b.lt`, eliminating boolean materialization.

### 3. Dispatch at block exits

Each exit point emits 3-4 instructions: advance pc, load handler, preload nh, branch. All other registers are already correct because the compiled block uses the same physical register mapping as the interpreter.

Branch targets: use ARM64 PC-relative literal loads from a small literal pool appended to the code block. Target `Instruction*` addresses are known at codegen time.

### 4. Inline bounds checks for trapping ops

Memory loads/stores emit: compute effective address → compare against `ctx_mem0_size(ctx)` → conditional branch to error path. Error path sets up trap state and branches to the existing trap handler (not a function call — a tail-branch that reuses the current register state).

## Hard Parts and Corner Cases

### TOS window overflow (depth > 4)

When a block pushes more than 4 values, the micro-assembler must spill to frame memory: `str x_reg, [x22, #(spill_slot * 8)]`. The spill slot indices must match what the interpreter's static compilation assigns, so that if we dispatch out of the compiled block mid-sequence, the interpreter can pick up correctly.

**This is the trickiest correctness issue.** The compiled block must maintain the invariant that the interpreter's spill_depth accounting stays consistent. On entry, the block knows its TOS depth and spill_depth. On exit, both must match what the next handler expects.

Decision: initially, limit compilation to blocks where TOS depth stays ≤ 4 throughout. This covers most hot loops (arithmetic + locals). Extend to spilling blocks later.

### Non-hot locals

`local_get(idx)` where idx ∉ {0,1,2} emits `ldr x_tmp, [x22, #(idx*8)]`. This is a real memory access. The alias tracker can still help: if the same non-hot local is read twice in the block without an intervening write, the second load can reuse the register from the first.

### The error path for trapping ops

When a bounds check fails, we need to get back to the interpreter's trap handling. The challenge: `c_trap()` is a function call, which breaks the leaf-function property.

**Preferred**: Don't call `c_trap()`. Instead, emit a branch to a pre-compiled `preserve_none` trap stub that sets `ctx->trap_reason` and dispatches to the interpreter's trap handler. The compiled block remains a leaf function.

The one subtlety: `t3 = x0` and `nh = x1` overlap with the ARM64 standard ABI's first two argument/return registers. If the block calls any standard-ABI function (even a helper), x0 and x1 get clobbered. This is why trapping ops need the stub approach rather than a direct function call.

### Interaction with existing dispatch chain

The compiled block replaces `instruction[block_start].handler`. The remaining instructions in the block still exist in the instruction stream (needed for `pc + N` computation and for fallback). If the compiled block is invalidated (e.g., module unloaded), we restore the original handler pointers.

### Memory management

Each compiled block is a small allocation (typically 50-500 bytes). Arena allocator: `mmap` a large region (e.g., 1MB), bump-allocate blocks within it. Arena with generation tracking for cleanup. Standard approach for JIT code buffers.

### x86-64 portability

ARM64 has fixed-width 4-byte instructions — trivial to emit. x86-64 has variable-length encoding (1-15 bytes per instruction), making the assembler significantly more complex. Options:
- ARM64 only initially
- Use an existing x86-64 assembler crate (`iced-x86`, `dynasm-rs`)
- Write a minimal x86-64 emitter for just the ~30 instruction patterns we need

The `preserve_none` register mapping on x86-64 is different (uses RDI, RSI, RDX, RCX, R8-R15, RAX for arguments). The same design applies, just different register names.

### When NOT to compile a block

Some blocks are better left to the interpreter:
- Very short blocks (1-2 instructions) — dispatch overhead is already minimal with static fusion
- Blocks dominated by `call` / `call_indirect` — these exit the block immediately
- Blocks with `br_table` — multi-way dispatch is complex to emit
- Blocks where TOS depth exceeds 4 (initially)

A simple heuristic: compile blocks with ≥ 4 instructions where ≥ 50% are arithmetic/local ops.

## What Makes This Different from Existing Work

- **Not copy-and-patch**: we don't copy pre-compiled handler blobs. We emit fresh machine code per block, with knowledge of the specific immediate values and register state. This enables constant folding and alias elimination that copy-and-patch cannot do.
- **Not a baseline JIT (Winch/Liftoff)**: those do single-pass register allocation over the virtual stack. We skip register allocation entirely — the TOS + L0/L1/L2 system provides a fixed, pre-determined mapping.
- **Not wasm2c**: we don't generate C or invoke a compiler. The assembler is embedded in the interpreter binary (~300-500 lines of Rust for ARM64).
- **Leverages the interpreter's architecture**: the TOS window and hot local cache, designed for interpreter fusion, turn out to be exactly the right abstraction for trivial code generation. The register assignment that makes fusion handlers fast also makes assembly trivial.

## Size and Embedded Positioning

The micro-assembler adds an estimated ~10-20 KB to the binary (ARM64 instruction encoding for ~30 patterns + codegen logic + code buffer management). Combined with the ~230 KB interpreter core, the total is approximately **~250 KB** for a Wasm runtime with JIT capability.

For comparison, existing Wasm JIT runtimes:

| Runtime | Approximate Size |
|---------|-----------------|
| V8 Wasm engine | ~30 MB |
| Wasmtime + Cranelift | ~15-20 MB |
| Wasmer + LLVM | ~100+ MB |
| WAMR + LLVM JIT | ~50+ MB |
| WAMR Fast JIT | ~1-2 MB |
| **Silverfir-nano + micro-JIT** | **~250 KB** |

This is 60-400x smaller than any existing Wasm JIT. On MCU-class devices with AArch64 or RV64 cores (Cortex-A35/A53 in IoT, SiFive RISC-V), SRAM is typically executable without `mmap` — just allocate and write code. This could be the first Wasm JIT that runs on resource-constrained embedded devices where existing JIT runtimes cannot fit.

The reason this is possible: existing JIT compilers need a full compiler framework (IR, optimization passes, register allocator). The micro-JIT needs none of this because the interpreter's TOS + L0/L1/L2 architecture already provides the register assignment. The "compiler" is just a template assembler — the architectural innovation that made the interpreter fast is the same innovation that makes the JIT trivially small.

## Expected Performance

Per-basic-block compilation eliminates all intra-block dispatch overhead and enables constant folding + alias elimination. Estimated improvement over current static fusion: **10-25%** on compute-intensive workloads (CoreMark, tight loops). The gain is larger for blocks that static fusion handles poorly (long sequences exceeding the 3-imm capacity, patterns not in the static set).

The ceiling is below Cranelift because we can't optimize across block boundaries. But the simplicity is the point — a few hundred lines of Rust vs a full compiler.

## Implementation Plan: Step-by-Step

The guiding principle: **every step is independently verifiable without executing generated code until we've proven the encoding is correct**. A single wrong bit in a u32 causes an undebuggable segfault. We build trust incrementally — encoding correctness first, then trivial execution, then progressively complex handlers.

### Debugging Strategy

Before any step involves execution, we establish two tools:

1. **Disassembly comparison**: Every emitted `u32` is disassembled (via `llvm-objdump` or a disassembler crate) and compared against expected text. This catches encoding bugs as text diffs, not segfaults.

2. **Trace buffer**: A memory region where JIT code writes register snapshots (`str x23, [x_trace, #offset]`) at instruction boundaries. After execution, the Rust side diffs the trace against the interpreter's execution of the same block. Any divergence pinpoints exactly which instruction emitted wrong code. No function calls needed (just stores), so no ABI concerns.

### Step 1: ARM64 Instruction Encoder (pure functions, no execution)

Build encoding functions: each takes register operands + immediates, returns a `u32`. Approximately 30 patterns needed: `add/sub/and/orr/eor/mul/sdiv/udiv` (reg and imm forms), `lsl/lsr/asr`, `cmp`, `cset`, `ldr/str` (various widths), `mov/movz/movk`, `b/b.cond/cbz/cbnz/br`, `sxt/uxt`.

**Verification**: Write unit tests that encode each instruction and compare the bytes against known-good output from `as` or `llvm-mc`. Example: `arm64_add_imm(W23, W23, 1)` must produce `0x110005f7`. Feed the expected bytes through a disassembler to triple-check. No mmap, no execution — pure data transformation. This step has zero risk.

### Step 2: Code Buffer (mmap + trivial execution)

Implement the executable memory allocator: `mmap` a page with `PROT_READ|PROT_WRITE`, write instructions, `mprotect` to `PROT_READ|PROT_EXEC` (or use `MAP_JIT` + `pthread_jit_write_protect_np` on macOS).

**Verification**: Write the simplest possible function — a single `ret` instruction (or `br x30`). Cast the buffer to a function pointer, call it. If it returns without crashing, the mmap/execution pipeline works. Then test `mov x0, #42; ret` and verify the return value equals 42. This tests the execution infrastructure with code simple enough to verify by inspection.

### Step 3: The Dispatch Stub (preserve_none ABI compatibility)

This is the first step that touches the interpreter's dispatch chain. Emit a **no-op handler**: it does zero work, just dispatches to the next instruction. The emitted code:

```
add  x21, x21, #0x20     ; advance pc by 1 instruction (32 bytes)
ldr  x2, [x21]           ; load handler at new pc
ldr  x1, [x21, #0x20]    ; preload nh
br   x2                   ; tail-jump
```

**Verification**: Write a test Wasm module with a known instruction sequence, e.g., `i32_const(1), i32_const(2), i32_add`. Replace the middle instruction's handler with the dispatch stub (the stub skips that instruction). If the module still produces the right result via the remaining handlers, the ABI contract is proven — our generated code can enter and exit the dispatch chain correctly.

This is the **most critical milestone**. If this works, we've proven: mmap code is callable, preserve_none register layout is correct, pc advancement works, nh preloading works, and tail-jump dispatch is ABI-compatible. Everything after this is incremental.

### Step 4: Single-Instruction Emission (one opcode at a time)

Emit a handler for ONE instruction type. Start with `i32_add` at a specific TOS depth (D2, the most common). The emitted code does the operation on the known registers, then dispatches (reusing the dispatch stub from Step 3).

**Verification**: Write a test module that executes `i32_add`. Replace the interpreter's handler with the JIT-emitted one. Compare output. Then add `i32_sub`, `i32_and`, `i32_or`, `i32_xor`, `i32_mul`, `i32_shl`, `i32_shr_u`, `i32_shr_s` — each is one additional `u32` encoding pattern, independently testable.

**Key constraint**: at this step, each JIT handler covers exactly ONE instruction and dispatches. The emitted code is functionally equivalent to the C handler it replaces — same registers, same semantics, same dispatch. This means the spec test suite can validate correctness for each opcode we add.

### Step 5: Multi-Instruction Concatenation (no optimization)

Emit code for a basic block of N instructions in sequence. No alias tracking, no constant folding — just the naive emission: each `local_get_lN` emits a `mov`, each `i32_const` emits a `movz`, each arithmetic op emits the operation on TOS registers. The dispatch stub goes at the end only (not after each instruction).

**Verification**: Start with a 2-instruction block. Then 3. Then N. At each step, compare the module's output against the interpreter. The key invariant: the JIT block produces identical register state at its exit as the interpreter would after executing the same N instructions.

Use the trace buffer here: emit `str` instructions at each internal boundary to record register values. Diff against interpreter execution.

### Step 6: Alias Tracking

Add the `RegState` tracker (Alias/Const/Value/Empty). When `local_get_lN` is encountered, record the alias instead of emitting a mov. When a subsequent instruction reads that TOS slot, resolve the alias to the physical register.

**Verification**: For known instruction sequences, assert on the number of emitted instructions. Example: `local_get_l0 → local_get_l1 → i32_add` should emit 1 instruction (the add), not 3. Disassemble and compare. Then run the module and verify output — fewer instructions, same result.

### Step 7: Constant Folding

When `i32_const(K)` is encountered, record K in the tracker. When consumed by the next instruction, fold K into the immediate field if it fits (12-bit for add/sub/cmp). If it doesn't fit, materialize with `movz`/`movk`.

**Verification**: `i32_const(1) → i32_add` should produce an add-immediate instruction, not a movz + add-register. Check the disassembly. Then verify with constants that don't fit (e.g., 0xDEAD) to test the materialization path.

### Step 8: Destination Forwarding

When the instruction after an operation is `local_set_lN`, emit the operation with lN's register as the destination instead of the TOS register.

**Verification**: `local_get_l0 → i32_const(1) → i32_add → local_set_l0` should produce a single `add w23, w23, #1`. Disassemble, verify, run.

### Step 9: Conditional Branch Dispatch (br_if)

Emit both exit paths: taken (branch to target) and fall-through. For the target, load the `Instruction*` address from a literal pool appended to the code block. For compare+branch fusion (`i32_lt_s + br_if` → `cmp + b.lt`), implement as a peephole on the last two instructions of the block.

**Verification**: Write a test module with a loop (the simplest `br_if` target — back to loop header). Verify it terminates correctly and produces the right output. This is the first test of branching in JIT code.

**Corner case**: the literal pool must be placed after the last instruction and aligned. ARM64 `ldr Xn, label` is PC-relative with ±1MB range. The offset must be computed after all code is emitted (requires a fixup pass or two-pass emission).

### Step 10: Trapping Ops (memory access, division)

Emit inline bounds checks. The key decision: on trap, branch to a pre-compiled `preserve_none` trap stub rather than calling `c_trap()`. The stub must be compiled as part of the interpreter binary (a Rust/C function with the right ABI) and its address baked into the JIT code as a literal pool entry.

**The hard part**: loading `ctx_mem0_size(ctx)` requires knowing the offset of the memory size field within the `Context` struct. This creates a coupling between the JIT codegen and the Context layout. If Context changes, the JIT-emitted offsets become wrong — silently. Solution: a compile-time constant (or test) that verifies the offset.

**Verification**: Test with Wasm modules that perform in-bounds and out-of-bounds memory access. Verify that in-bounds produces correct values and out-of-bounds produces the expected trap.

### Step 11: Block Selection and Integration

Wire everything together: at module load time (or lazily), identify basic blocks that meet the compilation criteria (≥ N instructions, TOS depth ≤ 4, supported opcodes). Compile eligible blocks and patch their handlers.

**Verification**: Run the full spec test suite with micro-JIT enabled. Any failure indicates a codegen bug — the failing test isolates the problematic opcode/pattern. Then benchmark CoreMark.

### Step Dependencies

```
Step 1 (encoder)
  └→ Step 2 (mmap)
       └→ Step 3 (dispatch stub)         ← CRITICAL MILESTONE
            └→ Step 4 (single instructions)
                 └→ Step 5 (multi-instruction, naive)
                      ├→ Step 6 (alias tracking)
                      ├→ Step 7 (constant folding)
                      ├→ Step 8 (destination forwarding)
                      └→ Step 9 (br_if dispatch)
                           └→ Step 10 (trapping ops)
                                └→ Step 11 (integration)
```

Steps 6-8 are independent optimizations that can be added in any order. Each is a diff on the codegen pass, independently testable.
