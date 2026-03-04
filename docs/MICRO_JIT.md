# Micro-JIT: Runtime Fusion via Micro-Assembly

## Core Idea

The TOS register window + L0/L1/L2 hot local cache already define a fixed physical register assignment for all hot values. The builder's depth variant selection (D1-D4) determines which physical registers each instruction operates on. This means a "JIT compiler" doesn't need SSA, register allocation, or an optimizing compiler — it just needs a **micro-assembler** that emits 1-2 ARM64 instructions per variant instruction using the known register mapping.

The micro-JIT **is** the fusion system, not a layer on top of it. It plugs into the existing builder pipeline after variant selection and spill/fill insertion. The builder still decodes Wasm, tracks TOS depth, selects D1-D4 variants, and inserts spill/fill instructions. The JIT replaces the final stage: instead of assigning pre-compiled C handler pointers, it groups consecutive variant instructions into **JIT groups**, emits machine code for each group, and uses the machine code address as the handler pointer. Opcodes the JIT doesn't handle (calls, br_table, etc.) keep their pre-compiled C handlers — the dispatch chain interleaves JIT'd groups and built-in handlers seamlessly.

This replaces static fusion on JIT-capable platforms, removing all of static fusion's limitations: the 3-immediate encoding budget, the finite pattern set, the workload-dependent discovery step.

No external compiler dependency. Microsecond assembly time. Self-contained.

## ARM64 Register Map (from preserve_none ABI)

```
ctx=x20  pc=x21  fp=x22
l0=x23   l1=x24  l2=x25
t0=x26   t1=x27  t2=x28  t3=x0
nh=x1
```

## Key Design Decisions

### 1. Opcode-to-machine-code mapping (Rust, independent of C SEM_*)

The micro-JIT's codegen is a separate system from the C `SEM_*` macros in `semantics.h`. Both encode the same Wasm semantics, but in different representations:

- **C SEM_***: `SEM_I32_ADD(a, b)` → C expression → clang → machine code (static fusion)
- **JIT codegen**: `emit_i32_add(dst, src1, src2)` → ARM64 `u32` → code buffer (micro-JIT)

They coexist independently. The C SEM_* remain the source of truth for static fusion on non-JIT platforms. The Rust codegen functions are the source of truth for the micro-JIT.

### 2. Four optimizations (deferred — not for initial bring-up)

All four are single-forward-pass peephole optimizations with 1-instruction lookahead. They are **nice-to-have** and should be implemented only after the naive JIT (Steps 1-5, 9-11) is stable and passing spec tests.

- **Alias tracking**: `local_get_lN` records that the TOS slot aliases lN's register. No mov emitted. When a subsequent instruction reads that TOS slot, the assembler substitutes lN's physical register directly.
- **Constant tracking**: `i32_const(K)` records K as a pending constant. When the next instruction consumes it, K is folded into the instruction's immediate field (ARM64 12-bit for add/sub/cmp, bitmask encoding for logical ops).
- **Destination forwarding**: When an operation's result flows immediately into `local_set_lN`, emit the operation with lN's register as the destination, eliminating the mov.
- **Compare+branch fusion**: `i32_lt_s + br_if` → `cmp + b.lt`, eliminating boolean materialization.

### 3. Dispatch at group exits

Each JIT group exit advances pc past its instruction slot, loads the next handler, and branches. The dispatch sequence can be any number of instructions — there is no fixed template.

The next handler might be another JIT group or a pre-compiled C handler (e.g., `call`). It doesn't matter — both follow the same `preserve_none` ABI register layout, so the dispatch is identical. The register contract (ctx=x20, pc=x21, fp=x22, l0-l2=x23-x25, t0-t3=x26-x28/x0, nh=x1) must be maintained at every group boundary.

Branch target `Instruction*` pointers can be stored in the group's Instruction slot (imm0), loaded via `ldr x_target, [x21, #8]`. This reuses the existing finalizer's two-pass target patching and avoids the need for a separate literal pool — the target address is on the same cache line as the handler pointer we just fetched.

### 4. Inline bounds checks for trapping ops

Memory loads/stores emit: compute effective address → compare against memory size → conditional branch to error path. The error path is inlined (same as the interpreter's `c_trap()` which is `FORCE_INLINE`): store the trap message, load the terminal instruction, and dispatch to the terminal handler.

**Platform abstraction (critical)**: The codegen layer must NEVER contain hardcoded register names (`x20`), struct offsets (`#32`), or instruction encodings. All platform-specific details must be behind abstraction traits:

- **Register references**: `Reg::CTX`, `Reg::PC`, `Reg::FP`, `Reg::L0`, `Reg::T0`, etc. — mapped to physical registers by the platform backend.
- **Context field offsets**: `ctx_offset::MEM0_BASE`, `ctx_offset::TRAP_MESSAGE`, etc. — defined once via `offset_of!`, used by name everywhere.
- **Instruction emission**: `emit.load(dst, base, offset)`, `emit.store(src, base, offset)`, `emit.add_reg(dst, src1, src2)` — the backend translates to ARM64/x86-64/etc.

This separation ensures: (1) the codegen logic is portable across backends, (2) no silent breakage when struct layouts change, (3) future x86-64 support requires only a new backend, not rewriting codegen.

**Context field offset safety**: The JIT hardcodes offsets into the Context struct (mem0_base at 16, mem0_size at 24, trap_message at 32, term_inst at 40). These must be guarded by compile-time assertions (`offset_of!`) so that any struct change causes a build failure rather than silent JIT breakage.

### 5. Coexistence with static fusion

The micro-JIT and static fusion are **alternative fusion strategies**, selected via feature gate:

- `#[cfg(feature = "micro-jit")]` — ARM64 JIT fusion (this document)
- `#[cfg(feature = "fusion")]` — static fusion fallback (for non-JIT platforms)
- Neither — baseline interpreter (1 handler per opcode)

**Feature gate discipline**: The current `feature = "fusion"` has a single central entry point in `on_stream()`. The `micro-jit` feature must follow the same pattern — one (or very few) entry gates that can be disabled cleanly. Do not scatter `#[cfg(feature = "micro-jit")]` throughout the codebase.

Both share the same builder pipeline (Wasm decode → stack tracking → TempInst emission → finalization). The difference is at the emit stage: static fusion matches pre-discovered patterns and assigns pre-compiled C handlers; the micro-JIT groups consecutive opcodes and emits machine code on the fly.

On JIT-capable platforms, the micro-JIT replaces static fusion entirely. There is no "JIT on top of fusion" — the JIT IS the fusion.

### 6. Float operations (deferred)

Float values are stored as bit-punned `uint64_t` in the integer TOS registers. ARM64 float arithmetic instructions (`fadd`, `fsub`, etc.) only operate on FPR registers (s0-s31/d0-d31), requiring explicit GPR↔FPR transfers (`fmov`) around each operation — 4-5 ARM64 instructions per Wasm float op, not 1-2.

Initial scope: **integer operations only**. Float opcodes are group boundaries — they force the JIT group to end, and the float instruction keeps its pre-compiled C handler. Float support can be added later by extending the instruction encoder with FPR transfer patterns.

## Hard Parts and Corner Cases

### TOS window overflow (depth > 4)

The micro-assembler does NOT manage TOS overflow. The builder's `StackTracker` already handles this: before any push that would exceed 4 TOS registers, it emits an explicit `spill_1` instruction; before any consume that needs values from memory, it emits a `fill_N` instruction. These appear as separate instructions in the builder's output stream.

The JIT sees `spill_1` and `fill_N` as regular instructions and emits them as `str`/`ldr` to/from the operand stack. The spill slot address is `fp[operand_base + spill_depth]` where `operand_base = frame_size + FRAME_METADATA_SLOTS (3)`. The slot index is already computed by the builder and available in the instruction's `PatternData::Spill1 { slot }` / `PatternData::Fill1 { slot }`. `frame_size` is per-function and known at JIT emission time; it gets baked into the emitted offset immediates.

Initially, JIT groups containing spill/fill instructions may be excluded as a simplification (the spill/fill keeps its C handler as a 1-instruction non-JIT slot). Once the assembler handles spill/fill as `str`/`ldr`, this restriction is removed.

### Non-hot locals

`local_get(idx)` where idx ∉ {0,1,2} emits `ldr x_tmp, [x22, #(idx*8)]`. This is a real memory access. In principle, if the same non-hot local is read twice in the group without an intervening write, the second load could reuse the register from the first. However, this requires a separate "value-in-register" tracker beyond the TOS alias system — after the first load's TOS slot is consumed by a subsequent operation, the value may no longer be in a known register. This is deferred as a future optimization task, not for initial bring-up. Initially, every non-hot `local_get` emits a load.

### The error path for trapping ops

When a bounds check fails, we inline the trap logic (same as the interpreter's `c_trap()` which is `FORCE_INLINE` — just two memory operations). The JIT emits:

```
movz x2, #msg_lo          ; load trap message address (lower 16 bits)
movk x2, #msg_hi, lsl #16 ; (upper bits — extend to 48/64 as needed)
str  x2, [x20, #32]       ; ctx->trap_message = msg_addr
ldr  x21, [x20, #40]      ; pc = ctx->term_inst
ldr  x2, [x21]            ; handler = term_inst->handler
ldr  x1, [x21, #0x20]     ; nh = (term_inst+1)->handler
br   x2                   ; dispatch to op_term
```

The trap message string address is materialized via `movz`/`movk` (2-4 instructions depending on address width). The `ldr x21` loads directly into the pc register — no intermediate register needed. The JIT group remains a leaf function — no function calls, no register save/restore needed.

### Instruction stream layout

Each JIT group of N Wasm opcodes compacts into 1 Instruction slot — the same N-to-1 compaction that static fusion already does. The existing finalizer pipeline (`compute_keep_mask` → `build_index_map` → `compact_and_patch`) handles branch target index remapping. The JIT group emits a single `TempInst` whose handler pointer is the JIT'd machine code, and the finalizer treats it identically to a static fusion handler.

Unsupported opcodes (calls, br_table, float ops, etc.) produce 1 Instruction slot each with their pre-compiled C handler, unchanged from today. The resulting instruction stream interleaves JIT'd group slots and non-JIT'd instruction slots.

**What goes where**: Wasm operand values (constants, local indices, memory offsets) are baked directly into the machine code by the JIT emitter — they do not consume Instruction slot immediates. The slot's imm0/imm1/imm2 fields are reserved for data that must be patched by the finalizer: branch target `Instruction*` pointers (two-pass patching) and any per-group metadata. This is why a JIT group only needs 1 Instruction slot regardless of how many Wasm operands it contains.

**Imm usage audit** (complete):

*Can be baked into machine code* (all compile-time constants):
- `Const { value }` — `movz`/`movk` or add-immediate
- `LocalGet/Set/Tee { idx }` — immediate offset in `ldr`/`str` from fp
- `Load/Store { offset, memidx }` — address computation immediate
- `Spill1/Fill1 { slot }` — immediate offset in `str`/`ldr` from fp
- All binop/unop — `PatternData::Raw { 0, 0, 0 }`, zero immediates needed; the depth variant determines registers
- `Global { global_idx }` — index for pointer chase

*Must remain in instruction slot imm fields* (runtime addresses patched by finalizer):
- **Branch targets** (`If`/`Br`/`BrIf`/`BrIfSimple`/`Else`) — `Instruction*` pointer in imm0
- **Call entry** (`CallLocal { entry }`) — compiled function entry pointer in imm0
- **`br_table`** inline data — per-entry target offsets

`stack_drop` for `br`/`br_if` is a compile-time constant (stored in imm1) that can be baked into machine code. For `br_if_simple` (the most common loop back-edge), there is no stack_drop — only the target in imm0.

**Conclusion**: For non-branching JIT groups, the instruction slot's imm fields are unused. For groups ending with `br_if`, only imm0 (branch target) is needed. All operand data is in the machine code.

### Builder pipeline and JIT integration point

The micro-JIT does NOT operate on raw Wasm opcodes. It plugs into the existing builder pipeline at the handler-selection stage. The full pipeline:

1. **Wasm decode** → `dispatch.rs::decode_and_dispatch()` reads raw Wasm bytecode
2. **Spill/fill injection** → `StackTracker` inserts explicit `spill_1`/`fill_N` instructions to manage TOS overflow. Before a push with 4 values already in registers → spill. Before a consume with operands in memory → fill. Before calls → spill all.
3. **Depth variant selection** → each opcode gets a D1/D2/D3/D4 variant based on current TOS depth. The variant determines which physical registers hold the operands:
   - D1: result in t0(x26)
   - D2: operands in t1(x27),t0(x26), result in t1(x27)
   - D3: operands in t2(x28),t1(x27), result in t2(x28)
   - D4: operands in t3(x0),t2(x28), result in t3(x0)
4. **[Static fusion]** matches pre-discovered patterns, assigns C handler → TempInst emission
5. **[Micro-JIT]** groups consecutive variant instructions into JIT groups, emits machine code → TempInst emission with JIT handler pointer
6. **Finalization** → compaction, branch target patching, encoding to `Instruction` array

The micro-JIT replaces step 4 with step 5. Steps 1-3 and 6 are shared and unchanged. The JIT's input is the post-variant stream where each instruction has a compile-time-known depth variant and spill/fill instructions are already inserted. The assembler just needs to map variant → physical register and emit the corresponding machine code.

### TOS state at group boundaries

At every group boundary, TOS registers are in **canonical state** because the builder's variant selection ensures it. The JIT doesn't need to track or manage TOS state — it's already resolved. Each instruction in the group has a known variant (D1-D4), and the JIT emits the operation on the corresponding physical registers. Spill/fill instructions within the group are emitted as `str`/`ldr`. The dispatch stub at the group exit doesn't touch registers — the last instruction's output is already in the correct TOS register.

### Memory management

Each JIT group is a small allocation (typically 20-500 bytes). Arena allocator: allocate a large executable region, bump-allocate groups within it. Arena with generation tracking for cleanup.

**no_std consideration**: The current core is `no_std`. `mmap`/`mprotect` are OS syscalls not available in `no_std`. Options: (1) the micro-JIT module uses a thin platform abstraction layer that calls OS APIs directly via `extern "C"` (no std dependency needed — just libc FFI), (2) on bare-metal targets, the allocator takes a caller-provided executable memory region instead of calling `mmap`. This needs design discussion before implementation.

### x86-64 portability

ARM64 only for the initial implementation. x86-64 has variable-length encoding (1-15 bytes per instruction) and a different `preserve_none` register mapping — a future backend.

**Design rule**: The codegen layer (opcode grouping, alias/constant tracking, spill/fill logic) must be **platform-independent**. Only the instruction encoder backend is platform-specific. The architecture:

```
codegen (platform-independent)
  → trait EmitBackend { fn add_reg(...); fn load(...); fn store(...); ... }
    → Arm64Backend (initial)
    → X86_64Backend (future)
```

No ARM64-specific logic in the codegen layer. No hardcoded register names, instruction widths, or encoding details outside the backend. This means the x86-64 port requires only a new `EmitBackend` implementation, not a rewrite of the grouper or optimizer.

### Group boundaries

A JIT group is a maximal sequence of consecutive Wasm opcodes that the JIT can handle. The group ends (and a new group or non-JIT instruction begins) at:

**Mandatory boundaries** (control flow):
- **Branch targets**: any opcode that is the target of a `br`, `br_if`, `br_table`, or `loop` header. Must be reachable as its own handler entry.
- **Unconditional branches**: `br`, `return`, `unreachable` — group terminators.
- **`br_if` / `if_`**: these end the group. The JIT emits the conditional branch as the group's exit.
- **`else`**: changes control flow path; starts a new group.

**Mandatory boundaries** (non-JIT'd opcodes):
- **Call instructions**: `call`, `call_indirect`, `call_ref`, `return_call`, `return_call_indirect`. These involve frame setup, l0/l1/l2 spill/fill, TOS reset, and non-linear dispatch. They keep their pre-compiled C handlers.
- **`br_table`**: multi-way dispatch — complex to emit, keep C handler.
- **Float opcodes** (`f32.*`, `f64.*`): deferred (see §6). Keep C handlers.

**Soft boundaries** (initial simplification):
- **TOS depth > 4**: handled by the StackTracker before the assembler sees it. The StackTracker inserts spill/fill instructions as needed, so the assembler always sees compile-time-known TOS positions (d1/d2/d3/d4). Initially, groups containing spill/fill may be excluded as a simplification; once the assembler handles spill/fill as regular store/load, this boundary disappears.
- **`select`**, **`global_get`/`global_set`**, **`memory_grow`/`memory_size`**: could be JIT'd but are uncommon in hot loops. Initially keep C handlers; add JIT support incrementally.

Opcodes between group boundaries that the JIT handles: `local_get/set/tee` (including lN variants), `i32_const`, `i64_const`, all integer arithmetic/comparison, all integer load/store, `drop`.

A single unsupported opcode between two JIT groups is just one Instruction slot with its C handler — no overhead beyond the normal dispatch.

## What Makes This Different from Existing Work

- **Not copy-and-patch**: we don't copy pre-compiled handler blobs. We emit fresh machine code per group, with knowledge of the specific immediate values and register state. This enables constant folding and alias elimination that copy-and-patch cannot do.
- **Not a baseline JIT (Winch/Liftoff)**: those do single-pass register allocation over the virtual stack. We skip register allocation entirely — the TOS + L0/L1/L2 system provides a fixed, pre-determined mapping.
- **Not wasm2c**: we don't generate C or invoke a compiler. The assembler is embedded in the interpreter binary (~300-500 lines of Rust for ARM64).
- **Not a bolt-on optimization**: the micro-JIT is an alternative fusion strategy, not a layer on top of the interpreter. It uses the same builder pipeline (decode → stack track → emit → finalize) and produces the same instruction stream format. The only difference is what emits the handlers.
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

This is 60-400x smaller than any existing Wasm JIT. On resource-constrained embedded Linux devices with AArch64 cores (Cortex-A35/A53 in IoT gateways), binary size is the primary constraint — these devices have virtual memory and `mmap`, but cannot afford 15+ MB runtimes. On bare-metal RISC-V or Cortex-M targets with executable SRAM, the micro-JIT could work without `mmap` at all (just allocate and write code), though those would need RV32/Thumb-2 backends rather than AArch64. Either way, this could be the first Wasm JIT that fits in resource-constrained environments where existing runtimes cannot.

The reason this is possible: existing JIT compilers need a full compiler framework (IR, optimization passes, register allocator). The micro-JIT needs none of this because the interpreter's TOS + L0/L1/L2 architecture already provides the register assignment. The "compiler" is just a template assembler — the architectural innovation that made the interpreter fast is the same innovation that makes the JIT trivially small.

## Expected Performance

Within each JIT group, all intra-group dispatch overhead is eliminated and constant folding + alias elimination apply across the entire sequence — not just pre-discovered 2-4 instruction patterns. Estimated improvement over current static fusion: **10-25%** on compute-intensive workloads (CoreMark, tight loops). The gain is larger for sequences that static fusion handles poorly (long runs exceeding the 3-imm capacity, patterns not in the static set, workload-specific hot paths).

The ceiling is below Cranelift because we can't optimize across group boundaries (branch targets force a dispatch). But the simplicity is the point — a few hundred lines of Rust vs a full compiler.

## Implementation Plan: Step-by-Step

The guiding principle: **every step is independently verifiable without executing generated code until we've proven the encoding is correct**. A single wrong bit in a u32 causes an undebuggable segfault. We build trust incrementally — encoding correctness first, then trivial execution, then progressively complex handlers.

### Debugging Strategy

Before any step involves execution, we establish two tools. Additionally, once the JIT can handle a bare minimum of opcodes (Step 4+), **spec tests** (`cargo run --bin sf-nano-spectest`) become the primary validation — every subsequent step must pass the full spec test suite. Regressions are caught immediately.

1. **Disassembly comparison**: Every emitted `u32` is disassembled (via `llvm-objdump` or a disassembler crate) and compared against expected text. This catches encoding bugs as text diffs, not segfaults.

2. **Trace buffer**: A memory region where JIT code writes register snapshots (`str x23, [x_trace, #offset]`) at instruction boundaries. After execution, the Rust side diffs the trace against the interpreter's execution of the same block. Any divergence pinpoints exactly which instruction emitted wrong code. No function calls needed (just stores), so no ABI concerns.

### Step 1: ARM64 Instruction Encoder (pure functions, no execution)

Build encoding functions: each takes register operands + immediates, returns a `u32`. Approximately 30 patterns needed: `add/sub/and/orr/eor/mul/sdiv/udiv` (reg and imm forms), `lsl/lsr/asr`, `cmp`, `cset`, `ldr/str` (various widths), `mov/movz/movk`, `b/b.cond/cbz/cbnz/br`, `sxt/uxt`.

**Verification**: Write unit tests that encode each instruction and compare the bytes against known-good output from `as` or `llvm-mc`. Example: `arm64_add_imm(W23, W23, 1)` must produce `0x110006f7`. Feed the expected bytes through a disassembler to triple-check. No mmap, no execution — pure data transformation. This step has zero risk.

### Step 2: Code Buffer (mmap + trivial execution)

Implement the executable memory allocator. Platform-specific approaches:

- **macOS Apple Silicon** (required): `mmap` with `MAP_JIT` flag, then toggle between write and execute modes per-thread via `pthread_jit_write_protect_np(0)` (writable) and `pthread_jit_write_protect_np(1)` (executable). The standard `mmap RW → mprotect RX` path does NOT work on Apple Silicon — the OS enforces the `MAP_JIT` + toggle pattern.
- **Linux**: `mmap` with `PROT_READ|PROT_WRITE`, write instructions, then `mprotect` to `PROT_READ|PROT_EXEC`.

**I-cache invalidation**: ARM64 has separate I-cache and D-cache that are not coherent. After writing code to the buffer and before executing it, the I-cache must be explicitly invalidated over the written range. On macOS: `sys_icache_invalidate(addr, len)`. On Linux: `__builtin___clear_cache(start, end)`. Forgetting this causes stale-instruction execution — correct code that runs wrong, intermittently.

**Verification**: Write the simplest possible function — a single `ret` instruction (or `br x30`). Cast the buffer to a function pointer, call it. If it returns without crashing, the mmap/execution pipeline works. Then test `mov x0, #42; ret` and verify the return value equals 42. This tests the execution infrastructure with code simple enough to verify by inspection.

### Step 3: The Dispatch Stub (preserve_none ABI compatibility)

This is the first step that touches the interpreter's dispatch chain. Emit a **no-op handler**: it does zero work, just dispatches to the next instruction. The emitted code:

```
add  x21, x21, #0x20     ; advance pc by 1 instruction (32 bytes)
ldr  x2, [x21]           ; load handler at new pc
ldr  x1, [x21, #0x20]    ; preload nh
br   x2                   ; tail-jump
```

**Note**: Under the 1-slot compaction model (§Instruction stream layout), each JIT group occupies exactly 1 Instruction slot, so pc always advances by `#0x20`. The dispatch stub above is the universal exit sequence for all groups.

**Verification**: Write a test Wasm module with a known instruction sequence, e.g., `i32_const(1), i32_const(2), i32_add`. Replace the middle instruction's handler with the dispatch stub (the stub skips that instruction). If execution doesn't crash and reaches the expected next handler, the ABI contract is proven — our generated code can enter and exit the dispatch chain correctly. (The result will be wrong because the skipped instruction's work is missing, but the test proves dispatch integrity, not semantic correctness.)

This is the **most critical milestone**. If this works, we've proven: mmap code is callable, preserve_none register layout is correct, pc advancement works, nh preloading works, and tail-jump dispatch is ABI-compatible. Everything after this is incremental.

### Step 4: Single-Instruction Emission (one opcode at a time)

Emit a handler for ONE instruction type. Start with `i32_add` at a specific TOS depth (D2, the most common). The emitted code does the operation on the known registers, then dispatches (reusing the dispatch stub from Step 3).

**Verification**: Write a test module that executes `i32_add`. Replace the interpreter's handler with the JIT-emitted one. Compare output. Then add `i32_sub`, `i32_and`, `i32_or`, `i32_xor`, `i32_mul`, `i32_shl`, `i32_shr_u`, `i32_shr_s` — each is one additional `u32` encoding pattern, independently testable.

**Key constraint**: at this step, each JIT handler covers exactly ONE instruction and dispatches. The emitted code is functionally equivalent to the C handler it replaces — same registers, same semantics, same dispatch. This means the spec test suite can validate correctness for each opcode we add.

### Step 5: Multi-Instruction Groups (no optimization)

Emit code for a JIT group of N consecutive opcodes. No alias tracking, no constant folding — just naive emission: each `local_get_lN` emits a `mov`, each `i32_const` emits a `movz`, each arithmetic op emits the operation on TOS registers. The dispatch stub goes at the end only (not after each instruction). The group compacts N opcodes into 1 Instruction slot.

**Verification**: Start with a 2-instruction group. Then 3. Then N. At each step, compare the module's output against the interpreter. The key invariant: the JIT group produces identical register state at its exit as the interpreter would after executing the same N instructions.

Use the trace buffer here: emit `str` instructions at each internal boundary to record register values. Diff against interpreter execution.

### Step 6: Alias Tracking

Add the `RegState` tracker (Alias/Const/Value/Empty). When `local_get_lN` is encountered, record the alias instead of emitting a mov. When a subsequent instruction reads that TOS slot, resolve the alias to the physical register.

**Verification**: For known instruction sequences, assert on the number of emitted instructions. Example: `local_get_l0 → local_get_l1 → i32_add` should emit 1 instruction (the add), not 3. Disassemble and compare. Then run the module and verify output — fewer instructions, same result.

### Step 7: Constant Folding

When `i32_const(K)` is encountered, record K in the tracker. When consumed by the next instruction, fold K into the immediate field if it fits. The encoding depends on the consuming instruction:

- **add/sub/cmp**: 12-bit unsigned immediate (0-4095), optionally shifted left by 12.
- **and/orr/eor**: ARM64 bitmask immediate encoding — a non-obvious set of representable values (repeating bit patterns). Not all 32-bit values are encodable; the encoder must check and fall back to materialization.
- **Other ops**: no immediate form; always materialize.

If K doesn't fit the consuming instruction's immediate encoding, materialize with `movz`/`movk`.

**Verification**: `i32_const(1) → i32_add` should produce an add-immediate instruction, not a movz + add-register. `i32_const(0xFF) → i32_and` should produce an and-immediate. Check the disassembly. Then verify with constants that don't fit (e.g., 0xDEAD for add, 0x3 for and-bitmask) to test the materialization fallback.

### Step 8: Destination Forwarding

When the instruction after an operation is `local_set_lN`, emit the operation with lN's register as the destination instead of the TOS register.

**Verification**: `local_get_l0 → i32_const(1) → i32_add → local_set_l0` should produce a single `add w23, w23, #1`. Disassemble, verify, run.

### Step 9: Conditional Branch Dispatch (br_if)

`br_if` ends a JIT group with two exit paths: taken (branch to target) and fall-through. The branch target `Instruction*` is stored in the group's Instruction slot (imm0), loaded via `ldr x_target, [x21, #8]`. For compare+branch fusion (`i32_lt_s + br_if` → `cmp + b.lt`), implement as a peephole on the last two opcodes of the group.

**Verification**: Write a test module with a loop (the simplest `br_if` target — back to loop header). Verify it terminates correctly and produces the right output. This is the first test of branching in JIT code.

### Step 10: Trapping Ops (memory access, division)

Emit inline bounds checks for memory access and division-by-zero/overflow guards. On trap, the error path is **inlined** (same approach as §4 and §"The error path for trapping ops"): store trap message address, load term_inst, dispatch. No stub, no function call — the trap logic is ~5 ARM64 instructions emitted inline at each trapping op site.

**The hard part**: loading `ctx_mem0_size(ctx)` requires knowing the offset of the memory size field within the `Context` struct. This creates a coupling between the JIT codegen and the Context layout. If Context changes, the JIT-emitted offsets become wrong — silently. Solution: compile-time assertions via `offset_of!` (see §4).

**Verification**: Test with Wasm modules that perform in-bounds and out-of-bounds memory access. Verify that in-bounds produces correct values and out-of-bounds produces the expected trap.

### Step 11: Full Integration and Spec Tests

Wire the JIT grouper into the builder pipeline as an alternative to static fusion (`#[cfg(feature = "micro-jit")]`). The builder walks Wasm bytecode, groups consecutive supported opcodes (breaking at boundaries listed in §Group Boundaries), emits a JIT handler per group, and emits non-JIT instructions unchanged. The finalizer compacts and patches branch targets as it does today.

**Verification**: Run the full spec test suite with micro-JIT enabled. Any failure indicates a codegen bug — the failing test isolates the problematic opcode/pattern. Then benchmark CoreMark.

### Step 12: Lazy JIT (optional, nice-to-have)

By default, all functions are JIT'd at load time. For modules with many cold functions, this wastes time and executable memory. Lazy JIT defers compilation to first call, using the existing `has_fast_code()` + lazy compile pattern already in the codebase.

**Verification**: Benchmark load time for large modules (e.g., Lua) with eager vs lazy JIT. Verify no functional difference.

### Step Dependencies

```
Step 1 (encoder)
  └→ Step 2 (mmap)
       └→ Step 3 (dispatch stub)         ← CRITICAL MILESTONE
            └→ Step 4 (single instructions)
                 └→ Step 5 (multi-instruction groups)
                      ├→ Step 6 (alias tracking)
                      ├→ Step 7 (constant folding)
                      ├→ Step 8 (destination forwarding)
                      └→ Step 9 (br_if dispatch)
                           └→ Step 10 (trapping ops)
                                └→ Step 11 (full integration)
                                     └→ Step 12 (lazy JIT, optional)
```

Steps 6-8 are independent optimizations that can be added in any order. Each is a diff on the codegen pass, independently testable.
