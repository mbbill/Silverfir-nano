- Each function is a flat array of fixed-size instructions; a non-branch op's
  successor is the contiguously-laid-out next instruction (pc+1), only a
  control-flow op stores an explicit branch target in the instruction word, and
  the original bytecode is never re-decoded at run time.

- A br_table's entries (target offsets, stack drops, arities) are stored inline as
  data pseudo-instructions immediately after the br_table; all per-function data
  lives in the single instruction array with no separate side blob or blob-base
  pointer.

- The handler chain uses the preserve_none calling convention (no callee-saved
  registers): the threaded state stays in registers across each musttail hop and
  no handler spills/reloads callee-saved registers between dispatches.

- The memory-0 base pointer and size are read from a C-visible hot prefix of the
  per-run context, not threaded as handler parameters, and do not widen the
  fixed dispatch calling convention.

- Each memory load/store opcode has two handlers chosen at build time by the
  static memory index: index 0 uses the C raw-pointer fast path, any other index
  uses a Rust slow-path handler that resolves the Nth memory instance through the
  store; the hot single-memory path keeps the raw-pointer access with no
  per-access memory-index test.

- Each handler receives the already-loaded function pointer for the next
  instruction's handler and preloads the one a step further ahead before
  tail-calling; on the linear fall-through path the preloaded pointer is used
  directly (hiding the load latency), while a branch/trap that redirects reloads
  the target's handler.

## Facts

- 2026-02-05 (041adb16) rationale: handlers split into two generated dispatch
  tails — the guard-check default preloads the next-next handler then branches on
  whether the next pc is the linear successor, using the preloaded pointer on the
  linear path and reloading on the redirected path (for always-linear handlers the
  compiler proves the guard true and removes it), while nonlinear handlers (br,
  br_table, else, call_local, return, term, data) never fall through and always
  reload (diff).

- 2026-02-06 (33291768) rationale: the guard-check branch that selects the
  preloaded next handler is emitted only on non-Windows targets — on Windows x86-64
  the generator still preloads but dispatches straight through with no guard,
  because there the CPU's indirect-branch predictor already handles the common
  linear case well and the guard is net overhead; the preload-and-guard win is
  specific to in-order-ish/ARM dispatch where the dependent handler-pointer load and
  misprediction dominate (diff).

- 2025-08-13 (2b4d0f10) pitfall: the op_* wrapper's tail call into the next handler
  must be unconditional for the musttail attribute to apply, so a guard branch
  (if next==NULL return) before it cannot coexist with musttail; the chain's
  terminus cannot be expressed as a conditional null-return in the wrapper and must
  be carried by a sentinel/exit handler instead (diff).

- 2025-08-17 (c68f672b) rationale: every op handler carries
  __attribute__((preserve_none)) — the chain uses a calling convention with no
  callee-saved registers, so the threaded state stays in registers across each
  musttail hop and no handler spills/reloads callee-saved registers between
  dispatches (diff).

- 2025-08-13 (d88a8edb) rationale: because each instruction threads to its
  successors by raw pointer, the builder emits in two phases — first a vector of
  temporary instructions referencing fallthrough and branch targets by index, then
  a pointer-stable final array into which the index references are patched as
  pointers — because pointers can only be taken once the backing storage will not
  move (diff).

- 2025-08-15 (7a4b152f) pitfall: the per-instruction successor pointers are absolute
  pointers into the instruction array, so they must be patched only after the array
  reaches its final heap allocation — the old code patched them on the arena slice
  and then called into_boxed_slice(), which reallocates and leaves every successor
  pointer dangling; patching now runs against the boxed allocation's base (diff).

- 2025-08-13 (b647883a) rationale: branch instructions are patched from the
  validator's JumpTable — each br/br_if carries a (stack_offset, arity) operand-drop
  fixup mirroring the in-place interpreter's stack-unwind semantics, and br_table
  stores its per-instruction entries with the same fixup, so the fast stream reuses
  the validator's already-computed jump table instead of recomputing block exits
  (diff).

- 2025-08-16 (75b3519a) pitfall: the dynamic memory address index popped from the
  stack must be truncated to its low 32 bits before widening to usize (a wasm i32
  memory index, not a host-width value), and the static offset folded in with
  checked_add that traps on overflow rather than wrapping_add — a wrapping effective
  address could wrap past heap_limit and pass the bounds check, so out-of-range
  index+offset takes the trap edge (diff).

- 2026-02-06 (04776e14) rationale: a specialized br_if_simple handler covers the
  common conditional branch that needs no branch fixup (arity 0 and stack_offset 0),
  encoding only the 64-bit target and skipping the operand-base/arity/height fields
  and result-relocation work of the general br_if; the builder emits it when those
  conditions hold and falls back otherwise, shrinking the hot path of the most
  misprediction-prone fast-backend opcode (diff).

- 2025-08-16 (5b54635f) pitfall: a taken branch's stack fixup must always discard
  the stack_offset intermediate slots beneath the carried results, even when the
  target block has no result type (arity==0) — an early-return guard that skipped
  the whole fixup whenever arity==0 left the stack_offset drop undone for value-less
  branches that still carry a stack-height adjustment (reachable: a branch out of a
  result-less block that drops live operands beneath); removing the guard restores
  the mandated drop (diff).

- 2025-10-01 (b7d4218c) rationale: the memarg memory index is encoded
  backward-compatibly — the leading LEB value is read as an align-flag, and only if
  it is >= 64 is a memory index decoded after it (real alignment being flag-64); a
  flag < 64 is the legacy Wasm-2.0 form with implicit memory 0, and both backends
  keep memory 0 on a fast path using the cached mem0 base/length pointers, falling
  back to a store lookup by index only when the index is non-zero (diff).

- 2025-12-10 (b80afcc9) rationale: multi-memory was handled by splitting at compile
  time rather than branching at run time — before this the C fast-path memory
  handlers trapped on a non-zero memory index (single-memory only); now the builder
  picks the C handler for the common index-0 case and a Rust store-routed handler
  otherwise, so the hot single-memory path keeps the raw-pointer access with no
  per-access index test (diff).

- 2025-10-05 (9b25df61) pitfall: the fast backend packs a memarg offset into a
  32-bit immediate slot, truncating to the low 32 bits — correct for 32-bit
  memories but unable to represent a memory64 offset above 2^32, an acknowledged gap
  left for later when the offset field widens to u64 (diff).

- 2025-12-10 (971a278c) rationale: the finalizer drops drop instructions from the
  emitted stream alongside nop/block/loop/end markers — in the slot-based model a
  wasm drop has no runtime effect (a produced value simply has no consumer
  referencing its slot), so drop carries no work and is removed at finalize rather
  than dispatched as a no-op handler (diff).

- 2025-08-19 (5dcf7436) rationale: after the explicit fallthrough pointer was
  removed the instruction header shrank to 24 bytes and is padded back to a 32-byte
  (power-of-two) size with an unused field so each header is naturally aligned for
  the instruction cache, even though only handler/alt/imm0/imm1 are live (diff).

- 2025-08-14 (63d0d66b) rationale: linear-memory accesses bounds-check the
  effective address against a heap_base/heap_limit pair cached in the hot register
  window (refreshed on entry and after any call), and the static access offset is
  folded into the instruction's immediate at build time so each access does one add
  plus one length check rather than re-reading the memory instance (diff).

## Moves

- 2025-08-17 (ce4fd170) replaced [[explicit-fallthrough-pointer]]: each
  non-terminal op's fallthrough is the contiguously laid-out next instruction, so
  storing an explicit fallthrough pointer was redundant; fallthrough becomes pc+1
  and only the alternate control-flow target (alt) stays in the header, dropping a
  per-instruction pointer field and its load on every non-branch op (diff).

- 2025-12-10 (df2532aa) replaced [[out-of-band-br-table-blob]]: the separate
  br_table blob required its own heap allocation plus a fast_blob_base pointer
  threaded through the Context and every CallFrame; storing each table's entries
  inline as data pseudo-instructions right after its br_table keeps all per-function
  data in the single instruction array, eliminating the blob allocation and the
  blob-base pointer entirely (diff).

- 2026-02-05 (91b0de39) replaced [[mem0-in-handler-params]]: the memory-0 base
  pointer and size were passed by value as two extra arguments in every handler's
  preserve_none signature (and dereferenced from double-pointers in the C handlers),
  widening the fixed dispatch ABI for values that change only on memory.grow; moving
  them into the C-visible CtxHot hot prefix of Context lets handlers read
  ctx_mem0_base/ctx_mem0_size directly and drops the two parameters from the calling
  convention, freeing argument registers in the hot dispatch path (diff).
