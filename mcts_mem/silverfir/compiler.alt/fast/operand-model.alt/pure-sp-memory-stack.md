- Each fast-interpreter handler receives a runtime stack pointer (threaded
  through the trampoline register set) and reads operands with pop_u64(sp) and
  writes results with push_u64(sp); the stack pointer is incremented and
  decremented on every push and pop.

- The trampoline threads the value-stack pointer sp alongside ctx, pc, mem0
  base/size, and locals base through every tail-chained handler call.

## Facts

- 2026-06-14 rationale: the pure sp-addressed memory stack is borrowed from
  wasm3 and is a genuine weighed alternative to the TOS operand model — it
  benchmarked well, eliminating many local.get by encoding the slot address
  directly in the operand. It was rejected because it caps fusion: once
  sp-addresses are flattened into the instruction, inlined handlers must each
  read and write the memory slot to preserve full semantics, because the
  redundant side effects can no longer be proven away; the TOS-slot operand
  model instead hands the slots to fused handlers as arguments the C compiler
  sees directly and optimizes the stack reads/writes across the fused chain — so
  the sp-stack's ceiling is lower than the fusion approach's. Backing argument in
  [[fast-interpreter.fact/interpreter-design-paper]] Part III (sourced).

## Moves

- 2025-08-17 (912cc440) replaced [[register-window]]: the by-pointer Regs bank
  with a t0/t1/t2/depth top-of-stack register window and lazy spill/fill is
  dropped in favor of operands living directly on the memory stack addressed by
  an sp pointer, with (sp, mem0 base, mem0 size, locals base) passed as by-value
  scalar arguments threaded through each preserve_none tail call instead of
  carried in a heap-shaped register struct (code).

- 2025-12-04 (9a490383) replaced by [[sto-no-stack]]: the stack-based model
  threaded a runtime stack pointer through every handler and executed an sp
  increment/decrement per push/pop; since a valid wasm function's stack height at
  each instruction is statically known, the IR builder precomputes each
  instruction's stack-top offset (STO) and handlers address operands by absolute
  frame slot index, eliminating the runtime stack pointer and its per-instruction
  maintenance entirely (code).
