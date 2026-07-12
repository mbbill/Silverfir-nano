- The micro-JIT compiles fused basic blocks to ARM64 machine code with a
  template micro-assembler and uses each group's machine-code address as a
  handler pointer in the fast interpreter's dispatch chain (`group`,
  `compile_jit_groups`).

- Generated groups run under the fast interpreter's preserve_none register ABI
  (ctx=x20, pc=x21, fp=x22, l0-l2=x23-x25, t0-t3, nh) and dispatch like
  handlers at group exits; ops the JIT cannot handle keep their pre-compiled C
  handlers in the same chain.

- The JIT consumes the fast interpreter's backend-lowered IR and shares its
  ResolvedInst backend pass and finalizer rather than owning its own code
  representation.

- ARM64 instructions are encoded through pure register-to-u32 functions
  (`arm64_enc`) that name registers by interpreter-ABI role (`Reg` enum) whose
  discriminant is the physical 5-bit register number under preserve_none.

- After decode, consecutive JIT-able TempInsts are scanned into maximal groups,
  each compiled to one ARM64 code block whose address becomes the group's
  handler; non-JIT-able opcodes and group boundaries (calls, br_table, br_if,
  float, insufficient TOS height) keep their C handlers.

- The micro-JIT lowers only an explicitly enumerated subset of IR op kinds
  (arithmetic, comparisons, conversions, consts, hot/frame local get/set/tee,
  loads/stores, spill/fill, drop/select); any other kind falls back to a 1:1
  base handler, memory ops are supported only for memory index 0, and every
  memory op emits a trap stub for the out-of-bounds path (`supports_kind`).

## Facts

- 2026-02-21 (2f5ac953) measurement: the function prologue's three hot-local
  swap+fill ops (init_l0/init_l1/init_l2) each cost a separate interpreter
  dispatch; folding them into one init_locals handler does all three swaps and
  fills in a single dispatch, saving two dispatches per function entry (code).

- 2026-03-03 (e2d2cc24) rationale: the JIT hardcodes the interpreter-ABI to
  physical-ARM64 register mapping into the `Reg` enum, and a build-time probe
  (`verify_abi.c`) compiles a preserve_none function, parses the emitted `str`
  instructions, and aborts the build if any argument's register no longer matches
  the enum (code).

- 2026-03-03 (e2d2cc24) rationale: the build-time ABI probe exists because Clang's
  preserve_none register assignment is an implementation detail that could
  silently change, and silent ABI drift would otherwise produce undebuggable
  miscompiled JIT code (sourced).

- 2026-03-03 (2ff0b000) rationale: Context field offsets baked into JIT code are
  guarded by `offset_of!` compile-time assertions so a struct-layout change fails
  the build rather than silently miscompiling (code).

- 2026-03-03 (2ff0b000) statement: a portability rule was fixed for the codegen
  layer — no hardcoded register names, struct offsets, or instruction encodings
  inline; all behind an EmitBackend trait so a new ISA is a new backend, not a
  codegen rewrite (sourced).

- 2026-03-05 (11d848b4) rationale: float values live bit-punned in the integer
  TOS registers, so each JIT float arithmetic op costs 4-5 ARM64 instructions
  (fmov GPR->FPR on both operands, the FPU op, fmov back) rather than 1-2; this
  is why float ops were initially group boundaries keeping their C handlers
  (code).

- 2026-03-06 (d022a64e) pitfall: zeroing a callee's locals in the local-call C
  handler must be spelled as an explicit 2-at-a-time pair-store loop so AArch64
  clang lowers it to `stp xzr, xzr`; a generic memset-shaped loop is turned
  into a `_bzero` call, forcing a non-leaf stp/ldp x29,x30 prologue/epilogue
  onto the whole handler (code).

- 2026-03-04 (a3a4f422) rationale: JIT memory accesses emit an inline bounds
  check (effective address vs memory size, B.HI to a per-group trap stub) instead
  of calling c_trap(): the trap stub materializes the OOB message, stores it to
  ctx.trap_message, loads ctx.term_inst into pc, and dispatches to the terminal
  handler — keeping the JIT group a leaf function with no calls, since t3=x0 and
  nh=x1 overlap the standard-ABI argument/return registers and any real call
  would clobber them (code).

- 2026-03-04 (c10348b6) rationale: group eligibility is keyed on PatternData —
  only non-fused ops (PatternData::Raw, or Const/LocalGet/LocalSet/LocalTee
  shapes) are JIT-able, because an already-fused TempInst carries specialized
  multi-opcode PatternData the per-opcode JIT emitter cannot reproduce; ops that
  consume TOS values are gated on sufficient pre_height since below that height
  the operands live on the frame, not in TOS registers (code).

## Moves

- 2026-03-07 (bc6c91c8) replaced by [[monolithic-per-arch-backends]]: the
  micro-JIT was embedded inside the handler-threaded preserve_none fast
  interpreter and its generated code retained interpreter-shaped overhead
  (loop-boundary dispatch, repeated memory-metadata loads, hybrid JIT/handler
  transitions), and its dependence on preserve_none could not port to
  RISC-V/ARM32/MCU targets; the native backend instead owns a self-defined VM
  ABI entered through a global-asm trampoline that threads native-entry
  addresses directly, so it no longer behaves as one more kind of
  fast-interpreter handler (code).
