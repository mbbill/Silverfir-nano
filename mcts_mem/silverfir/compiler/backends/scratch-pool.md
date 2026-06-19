- Ad hoc temporaries come from a per-backend scratch pool with explicit
  ownership: reserve before use, hold while live, release when dead
  (`scratch_pool`). The default is a scoped RAII guard; a temp that must
  outlive its lexical scope is taken out as an owned reservation that keeps the
  pool slot reserved until the token is dropped (`detach`, `DetachedScratch`).

- The shared pipeline's parallel-move cycle-break also exposes index-based
  alloc/free (`alloc_gp_scratch` / `free_gp_scratch`, FP equivalents) threading
  an explicit scratch_id — that protocol allocates a scratch in shared
  code and frees it across several separate `&mut self` ArchBackend trait calls,
  a lifetime no RAII guard can span.

## Facts

- 2026-04-08 (a1906ede) rationale: x86_64 cannot use the arch-agnostic
  interchangeable pool for its GP scratch because lowering sometimes requires
  exact named registers (RAX:RDX for div/idiv, RCX for variable shifts), so it
  tracks RAX/RCX/RDX as backend-owned via a target-local pool supporting both
  round-robin and named claims and removes those three from the dynamic GP bank
  — the second, ISA-specific form of the same ownership abstraction whose first
  consumer (arm64) uses only the generic interchangeable pool (code).

- 2026-04-10 (2e7114e0) pitfall: arm32 has only two GP scratch registers (R12
  and R14/LR), so one scratch *is* LR; the prelude's saved LR must stay on the
  stack until after the scratch-consuming function-tail copy loop finishes,
  re-materialized only for the final `bx lr`, or a live scratch holding LR
  clobbers the return target (code).

- 2026-04-27 (5ccc08ee) pitfall: the ARM32 preserved-helper-call return
  sequence staged a status code into one scratch and a 64-bit result into two
  more, needing three GP scratches where ARM32 has two, panicking the pool on
  struct.wast; the fix branches on status while C_RET0 still holds it and loads
  GP results directly into their destination registers, needing no third
  scratch (code).

- 2026-03-28 (9169cd4b) pitfall: the shared edge-stub parallel-move logic needs
  each backend to publish the width of every FP destination register so values
  transfer correctly between blocks; the freshly-migrated ARMv7-A backend was
  missing `set_fp_reg_width` after FP-writing instructions, surfacing as a
  'missing float-width tracking' error compiling lua.wasm — a contract a backend
  joining the shared pipeline must satisfy (code).

- 2026-05-24 (30779e5d) pitfall: arm32's fixed helper state save/restore drew its
  two saved registers from the 2-slot rotating scratch pool at both save and
  restore, but a BLX between them advances the pool cursor by one, so restore
  handed the slots back in swapped order (R12 <-> R14/LR) and the function
  returned through `bx lr` into garbage; the save/restore must name the physical
  registers (R12 and R14) directly rather than re-draw from pool rotation (code).

## Moves

- 2026-04-01 (db81af27) replaced [[release-based-scratch-guard]]: release()
  freed the pool slot immediately while the caller kept using the register, so
  a later scoped_alloc could hand the same physical register out again;
  detach() returns an owned DetachedScratch token that keeps the slot reserved
  with RAII until dropped, so a temp that must survive later &mut self emission
  calls stays protected (code).
