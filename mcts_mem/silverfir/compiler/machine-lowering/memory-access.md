- Memory and table accesses are lowered inline as explicit machine code: a computed
  effective address is bounds-checked against the cached length and the straight-line
  out-of-bounds guard is an inline trap instruction (the backend lowers it to a shared
  cold trap stub); only growth and bulk/segment ops go through helpers
  (`lower_memory_load`).

- A load or store to memory index 0 lowers against the pinned mem0_base/mem0_size
  registers directly, while accesses to other memories first load that memory's view
  base from the runtime context (`lower_mem0_load_continuation`).

- The bounds-check address arithmetic adds the static offset and the access width to
  the zero-extended 32-bit Wasm address as two separate adds; when a free transient is
  available the access-width add targets a distinct check register (clean effective
  address, residual 0), and only under register pressure does it fold into the address
  and return a residual the continuation subtracts (`lower_memory_load`).

## Facts

- 2026-03-13 (3875be1b) pitfall: computing the effective address by adding the static
  offset to the dynamic base in 32-bit arithmetic before zero-extending wraps mod 2^32
  for large dynamic bases (base+offset near u32::MAX yields a small effective address
  that passes the bounds check), defeating the inline check; the offset must be added
  in 64-bit after the base is zero-extended to I64 (code).

- 2026-03-14 (0736b065) pitfall: when a bounds-checked access splits into a
  continuation block and a trap block, the continuation was entered with empty params,
  dropping any transient value live across the split; the lowerer must thread the
  still-live transients (and those the continuation consumes) as continuation block
  parameters with matching edge args (code).

- 2026-04-08 (a1906ede) pitfall: Wasm table indices are i32 even on 64-bit hosts, but
  indirect-call lowering reloaded the index at the canonical GpWord width with no
  extension, so a stale high half in the published carrier perturbed the table bounds
  check and dispatch arithmetic; the fix reloads with U32 + ZeroExtend (and stores it
  back as U32), and being shared lowering every backend inherits it (code).

- 2026-03-15 (6346056c) rationale: the bounds check is made register-pressure-aware so
  the access continuation can reuse the effective address directly — when a free
  transient is available the access-width is added into a separate check register
  (leaving the clean zero-extended effective address, residual 0), and only under
  register pressure does it fall back to folding access_bytes into the address and
  returning that as a residual the continuation subtracts back, avoiding an
  add-then-subtract round-trip in the common case (code).

- 2026-03-27 (dcf7d2f8) pitfall: the bounds-check scratch register borrowed for
  check = addr32 + access_bytes must be filtered to differ from addr32 — when addr32
  came from a dead value it is back in the free transient pool and can be re-handed as
  the scratch, silently corrupting the address so the store/load hits
  mem[addr+access_bytes]; if no other free transient exists, fall through to the
  in-place path that reports the residual for later subtraction (code).

## Moves

- 2026-03-14 (5f7b0f37) replaced [[split-cfg-bounds-check]]: a straight-line out-of-bounds guard whose only cold behavior is to trap does not need a continuation split and a dedicated trap block; an inline TrapIf preserves explicit trap semantics while the backend lowers the guard to a shared cold trap stub (code).
