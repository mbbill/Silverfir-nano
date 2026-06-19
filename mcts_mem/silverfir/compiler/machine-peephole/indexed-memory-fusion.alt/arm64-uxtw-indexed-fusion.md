- The ARM64 backend's own peephole recognizes the address-computation-plus-
  load/store sequence (base+index, and the zero-extended UXTW variant with an
  optional absorbed Wasm offset) and rewrites it into a backend-private
  IndexedMemFusion::Load/Store value during code emission.

- The fused form is consumed only by ARM64 codegen and never appears in shared
  MachineIR.

## Facts

- 2026-03-15 (688918a7) rationale: the clean effective-address shape the
  lowering now produces (zext(addr) without the access-width folded in) is
  exactly the pattern UXTW addressing needs — feeding the original 32-bit
  wasm-address register to the load with UXTW lets the load do the
  zero-extension for free, so the explicit I64ExtendI32U disappears whenever no
  intervening offset add has rewritten the extended register (code).

- 2026-03-23 (d3dd880c) rationale: UXTW index fusion absorbs a trailing
  positive-i32 offset add into the fused access, emitting `add Xtmp, Xn, Wm,
  UXTW; ldr/str Rt, [Xtmp, #off]` (2 instructions) instead of bailing to the
  4-instruction unfused sequence; offsets above 0x7FFFFFFF are left unfused
  because they do not fit the ARM64 immediate-offset load encoding and are rare
  enough not to special-case (code).

## Moves

- 2026-03-23 (0a30b592) replaced by [[indexed-memory-fusion]]: a backend-private
  fusion form could only ever serve ARM64; lifting the fusion into the shared
  peephole as portable IndexedLoad/IndexedStore MachineIR ops makes one
  address-mode contract every backend maps to its best addressing mode (code)
