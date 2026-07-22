- Engine-internal boundary ops (memory.grow/fill/copy/init, data.drop,
  table.grow/fill/copy/init, elem.drop, plus reference and struct ops) are
  first-class MachineIR instructions that the native backend lowers into a
  preserved-helper call: semantic runtime helpers save the full dynamic
  caller-clobbered JIT set, write operands into a fixed native-stack I/O area,
  call one unified C-ABI dispatch `fn(ctx, op_code, io) -> u32` with a
  standardized slot layout (IMM0/IMM1/ARG0..ARG2/RET0), read the result back,
  and restore the saved registers (`preserved_op`, `preserved_io`).

- ARM64 has a separate narrow path for already-bounds-checked, infallible mem0
  `memory.fill`/`memory.copy`: it calls libc `memset`/`memmove` directly and may
  save only values proven live across that raw call. This exception does not
  weaken the full-save semantic-helper ABI (`refresh_infallible_helper_save_sets`).

- The preserved-helper path is owned by the native backends, not by MachineIR:
  each backend emits its own save/dispatch/restore sequence; the op is never a
  MachineIR call form.

## Facts

- 2026-07-22 pitfall: deriving the save set of every ARM64 preserved helper
  solely from ordinary MachineIR liveness passed emulator/x64 coverage but
  failed four of eight native array tests and produced 22 native GC/ref/table
  spectest errors. MachineIR's normal live-after inventory is not the complete
  semantic-helper ABI inventory. Restoring all dynamic caller-clobbered lanes
  for semantic helpers fixed every array test and the full native release
  spectest; liveness-only preservation remains valid only for the raw infallible
  libc bulk calls (sourced).

- 2026-03-13 (1f0c55d9) pitfall: the 32-bit memory.grow path decoded the
  delta-page count as a signed i32 and rejected any value with the high bit set as
  a negative grow (error sentinel); the spec value is an unsigned u32, so a request
  like 0x8000_0000 pages was wrongly refused — the decode must zero-extend the
  u32, not sign-extend it (code).

- 2026-03-31 (1f59fb0d) statement: when the remaining boundary ops were converted
  to first-class MachineIR (MemoryFill/Copy/Init, DataDrop, TableGrow/Fill/Copy/
  Init, ElemDrop) only the ARM64 backend implemented their preserved-helper
  lowering; at that commit the x86_64 dispatch had no arm for them and the emulator
  returned unimplemented, so the portable boundary ops existed ahead of three of
  their four backends — a labeled mandel-regression mid-refactor state, not the
  settled design (code).

- 2026-03-31 (1f59fb0d) statement: the trap-raising helper called from generated
  code (raise_trap) moved out of per-backend arch/arm64/helpers.rs into shared
  runtime/helpers.rs as a single C-ABI entry mapping a numeric trap kind to a
  WasmError on the context; unlike the boundary ops it was not folded into the
  unified preserved-helper op-code dispatch and keeps its own dedicated extern
  signature (code).

- 2026-04-11 (1a6e7864) pitfall: arm32 set up the three i64-pair-shift helper
  arguments as two sequenced moves — `mov R2, rhs` then
  `emit_pair_args_to_r0_r1(lhs_lo, lhs_hi)` — which silently corrupts whenever the
  allocator mapped lhs_lo/lhs_hi to physical R2: the first move clobbered that
  input before the pair-args step could read it, so the i64 shl/shr/rotl/rotr
  cross-lane carry word came out wrong (hot trigger: soft-float quad shifts in
  mandelbrot/c-ray). The durable rule is that helper-call arg setup must stage all
  args atomically (emit_values_to_regs_via_stack) rather than sequencing
  mov-into-Rn then pair-args-to-others (code).

- 2026-04-17 (9bcf20b4) pitfall: the arm64 preserved-helper call originally
  stashed the helper status in a scratch register, restored the full GP
  caller-saved save set, then branched on nonzero status; restoring that save set
  could clobber x0/x1 (C_RET0), silently dropping a trap raised by the helper after
  the call returned. The fix branches on C_RET0 immediately after the BLR (before
  any restore) to the function's body-local error tail, keeping C_RET0 intact
  (code).

- 2026-04-19 (125fe4cf) pitfall: AAPCS64 preserves only the low 64 bits of V8-V15
  across a C call — scalar arm64 builds keep FP values in the low D view and omit
  V8-V15 from the preserved caller-saved set, relying on the ABI; SIMD arm64
  builds carry full v128 values in that bank, so the preserved-helper wrapper must
  add V8-V15 to the explicitly saved set and spill whole 16-byte Q registers (not
  the 8-byte D half) or the upper halves are lost across a helper call (code).

## Moves

- 2026-07-22 replaced [[liveness-only-helper-save-set]]:
  ordinary MIR live-after data omits dynamic lanes required by semantic runtime
  helpers; full preservation fixes native GC/reference/table behavior while
  liveness-only remains scoped to raw infallible libc memory calls (code).

- 2026-03-30 (f9348326) replaced [[helper-backed-boundary]]: engine-internal
  helper-backed operations move from a per-op extern symbol plus a frame-slot
  metadata sidecar dispatched as a MachineIR CallHelper into first-class MachineIR
  ops the backend lowers through one unified preserved-helper entry (fn(ctx,
  op_code, io) -> u32) with a fixed native-stack I/O layout, owned by the native
  backends rather than by MachineIR (code).
