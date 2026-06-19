- The single IR layer after shared CFG + SSA LIR is a target-independent NativeIR
  that owns placed VM ABI locations, backend-owned spill slots, explicit edge
  copies, and native-owned call/helper lowering shape.

- Every IR operand is a NativePlace that is one of Location(NativeLocation),
  Frame(FrameSlot), or Spill(NativeSpillSlot), where NativeLocation is one of Ctx,
  Fp, Hot(reg), Tos(lane), or Tmp(reg) — VM register kinds and LIR slot types appear
  directly as first-class IR storage.

- IR validation rejects a Location operand whose register class has zero budget
  (Ctx/Fp/Hot/Tos/Tmp each checked against its configured count).

## Moves

- 2026-03-11 (0282f727) replaced by [[machine-ir]]: the old NativeIR carried VM register kinds (Ctx/Fp/Hot/Tos/Tmp) and LIR planning-provenance storage (Frame/Spill slots) directly in its operands, leaking VM meaning and lowering history past the backend boundary; the new IR uses generic MachineReg(u16) plus explicit MachineAddr and moves all runtime layout, pinned-input meaning, and call-link contract into separate ABI/contract metadata, so the ISA backend sees a real machine IR with no context, hot-local, TOS-lane, frame, or spill concepts (code).
