- Engine-internal boundary operations (memory.grow/fill/copy/init, data.drop,
  table.grow/fill/copy/init, elem.drop) are a distinct SSA-IR instruction kind,
  SsaBoundaryOp, flagged by is_boundary_primitive and lowered through a
  specialized native path rather than as ordinary leaf MachineIR ops.

- Each boundary op is lowered to a MachineIR CallHelper carrying a per-op extern
  helper symbol (MachineHelperSymbol::MemoryGrow, ...) and a per-op metadata
  sidecar struct (MemoryGrowMeta, MemoryFillMeta, ...) placed in the
  machine-module const pool.

- Operands and results cross the helper boundary through frame-slot regions
  described by the metadata; the Rust helper reads and writes them with
  region_read/region_write against the frame pointer.

- All cached locals dirty at the boundary are published to frame slots, mem0
  cache regs are reloaded, and cached locals are reloaded after the helper call.

## Moves

- 2026-03-13 (013fd297) replaced [[ssa-operand-lir-vocab]]: native lowering must
  not reconstruct stack or frame publication on its own, so every boundary op must
  already carry only canonical frame spans with all live SSA published to slots
  before it, rather than SSA operands (sourced).

- 2026-03-30 (f9348326) replaced by [[preserved]]: engine-internal helper-backed
  operations move from a per-op extern symbol plus a frame-slot metadata sidecar
  dispatched as a MachineIR CallHelper into first-class MachineIR ops the backend
  lowers through one unified preserved-helper entry (fn(ctx, op_code, io) -> u32)
  with a fixed native-stack I/O layout, owned by the native backends rather than by
  MachineIR (code).
