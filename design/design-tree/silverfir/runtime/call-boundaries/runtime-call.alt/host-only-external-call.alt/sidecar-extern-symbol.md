- External Wasm calls lower through a sidecar that interns a closed
  MachineHelperSymbol (CallExternal / CallIndirectExternal) into a per-module
  extern-binding table resolved during finalization, with direct and indirect
  external calls carrying distinct metadata records.

- The MachineIR external-call instruction is a generic helper call (CallHelper /
  MachineHelperCall) naming an opaque extern target id plus a sidecar metadata
  constant.

- The machine module carries an explicit externs list (MachineExternBinding) as a
  separate allocation domain alongside its constant records.

## Moves

- 2026-04-01 (2a753247) replaced by [[host-only-external-call]]: after
  memory/table ops left the helper path the closed helper-symbol enum and
  extern-binding
  indirection held only two call symbols and became dead weight; a plain machine
  constant pool subsumes the metadata and one ExternalCallMeta with an
  Immediate/FrameSlot target kind unifies the direct and indirect external-call
  paths into one runtime entry (diff).
