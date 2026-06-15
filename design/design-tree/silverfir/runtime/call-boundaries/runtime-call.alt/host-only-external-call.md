- The external-call runtime entry (ExternalCallEntry: extern "C" fn(ctx, frame,
  metadata) -> u32) dispatches calls to host functions only: its FrameSlot target
  kind reads a u32 function index from a frame slot, and call_external_by_index
  rejects any FunctionInst::Local target as an internal error.

- Both Immediate and FrameSlot target kinds resolve to a plain function index;
  there is no encoding for dispatching a call through a function-reference handle.

## Moves

- 2026-04-01 (2a753247) replaced [[sidecar-extern-symbol]]: after memory/table
  ops left the helper path the closed helper-symbol enum and extern-binding
  indirection held only two call symbols and became dead weight; a plain machine
  constant pool subsumes the metadata and one ExternalCallMeta with an
  Immediate/FrameSlot target kind unifies the direct and indirect external-call
  paths into one runtime entry (diff).

- 2026-04-16 (9ff58dcd) replaced by [[runtime-call]]: the external boundary only
  dispatched to host functions by func index and could not express call_ref — it
  had no way to resolve a function-reference handle, type-check the target against
  an expected type, or dispatch a local callee — so the boundary is re-cut into a
  runtime-call entry whose FrameSlot kind carries a RefHandle resolved through the
  store's function-entry registry with call_ref type-checking (diff).
