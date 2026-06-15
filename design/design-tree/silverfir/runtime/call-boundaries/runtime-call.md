- Wasm call sites that cannot transfer directly into another compiled MachineIR
  frame — host/WASI targets, `call_ref`, and runtime-resolved local callees — are
  lowered as an inline MachineIR runtime call dispatched through one runtime-call
  entry (`runtime_call_entry`, `RuntimeCallMeta`), using frame slots as argument
  and result transport.

- The entry's FrameSlot target kind carries a function-reference handle resolved
  through the store's function-entry registry, with `call_ref` type-checking
  against an expected type; it also dispatches host functions and local callees by
  index (`call_runtime_by_handle`, `call_runtime_by_local_index`).

## Facts

- 2026-06-14 rationale: the multi-generation reshaping of the import/external-call
  boundary (sidecar extern symbols -> host-only-external-call -> this runtime-call
  entry, and the parallel import/local entity reshaping) was driven by correctness
  — wasm imports and the import-vs-local entity split get tricky fast, and each
  re-cut closed a real expressivity gap (resolving a function-reference handle,
  type-checking call_ref, dispatching a local callee) — not by exploratory churn;
  the generational chain is legitimate and should not be collapsed as settling
  (author).

## Moves

- 2026-04-16 (9ff58dcd) replaced [[host-only-external-call]]: the external
  boundary only dispatched to host functions by func index and could not express
  call_ref — it had no way to resolve a function-reference handle, type-check the
  target against an expected type, or dispatch a local callee — so the boundary is
  re-cut into a runtime-call entry whose FrameSlot kind carries a RefHandle
  resolved through the store's function-entry registry with call_ref type-checking
  (diff).
