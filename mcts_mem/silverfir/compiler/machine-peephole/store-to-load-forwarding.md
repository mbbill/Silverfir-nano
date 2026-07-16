- Store-to-load forwarding tracks each store's width and forwards a stored
  value into a later load only when the load matches the same width (U32->U32,
  U64->U64), never synthesizing a narrower load from a wider store; the U32
  pair an i64 wasm-local spill/fill lowers to on 32-bit GP backends forwards
  (`forward_stored_values`, `is_forwardable_width`).

- Both the store-forwarding and load-reuse trackers invalidate on conservative
  may-alias rules rather than exact address equality: a store with the same
  base register kills entries by precise byte-range overlap, a store through a
  different base register kills every entry unless one side's base is
  runtime-owned (the frame or the runtime context, which a wasm-visible store
  can never write), and stores with unknown target ranges — indexed stores,
  bulk-memory ops, table writes — kill every non-runtime-owned entry
  (`store_may_alias`).

## Facts

- 2026-04-23 (d2a36772) statement: widening store-to-load forwarding to the U32
  pair shape was the change that let the self-reload around gp32 i64
  multiplies actually be eliminated — the precondition the SMULL fusion's
  spill/reload tracking assumed but that the U64-only forwarder could not
  satisfy on 32-bit GP backends, where the i64 spill/fill is a U32 pair (code).

## Moves

- 2026-04-23 (d2a36772) replaced [[u64-only-store-to-load-forwarding]]: the
  forwarder hard-coded `width == U64` for both the tracked store and the
  matching load, so on 32-bit GP backends — where an i64 wasm-local spill/fill
  lowers to a *pair* of U32 store/load ops — it was structurally blind to the
  pair shape and the self-reload ping-pong around gp32 i64 muls survived into
  emitted code; tracking each store's width and forwarding only same-width
  store->load pairs (U32->U32, U64->U64, never synthesizing a narrower load
  from a wider store) lets the U32 pair forward and removes the surviving
  reloads (code).
