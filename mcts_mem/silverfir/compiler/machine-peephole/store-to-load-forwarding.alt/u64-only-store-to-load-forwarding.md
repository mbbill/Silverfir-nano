- Store-to-load forwarding tracks and forwards only exact `U64` store/load
  pairs: the tracker pushes a store solely when its width is `U64`, matches a
  later load solely when its width is `U64`, and treats every tracked store as
  `U64`-wide for overlap invalidation; narrower (`U32`) store/load pairs are
  never forwarded.

## Moves

- 2026-04-23 (d2a36772) replaced by [[store-to-load-forwarding]]: the forwarder
  hard-coded `width == U64` for both the tracked store and the matching load,
  so on 32-bit GP backends — where an i64 wasm-local spill/fill lowers to a
  *pair* of U32 store/load ops — it was structurally blind to the pair shape
  and the self-reload ping-pong around gp32 i64 muls survived into emitted code;
  tracking each store's width and forwarding only same-width store->load pairs
  (U32->U32, U64->U64, never synthesizing a narrower load from a wider store)
  lets the U32 pair forward and removes the surviving reloads (code).
