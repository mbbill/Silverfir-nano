- Engine code allocates growable buffers with `alloc::vec::Vec` and the
  `alloc::vec!` macro directly at each site; there is no single facade through
  which collection allocation is routed.

## Moves

- 2026-04-10 (d08eb19d) replaced by [[tracked-vec-facade]]: scattered alloc::Vec
  gives no single place to instrument, account for, or re-back the engine's heap
  traffic; routing every Vec through one collections facade (initially the
  sf-nano-tracked-vec crate, a zero-cost alloc::Vec alias unless its tracking
  feature is on) makes the engine's collection allocations centrally observable
  and replaceable (code).
