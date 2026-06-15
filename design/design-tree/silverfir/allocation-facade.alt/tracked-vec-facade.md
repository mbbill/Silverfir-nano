- The tracked-allocation facade (`sf_nano_tracked_vec`) wraps only `Vec`: it
  exports a tracked `Vec`, the `vec!` macro, and the snapshot/reset hooks, while
  engine code reaches Box, String, Rc, BTreeMap, and `format!` through `alloc::*`
  directly, outside the facade.

## Moves

- 2026-04-10 (d08eb19d) replaced [[direct-alloc]]: scattered alloc::Vec gives no
  single place to instrument, account for, or re-back the engine's heap traffic;
  routing every Vec through one collections facade (initially the
  sf-nano-tracked-vec crate, a zero-cost alloc::Vec alias unless its tracking
  feature is on) makes the engine's collection allocations centrally observable
  and replaceable (diff).

- 2026-04-11 (4a9c0f17) replaced by [[allocation-facade]]: the tracking facade
  only wrapped Vec, so every other heap allocation in the engine (Box, String,
  Rc, BTreeMap, format!) still went through alloc directly and was invisible to
  allocation tracking and uncontrolled by the facade; broadening it into
  tracked_alloc that re-exports boxed/string/rc/collections submodules and a
  format! macro lets all engine heap traffic route through one facade so tracking
  can see every allocation and the engine has a single allocation seam (diff).
