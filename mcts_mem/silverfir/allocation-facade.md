- All engine heap allocation — Vec, Box, String, Rc, BTreeMap, and `format!` —
  routes through one `tracked_alloc` facade rather than `alloc::*` directly,
  giving every allocation a single centrally observable, re-backable seam
  (`tracked_alloc`, `collections`).

## Moves

- 2026-04-11 (4a9c0f17) replaced [[tracked-vec-facade]]: the tracking facade only
  wrapped Vec, so every other heap allocation in the engine (Box, String, Rc,
  BTreeMap, format!) still went through alloc directly and was invisible to
  allocation tracking and uncontrolled by the facade; broadening it into
  tracked_alloc that re-exports boxed/string/rc/collections submodules and a
  format! macro lets all engine heap traffic route through one facade so tracking
  can see every allocation and the engine has a single allocation seam (code).
