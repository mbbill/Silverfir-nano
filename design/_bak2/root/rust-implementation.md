- The runtime is implemented in Rust.
- Ownership keeps the heavily cross-referenced runtime structures sound by
  construction: no dangling objects, leaks, use-after-free, reentrance hazards.
- Runtime structures are owned, with plain borrows: no `Rc`, no `RefCell` —
  the single-module model removes any need for shared ownership.
