- WASI is delegated to wasmtime-wasi (p2 `WasiCtx`) behind a `WasiBridge`,
  exposed to guests through generated external-function adapters registered
  under `wasi_snapshot_preview1`.

- Type-safe WASI bindings are generated at build time from the official
  preview1 WITX via wiggle's `from_witx!` macro, against vendored `.witx`
  definitions.

- A `SilverfirMemory` adapter bridges the engine's `Rc<RefCell<Vec<u8>>>`
  linear memory to wiggle's `GuestMemory` trait, letting generated bindings reach
  guest memory.

## Facts

- 2025-06-24 (d6614342) statement: the wasmtime-wasi adapter shipped as a
  scaffold — every bridged WASI function only logged a warning and returned
  zero/default result words; the wasmtime `WasiCtx` was constructed but never
  invoked, so no real syscall executed through this path (diff).

- 2025-06-25 (01ea170e) statement: the era then tried to generate type-safe
  bindings from the official WITX via wiggle's `from_witx!` macro (adding
  wiggle and witx deps and vendoring the preview0/preview1 `.witx` files), with
  a `SilverfirMemory` adapter bridging the engine's `Rc<RefCell<Vec<u8>>>`
  memory to wiggle's `GuestMemory` trait; the binding bodies were never wired
  to real implementations (diff).

## Moves

- 2025-06-24 (d6614342) replaced [[hand-rolled-native-preview1]]: delegate WASI
  to the mature wasmtime-wasi implementation instead of maintaining a
  hand-rolled one (diff).

- 2025-08-07 (7d16c093) replaced by [[wasi]]: wasmtime-wasi/wiggle
  proved too complicated and heavy to carry — the attempt left wiggle's
  GuestMemory unimplemented and every generated binding body a todo!() — and nano
  does not need that complexity, so the external wasmtime-wasi/wiggle/witx deps
  were removed for the lean hand-rolled in-tree preview1 implementation (author).
