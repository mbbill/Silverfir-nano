- Silverfir is a WebAssembly interpreter written in Rust, built from scratch
  rather than wrapping an existing Wasm engine.

- The crate layout splits a reusable library (`sf-core`) from a thin
  command-line front-end (`sf_loader`) that reads a `.wasm` file and hands its
  bytes to the library.

- Loading a module parses the binary and validates it; an invalid or malformed
  binary is rejected before the module is usable, surfacing as a single error
  type (`WasmError`) whose variants name the failure class (malformed, invalid,
  unlinkable, exhaustion, trap, exit).

- The library borrows the caller's module bytes wherever it can and copies only
  when it must, so a module can be backed by either borrowed or owned memory.

## Facts

- 2024-01-22 (e4a20f95) statement: the project describes itself as "a fast
  WebAssembly interpreter written in Rust" — speed is a stated goal from the
  first commit (diff).
