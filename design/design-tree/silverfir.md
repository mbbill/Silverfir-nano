- Silverfir is a WebAssembly interpreter implemented from scratch in Rust,
  with no dependency on an external Wasm engine or parser.

- The codebase is a Cargo workspace: a library crate holds the engine
  (`sf-core`) and a separate binary crate is the command-line loader that
  reads a `.wasm` file and drives the engine.

- A loaded module passes through fixed stages — parse, then validate — before
  it is available to run; parse failures and validation failures both surface
  as the same module-level error type.

- Errors are a single flat enum spanning the whole pipeline (parse through
  runtime), tagged by Wasm's own failure categories (malformed, invalid,
  unlinkable, exhaustion, trap, exit) (`WasmError`).
