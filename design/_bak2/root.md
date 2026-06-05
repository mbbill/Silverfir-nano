- A fast WebAssembly runtime, written in Rust.
- Speed is a primary design property, not an incidental one.
- Small binary size is a primary design property: the core crate is
  `#![no_std]` + alloc with zero runtime dependencies.
- Execution is JIT-only: every function compiles to native code before it
  runs; no interpreter exists.
