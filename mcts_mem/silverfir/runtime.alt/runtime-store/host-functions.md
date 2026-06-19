- A host-provided function is a Rust value implementing a callback trait
  (`ExternalFunction`) over typed values plus its declared signature; a
  function instance's data is either a Wasm function's module instance or a host
  callback, placing host and Wasm functions in one shared function-instance
  space.

- A host external function is invoked with its typed argument values together
  with the store and the calling module instance, giving it access to the
  caller's guest linear memory.

## Facts

- 2025-06-23 (09dbe4a3) rationale: when resolving a function import,
  instantiation consults the runtime's external-function registry (keyed by
  module+field) first and verifies the registered callback's signature equals
  the import's declared type; only if no external function is registered does it
  fall back to resolving the import against another module's Wasm exports, so a
  host function transparently shadows a Wasm import of the same name (code).

## Moves

- 2025-06-23 (40c1e696) replaced [[host-functions.alt/args-only-abi]]: a host
  function that only receives its typed arguments cannot reach the caller's
  guest linear memory, so the call ABI now also passes the store and the calling
  module instance (code).

- 2026-02-14 (a8528504) replaced by [[runtime]]: the host-function hook is a bare
  `fn` pointer rather than a `dyn` trait object because that is zero-alloc and
  no_std-friendly and carries multi-value results through a caller buffer; WASI
  and other host capabilities are provided by an external crate passing function
  pointers in, not built into the core (code).
