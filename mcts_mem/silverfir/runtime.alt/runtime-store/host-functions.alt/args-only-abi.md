- A host external function is invoked with only its typed argument values
  (`call(&self, args: &[Value]) -> Result<Vec<Value>, WasmError>`); it has no
  handle to the store or the calling module and cannot read or write guest
  memory.

- A function instance's evaluation dispatches an external function directly by
  calling its callback with the argument values.

## Moves

- 2025-06-23 (40c1e696) replaced by [[host-functions]]: a host function that
  only receives its typed arguments cannot reach the caller's guest linear
  memory, so the call ABI now also passes the store and the calling module
  instance (code).
