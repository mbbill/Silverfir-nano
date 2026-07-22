- Every host import is a bare non-capturing `HostFn` pointer with the typed
  Caller/params/results ABI; no allocation or dynamic callback dispatch is
  needed to store it.

## Moves

- 2026-07-22 (9ae4eb43) replaced by [[host-callbacks]]: stateful adapters and
  imports were not representable as plain `fn` values; shared capturing
  callbacks retain the typed ABI and plain-function compatibility (code).
