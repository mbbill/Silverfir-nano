- Each activation owned an independently heap-allocated, zero-initialized
  frame; call arguments were copied from the caller's staged slots into
  the callee frame, and results copied back into the caller on return.

- Every call and return crossed the native-chain boundary: the chain
  exited to the Rust driver, which allocated or dropped the frame,
  managed the activation stack, and re-entered the chain.

## Facts

- 2026-07-23 measurement: 125M call/return boundary crossings on one
  CoreMark run made calls the dominant cost after Select/BrTable went
  native — the exit round-trip plus per-call allocation, not the callee's
  work (code).

## Moves

- 2026-07-23 replaced by [[calls]]: every call paid a heap allocation,
  two value copies, and a native-chain exit and re-entry; the overlapped
  contiguous stack makes argument and result movement structural (code)
