---
status: abandoned
---
# Function-pointer table dispatch

Each opcode handler is an ordinary function, and dispatch indexes a table of
function pointers and performs a normal call. Each handler is an independent
function (its own BTB entry, independently optimizable by the compiler), but each
dispatch is a full call/return.

## In practice

Must:
- Implement each handler as an independent function reached by a normal indirect
  call through a function-pointer table (one BTB entry per handler).

Must not:
- Rely on register residency across handlers: a normal call/return pays
  prologue/epilogue and ABI-mandated register spills on the hot path, which evicts
  the threaded TOS-window and L0/L1/L2 hot-local registers the design depends on
  (see facts/function-pointer-table-call-overhead-destroys-residency.md).
- Be used where tail-call dispatch is available — tail calls keep the per-handler
  independence without the call overhead.
