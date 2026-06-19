- Stack-overflow protection is a per-call-site precheck: every direct local call
  computes the proposed callee frame base and compares it against
  NativeContext.stack_end, trapping before transfer; dynamic/indirect local calls
  load the callee frame size from local-call metadata and perform the same check.

- The wasm value stack is a plain heap Vec<u64> with no guard range, and the
  guard-page signal handler classifies every JIT-attributed fault as
  MemoryOutOfBounds (trap_kind = 1).

## Moves

- 2026-05-14 (d3af717a) replaced by [[stack-overflow]]: computing the proposed
  callee frame base and comparing it against stack_end before every local call
  adds arithmetic and a branch to each call site; on guard-page 64-bit targets the
  wasm value stack can instead be allocated with an inaccessible guard range sized
  to the module's largest frame, so a single body-entry probe of the highest frame
  slot faults into the guard on overflow and the signal handler classifies it as
  StackOverflow by fault address, removing the per-call precheck from the hot
  local-call path (code).
