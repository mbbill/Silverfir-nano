# Tail-call dispatch

Each opcode handler is an independent leaf function that ends by tail-calling the
next handler. With a guaranteed `musttail` plus the `preserve_none` calling
convention, the tail call eliminates the prologue/epilogue: the handler chain runs
as one continuous flow with no call/return overhead, while each handler keeps its
own branch-target-buffer entry and stays individually optimizable by the C
compiler.

The handler signature carries the threaded register state end-to-end — `ctx, pc,
fp, l0–l2, t0–t3, nh` — which is how the TOS window and the L0/L1/L2 hot-local
cache stay physically in registers across the whole chain. `preserve_none` plus
the tail call are precisely what keep those values from being spilled at each
handler boundary.

## In practice

Must:
- End every hot handler with a `musttail` tail call to the next handler, under the
  `preserve_none` calling convention, so the chain executes with no
  prologue/epilogue and no per-boundary spill.
- Give each handler its own BTB entry (each handler is a distinct function).
- Thread the full register state (`ctx, pc, fp, l0–l2, t0–t3, nh`) through the
  handler signature so the TOS window and hot-local cache remain register-resident
  across handlers.
- Write the hot handler chain in generated C reached through a Rust→C trampoline,
  because stable Rust provides neither `musttail` nor `preserve_none` (see
  facts/stable-rust-lacks-musttail-and-preserve-none.md).

Must not:
- Emit any prologue/epilogue or ABI spill on the per-handler dispatch path.
- Drop or reorder the threaded register arguments such that a hot value loses its
  fixed physical home across a handler boundary.
