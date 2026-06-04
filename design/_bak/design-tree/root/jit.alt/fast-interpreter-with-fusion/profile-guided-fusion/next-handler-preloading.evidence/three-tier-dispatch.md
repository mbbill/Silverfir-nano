---
commit: 5e9345d
---
After fusion, the TOS window, and the hot-local cache, a typical handler body is
a single arithmetic op, so dispatch dominates and the bottleneck is the
load-to-use latency of fetching the next handler pointer from memory. Next-handler
preloading hides it: each handler receives the next handler pointer (`nh`) already
loaded by the previous handler, and while doing its own work it loads the handler
one further ahead (`new_nh = pc_next(np)->handler`), giving the CPU a full
handler's worth of time to cover the latency.

A three-tier dispatch strategy is generated automatically per handler (the
`DispatchMode` returned per handler, commit 5e9345d). (1) Always-linear handlers —
whose impl always returns `pc_next(pc)` — let the compiler prove the guard
`np == pc_next(pc)` is always true and eliminate the branch; they tail-call
straight through the preloaded `nh`. The actual AArch64 disassembly of
`op_i32_add_D2` is five instructions total (one `add` of real work, the rest
dispatch: save preloaded nh, preload new_nh, advance pc by 32 bytes, branch),
with no guard, no prologue, no epilogue, and the four TOS registers, three hot
local registers, frame pointer, and pc all live across the call. (2)
Potentially-branching handlers (e.g. `br_if`) keep the guard behind a `likely()`
hint and use the preloaded `nh` on the linear fast path, reloading only on a taken
branch. (3) Always-nonlinear handlers (`br`, `return`, `call`) discard the
preloaded `nh` and always reload from `np->handler` (`DispatchMode::Nonlinear`).

This was the last interpreter-era dispatch optimization. Once the per-handler body
is this small and residency this tight, what remains is pure dispatch-shaped
overhead in hot loops — the structural ceiling the JIT pivot exists to break.
