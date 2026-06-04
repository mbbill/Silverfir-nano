# Next-handler preloading with guard-check dispatch

After fusion and the register windows, the handler body is often a single
arithmetic op — so the gap between loading the next handler pointer and jumping to
it is almost nothing, and the CPU stalls on load-to-use latency. Preloading hides
it: each handler receives the next handler pointer (`nh`) already loaded by the
previous handler, and while executing loads the handler one further ahead.

A three-tier dispatch strategy is generated automatically per handler:
always-linear handlers eliminate the guard entirely, potentially-branching
handlers keep a `likely()` guard, and always-nonlinear handlers skip the preload.

## In practice

Must:
- Pass the next handler pointer (`nh`) as a handler argument, preloaded by the
  previous handler; each handler additionally loads the handler one further ahead
  (`new_nh = pc_next(np)->handler`) while doing its own work.
- Classify each handler into one of three dispatch modes and generate accordingly
  (see facts/next-handler-preloading-three-tier-dispatch.md):
  - always-linear: guard `np == pc_next(pc)` is provably true and eliminated;
    tail-call straight through the preloaded `nh` (e.g. `op_i32_add_D2` is 5
    AArch64 instructions, no branch);
  - potentially-branching: keep the guard behind a `likely()` hint, use `nh` on
    the linear fast path, reload from `np->handler` only on a taken branch;
  - always-nonlinear (`br`/`return`/`call`): discard the preloaded `nh` and always
    reload (`DispatchMode::Nonlinear`).

Must not:
- Insert a runtime guard on always-linear handlers (the whole point is that the
  compiler proves it away).
- Use the preloaded `nh` on an always-nonlinear handler.
