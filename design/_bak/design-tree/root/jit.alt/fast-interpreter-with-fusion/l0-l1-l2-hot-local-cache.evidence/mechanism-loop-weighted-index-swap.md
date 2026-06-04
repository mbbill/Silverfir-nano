---
commit: 296a489, 309d0db
---
The hot-local register cache promotes the three hottest local variables into
register arguments (`l0`/`l1`/`l2`) threaded through the entire handler chain.
Its mechanism has two compile-time pieces. (1) Hot-local analysis: a single-pass
walk of the raw Wasm bytecode counts local accesses weighted by loop-nesting
depth — each nesting level multiplies the weight by 10, so a `local.get` inside a
double-nested loop counts 100x a flat one — and the top three locals by weight
become l0/l1/l2. No CFG or SSA is built. (2) Index swap at function entry: the
prologue emits swaps that remap the chosen hot locals to indices 0/1/2 and loads
them into the l0/l1/l2 registers; thereafter references go through register ops
(`local_get_l0`, `local_set_l1`, `local_tee_l2`, ...), and the displaced
originals use the remapped frame slots. Spill and fill are folded into the
existing call/return handlers (`fp[0..2] = l0,l1,l2` before a call; reload after
return) — no separate spill/fill instructions. The l0/l1/l2 ops are first-class
participants in fusion discovery and matching.

It grew incrementally: the first cut promoted one local (l0, commit 296a489), the
trampoline was extended to pass l1 (commit 309d0db), then the design generalized
to l0/l1/l2 caching the three hottest locals. The payoff is that when these ops
fuse with arithmetic the compiler eliminates the local access entirely — it
becomes a register-to-register copy folded into operand fields. A 6-instruction
hash/accumulator sequence (`local_get_l0 → local_get_l1 → i32.xor →
local_get_l2 → i32.add → local_set_l0`) compiles to two AArch64 instructions
(`eor`, `add`) with zero memory access; all four frame memory operations vanish.
This register-residency idea — give the hottest values a fixed physical home so
fused/compiled code never touches their frame slot — survived the JIT pivot intact
and became the cached-local concept the SSA-IR / joint-planner register model is
built around.
