# L0/L1/L2 hot-local register cache

Fusion removes the dispatch cost of `local.get`/`local.set`, but the fused handler
body still touches `fp[idx]` memory, and local-variable access is a large share of
all dispatches. This promotes the three hottest locals into register arguments
(`l0`/`l1`/`l2`) threaded through the handler chain — orthogonal to the TOS window,
adding no extra handler variants — with spill/fill folded into the call/return
handlers.

Compile-time hot-local analysis selects the three hottest locals; an index swap at
function entry remaps them to indices 0/1/2 and loads them into the l0/l1/l2
registers. The l0/l1/l2 ops are first-class fusion participants, so fused hot-local
operations compile to register arithmetic with zero frame-memory access.

## In practice

Must:
- Select the three hottest locals by a single-pass walk of the raw Wasm bytecode,
  counting local accesses weighted by loop-nesting depth (×10 per nesting level);
  no CFG or SSA is built (see
  facts/l0-l1-l2-hot-local-mechanism-loop-weighted-index-swap.md).
- Remap the chosen locals to indices 0/1/2 at the function prologue and load them
  into the l0/l1/l2 register arguments; route subsequent references through
  `local_get_l0`/`local_set_l1`/`local_tee_l2`/etc.
- Make the l0/l1/l2 ops first-class participants in fusion discovery and matching,
  so fused hot-local ops fold into operand fields and emit zero frame-memory
  accesses.
- Fold spill/fill into the existing call/return handlers (`fp[0..2] = l0,l1,l2`
  before a call; reload after return).

Must not:
- Emit separate spill/fill instructions for the hot-local registers.
- Add per-handler variants for the hot-local cache (it is orthogonal to the TOS
  window depth variants).
