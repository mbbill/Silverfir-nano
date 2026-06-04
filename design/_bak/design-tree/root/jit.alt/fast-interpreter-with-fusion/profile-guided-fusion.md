# Profile-guided automatic instruction fusion

Even with tail-call dispatch, the dominant interpreter cost is dispatch itself —
one handler jump per Wasm instruction. Fusion attacks that directly: discover
common consecutive-instruction patterns in real workloads and compile each pattern
to a single fused C handler, so one dispatch does the work of several.

The mechanism is a code-generation pipeline — `handlers.toml` +
`handlers_fused.toml` → `gen_fusion_*.rs` → a `FusedOp` enum, an `OpFuser`
matcher, `emit_fused`, and the C fused handlers — fed by an automatic discovery
tool that normalizes TOS variants, builds an N-gram trie (up to 8-grams), filters
patterns by encoding budget (≤ 3 immediate slots) and control/memory constraints,
then greedily selects the highest-savings patterns.

This option opens two sub-problems, both in force together (`profile-guided-fusion.all/`):
instruction-encoding (how wide each instruction is — the fixed slot layout fusion
fills) and hiding-dispatch-latency (once the body is ~1 op, load-to-use latency
dominates).

## In practice

Must:
- Discover fused patterns automatically from real workloads: normalize TOS
  variants, build the N-gram trie (up to 8-grams), filter by the encoding budget
  (≤ 3 immediate slots) and control/memory constraints, then greedily select the
  highest-savings patterns.
- Generate the fused handlers through the code-generation pipeline
  (`handlers.toml` + `handlers_fused.toml` → `gen_fusion_*.rs` → `FusedOp` enum,
  `OpFuser` matcher, `emit_fused`, C handlers) — fused handlers are generated, not
  hand-written per pattern.
- Reject any candidate pattern whose combined immediates exceed the 3 slots, or
  that crosses control/memory constraints.
- Target `local.get` and its hot successors specifically, since the dispatch mix
  is skewed (`local.get` ~26% of dispatches, local access ~38%) and fusing them
  removes roughly two-thirds of all dispatches (see
  facts/local-get-dispatch-profile-fusion-removes-two-thirds.md).

Must not:
- Require manual authoring of each fused superinstruction (discovery + codegen is
  what makes wide automatic fusion tractable on the stack model).
- Emit a fused pattern that does not fit the fixed instruction encoding.

## Ground rules — instruction-encoding
Must:
- Fix the per-instruction width and the immediate-slot count, and use that slot
  count as the budget fusion discovery enforces.
- Keep the decode fixed-stride and cache-aligned.

Must not:
- Choose an encoding that lets fusion select a pattern the encoding cannot hold.

## Ground rules — hiding-dispatch-latency
Must:
- Cover the load-to-use latency of fetching the next handler pointer so it does
  not stall the pipeline once handler bodies are ~1 op.
- Keep linear (non-branching) handlers free of any runtime dispatch guard.

Must not:
- Reintroduce a stall on the common linear dispatch path.
