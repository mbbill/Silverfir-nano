---
status: abandoned
---
# Static fusion (offline-discovered C handler table)

The interpreter era's fusion approach carried onto JIT-capable platforms: a
standalone offline tool analyzes Wasm binaries, extracts instruction N-grams
weighted by loop-nesting depth, propagates them across the call graph, and emits
a fusion table that is compiled into precompiled C handlers. The fused handler
set is finite, fixed at build time, and discovered from a chosen set of workloads.

At run time the builder matches the pre-discovered patterns against the decoded
stream and assigns each match its precompiled C handler, reusing the TOS window
and L0/L1/L2 hot-local mapping for value placement.

## In practice

Must:
- Generate the fusion table from an offline discovery step (N-gram extraction +
  call-graph propagation) and compile the resulting handler set into the binary
  ahead of run time.
- Keep each fused handler within the encoding budget of three 64-bit immediate
  slots; patterns exceeding that budget cannot be encoded as a single fused op.
- Re-run discovery when targeting a new workload class, since coverage is limited
  to the patterns the discovery inputs exercised.

Must not:
- Emit or assemble fused handlers at run time.
- Treat the fused pattern set as open-ended; it is finite and fixed once the
  table is built.
