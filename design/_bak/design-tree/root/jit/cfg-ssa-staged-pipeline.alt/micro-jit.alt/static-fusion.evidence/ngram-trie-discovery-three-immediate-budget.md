---
commit: 3b2933b, 9e81b63
---
The static-fusion discovery tool (a standalone Rust workspace member,
`tools/static-discover`, ~1500 lines across `discovery.rs`, `trie.rs`, `main.rs`)
analyzed Wasm binaries offline: it extracted instruction N-grams weighted by
loop-nesting depth, stored and retrieved candidate patterns in a trie, extracted
the call graph and propagated pattern weights across it, and emitted a TOML
fusion table (`[[fused]]` entries with encoding fields, TOS patterns, and names)
that compiled into precompiled C handlers. A separate stage generated each
pattern's encoding-field layout — which immediates to pack, their bit widths, and
their source indices.

The intrinsic ceiling FUSION.md recorded for this approach: a fused pattern was
rejected if its encoding budget exceeded 192 bits, i.e. three 64-bit immediate
slots. Combined with a finite, workload-discovered pattern set and a discovery
step that had to be re-run for new workloads, these are the three concrete limits
the micro-JIT removes by assembling fused code at run time. This is the
mechanism-and-budget fact behind static fusion's abandonment — distinct from the
later fact that records the static-discover tool and FUSION.md being deleted
outright; this one captures *what the tool did and why its coverage was bounded*.
