---
commit: 4bb1de8
---
Profiling LLVM-compiled Wasm workloads showed the dispatch mix is heavily skewed:
`local.get` alone is ~26% of all dispatches, local-variable access ~38% of all
dispatches, and arithmetic under 40%. The top-10 instructions that follow
`local.get` cover 88–92% of cases. Therefore fusing `local.get` with its ~10 hot
successors removes most `local.get` dispatches; measured on CoreMark, fusion
eliminates roughly two-thirds of *all* dispatches. This dispatch-frequency
measurement is what justified building the whole profile-guided fusion pipeline,
and the same ~38% local-access figure is what later justified caching hot locals in
registers (the fused handler body still touched `fp[idx]` memory).
