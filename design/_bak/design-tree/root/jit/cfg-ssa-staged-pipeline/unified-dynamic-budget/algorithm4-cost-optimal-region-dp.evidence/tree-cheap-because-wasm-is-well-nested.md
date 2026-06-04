---
commit: a50a44d
---
Two environment properties make ALGORITHM4's Lagrangian region-tree DP cheap
enough to be the *sole* per-function residency planner rather than the heavyweight
tier of an amortizing allocator. (1) Wasm block/loop/if are well-nested by
construction, so the Callahan–Koblenz region tree is read straight off the decode —
no SCC discovery, no irreducibility handling, no heuristic region formation. (2) A
typical Wasm function has tens of locals and single-digit-to-low-tens of regions,
so an O(regions)-per-iteration tree DP run for a fixed 8–12 subgradient rounds is a
few thousand ops per function — well below downstream instruction selection and
register handling.

The earlier constraint-based planners (ALGORITHM2 one global resident set,
ALGORITHM3 root set + per-loop override) minimized per-block frame-access cost but
ignored edge transition cost, causing boundary churn — locals repeatedly ensured
and dropped at edges. Reframing stability as a *cost* (Lagrangian relaxation over
the region tree) makes whole-function stability and loop overrides *emerge* rather
than be special-cased. This couples a structural fact about Wasm (well-nested
control flow) to a complexity argument (JIT-scale instances are tiny).
