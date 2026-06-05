---
commit: aed2ff42
---
Born as a 208-line in-repo design doc, before any code. Objectives, verbatim:
"Minimize dispatches via fused superinstructions selected from SSA trees.
Minimize per-dispatch work using direct-threaded dispatch and tiny handlers.
Keep the ISA static but parameterized and extensible for future fused
families. Preserve Wasm semantics (order, traps, structured control) with
conservative barriers." Declared a clean slate: "It does not inherit layout
or constraints from the existing fast interpreter." The doc already contains
the SU/tile-covering pipeline, IC/PIC plans for call_indirect, and an
8-phase plan each gated on spectest subsets.
