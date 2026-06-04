---
commit: 38809e6, 78b1f6d
---
The entire fast-interpreter and instruction-fusion build system was deleted
(2026-04-07/08): `sf-nano-core/build/fast_interp/*` (gen_fusion*, gen_handler*,
gen_encoding, gen_ir_resolve, op_classify, tos_config — ~6,000 lines), the
`discover_fusion` CLI subcommand (~494 lines), `INTERPRETER_DESIGN.md`, the paper
draft, and the spectest discovery module. Earlier the static-discover tool,
`FUSION.md`, `MICRO_JIT.md`, and the Native Backend Roadmap were also removed and
the README rewritten to describe the engine as the native/JIT pipeline. The micro-JIT
/ native pipeline became the sole execution backend; `feature = "micro-jit"` is now
the only execution-backend feature.

This is the abandonment fact that records the interpreter→JIT pivot as *complete*:
not "interpreter demoted to fallback" but "interpreter removed entirely." The
driver was that the new JIT pipeline had reached spectest-passing parity and
competitive benchmarks, making the interpreter + fusion stack redundant dead
weight. The README repositioned the project as a "no_std WebAssembly JIT engine
that goes head-to-head with V8 and Wasmtime."
