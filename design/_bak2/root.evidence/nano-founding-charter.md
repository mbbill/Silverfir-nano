---
commit: a852850
---
sf-nano.md ships in nano's initial commit as a decision table. Goal: a
standalone #![no_std] crate extracting the fast interpreter from sf-core
for small binary plus preserved speed, WebAssembly 2.0. Stripped wholesale:
the SSA interpreter, the in-place interpreter, the XIR backend, the whole
compiler pipeline, GC, profiler/trace, multi-module linking, Rc/RefCell,
logging, std itself. Kept non-negotiable: instruction fusion ("contributes
~40% of performance"). Validation became feature-gated pure validation —
the jump table and max-stack-height computation left the validator because
the fast interpreter builds its own IR. WASI moved outside the core,
implemented against plain function-pointer hooks.
