---
commit: 37c40ff
---
The Native Backend Roadmap (added 2026-03-06) set out the design principles that
shaped the CFG+SSA native backend's ABI, beyond the benchmark argument for
abandoning the micro-JIT. The interpreter-era hot path depended on `preserve_none`
for its fixed register ABI and zero/near-zero prologue cost, but `preserve_none`
is not portable: it is unavailable or unreliable on RISC-V, ARM32, and MCU-like
targets. The roadmap therefore required that `preserve_none` become an *optional,
target-specific optimization* — missing it should hurt performance but never
correctness — and that the native backend define its own internal VM ABI and use
direct code-to-code (JIT-to-JIT) branch chaining instead of entering each hot
opcode through an ordinary C/Rust ABI function whose prologue/epilogue and spills
destroy register residency.

This is the portability-principle fact behind the staged-pipeline ABI. It is
distinct from the toolchain-infeasibility fact that stable Rust lacks
`musttail`/`preserve_none` (which explains why the interpreter's handler chain was
generated C): this one is the forward-looking design constraint that the *next*
backend must not bake `preserve_none` into its structure, because the project
targets RISC-V, ARM32, and microcontrollers where it cannot be relied on. The
roadmap doc itself was later removed once the pipeline it described existed.
