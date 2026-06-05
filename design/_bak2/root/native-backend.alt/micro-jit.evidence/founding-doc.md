---
commit: 16c7e03
---
MICRO_JIT.md frames the JIT as the completion of the interpreter's own
register discipline, the exact inversion of the -rs compiler pipeline:
"a 'JIT compiler' doesn't need SSA, register allocation, or an optimizing
compiler — it just needs a micro-assembler that emits 1-2 ARM64
instructions per variant instruction using the known register mapping."
And: "The micro-JIT **is** the fusion system, not a layer on top of it" —
it removes static fusion's three limits (immediate budget, finite pattern
set, discovery step) while reusing the entire builder pipeline. Deferred
peepholes named at birth: alias tracking, constant tracking, destination
forwarding, compare+branch fusion.
