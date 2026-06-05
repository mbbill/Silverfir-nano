---
commit: 51c76585
---
Author (2026-06-04): the rotating TOS cache is what made the fast
interpreter beat the compiler pipeline. Rotation constrains the stack top
to N *positions*, so handler count grows linearly (a 5-slot window means 5
copies per handler) — versus the factorial permutations of true allocated
registers. Add a local cache and memory access is eliminated too. Fewer
handlers, no ParCopy, no register allocator at all, fewer dispatches, and
the model is super natural for fusion. That is why nano was born: a small
engine with no RA, most operations in registers via local cache plus
rotating TOS.
