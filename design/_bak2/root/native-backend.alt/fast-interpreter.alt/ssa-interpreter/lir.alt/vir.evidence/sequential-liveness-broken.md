---
commit: cd290851
---
The first SSA→VIR allocator computed liveness by sequential instruction
numbering across blocks — correct only if control flow is sequential, which
loops are not. Backward edges produced inflated live ranges, papered over
with hacks (reserve call-arg vregs, extra phi interference edges, extend
ranges to end of function) costing ~30% extra vregs and 500+ lines of
"CRITICAL FIX" workarounds. The redesign plan replaced it with textbook
backward-dataflow liveness (CFG → UEVar/Def → fixpoint → interference →
allocate), built alongside a trivial 1:1 allocator as the correctness
baseline, validated, then switched over.
