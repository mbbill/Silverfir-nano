---
commit: 00deb96a
---
Measured on the hottest CoreMark function: one block with 11 phi nodes
produced 22 copies, which became ~40 XIR window operations — 60% overhead —
because phis were lowered to sequential copies *before* register allocation
could see or coalesce them. The redesign keeps parallel-copy semantics
through RA (edge-attached ParCopy, affinity-coalescing weighted 10x on loop
back-edges, late resolution with cycle detection), citing LLVM/GCC/Cranelift
practice: late phi resolution is the industry standard.
