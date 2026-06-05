---
commit: ba69c050
---
Author (2026-06-04): in-place execution was the natural first move, not a
considered bet. The side jump-table design (precomputed target pc / arity /
stack offset per branch) was carried over from my previous C interpreter — I
implemented it there first; I am the first to implement this concept.
