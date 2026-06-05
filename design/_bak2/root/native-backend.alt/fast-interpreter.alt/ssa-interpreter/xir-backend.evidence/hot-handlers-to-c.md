---
commit: 782f0c91
---
C handler bodies were present from the XIR backend's first days
(handlers_c/arithmetic.c exists by 2025-10-20, alongside the permuted
wrappers). The C surface then grew incrementally; the 2025-11-30 wave
(68ee614e, 2a41e65f, 782f0c91 — "for performance" per the messages) moved
the remaining hot families over: fused ops, loads/stores of every size,
spill copies. Rust kept slow paths and the IR builder. This incrementally
retired the earlier principles "all op semantics live in Rust" and "safe
accessors only" for the hot path.
