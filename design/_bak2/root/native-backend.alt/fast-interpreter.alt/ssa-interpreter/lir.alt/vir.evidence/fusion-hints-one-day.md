---
commit: 030d5e92
---
Fusion-hint metadata on basic blocks was added (e7808d16) and removed
(030d5e92) within a day: with VIR lowering pattern-matching fused forms
directly, carrying hints through the IR was redundant. The exploration
lived less than 24 hours; matching at lowering won.
