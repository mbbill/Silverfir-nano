---
commit: c607440
---
Author (2026-06-04): "no SSA" was half true. SSA usually means an SSA IR
feeding a downstream pipeline — most likely a register allocator. Nano's
SSA is a limited-residency linear SSA: the number of simultaneously-live
SSA values is bounded, and linearity means every value is single-use.
Therefore the downstream register allocator is completely eliminated —
the SSA form itself is the allocation.
