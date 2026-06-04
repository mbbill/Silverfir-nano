---
commit: 111163a, 3a9284d
---
The SSA-based i64-on-32-bit legalization (chosen over the deleted MachineIR
`legalize.rs`) was validated on real ARMv7-A: armv7a reached a passing state via a
series of SSA-side fixes — sink-premap of i64 cached locals, skipping i64
const-folding on 32-bit, mem0 bounds-check register aliasing, `Int64PairCompare`
clobbering its lo operands, and moving R9 from local-cache to transient. Each fix
lived in the SSA / middle layer, not in a late MachineIR pass.

This is the fact that confirms the legalization-layer pivot: handling i64 pairs in
SSA (so the planner accounts their register pressure at planning time) actually
works on real 32-bit hardware, where the abandoned MachineIR-level approach had
caused high-pressure failures. The corroborating cross-target result is that all
four early targets (arm64, x86_64, armv7a/emu32, emu64) passed spectest by end of
March. Real-hardware + emulator validation, on 32-bit ARM specifically.
