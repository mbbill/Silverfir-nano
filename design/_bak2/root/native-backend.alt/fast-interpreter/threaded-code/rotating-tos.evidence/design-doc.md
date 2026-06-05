---
commit: e4830d87
---
The fast interpreter measured memory-bound: with dispatch eliminated by
tail-call threading, i32.add still cost 3 memory operations. The design
doc's key insight: "top register position is deterministic from stack
depth" — top = (depth-1) mod N — so rotation is a compile-time mapping
with zero runtime register motion, and depth-variant selection
(op_i32_add_D2, ...) happens at IR build from the no-stack model's static
heights. This sidesteps exactly the dynamic-register-selection cost the
XIR dispatch lab measured as the killer, and revives the original
register-ToS-lanes idea on a sound static foundation. Designed for 8 TOS
registers; reduced to 4 in implementation (8136fd44).
