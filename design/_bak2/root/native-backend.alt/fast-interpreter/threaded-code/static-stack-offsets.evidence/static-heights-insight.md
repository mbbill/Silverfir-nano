---
commit: e5ae6403
---
The no-stack design doc's verification section: for a valid WebAssembly
function, the stack height at every instruction is statically determinable
and identical across all control-flow paths reaching it (guaranteed by
validation). Therefore the operand stack can be treated as a fixed array of
virtual registers with compile-time slot addressing — register-machine
execution without building SSA.
