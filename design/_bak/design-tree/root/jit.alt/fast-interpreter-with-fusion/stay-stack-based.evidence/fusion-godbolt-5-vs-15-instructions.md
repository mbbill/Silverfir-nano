---
commit: 4bb1de8
---
Godbolt evidence comparing how a C compiler lowers a fused 3× `ctz` sequence under
different operand models: 5 instructions in the stack-machine model (the compiler
sees `sp[top]` as a compile-time constant and optimizes across the fused ops), 15
instructions in a register-machine model (operands loaded from the instruction
stream, compiler must assume aliasing, ops stay serialized), and 3 instructions
with a TOS register window and zero memory traffic. Separately, register-machine
fusion suffers a combinatorial explosion of operand-pattern variants (wasm3: one
extra meta-register grew a handler from 3 to 10 variants), which is why no register
interpreter automates superinstruction generation. This compiler-output observation
is the core argument that selected "stay stack-based" over converting to a register
machine: the stack model is what makes *automatic* fusion tractable.
