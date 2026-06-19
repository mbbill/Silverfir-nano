2026-06-14 (sourced)

xir and the fast interpreter use the same handler-table dispatch; xir is the
more refined, later form, with improvements that were never backported to fast.
So xir did not die from its dispatch style — it died from real bottlenecks that
are hard (not impossible) for an interpreter to clear:

1. xir needs register permutation. Each added register multiplies the
   handler-variant count, so the number of registers xir can use is capped by
   the handler explosion — past roughly eight true registers the permutation
   count runs into the tens of thousands.

2. fast's fixed top-of-stack mapping is O(n) in opcodes (one depth-variant per
   opcode), plus a single fixed cached local. Because there is no permutation, it
   can use every arm64/rv64 register without exploding the handler count, and it
   has no register allocator at all (rotating TOS + local cache).

3. xir tends to carry more shuffle instructions — SSA edge fixes and ParCopy
   moves at control-flow joins — that the stack-machine fast path never emits.

4. Instruction count is the interpreter's tax: every extra instruction is one
   more dispatch. On a JIT the same shuffle is nearly free, because a mov is
   essentially register renaming on a modern out-of-order core. Proving those
   shuffles away well enough for an interpreter is harder than just emitting a
   real binary — too much of a goal for an interpreter.

5. The xir pipeline grew into a full compiler pipeline, which is too complex.
   fast and the later microJIT have no register allocator, so they sidestep every
   xir downside above and stay far simpler.

6. Correctness and debugging on a full compiler pipeline are very hard; the
   TOS + local-cache method is trivial to reason about — "a slightly complicated
   wasm stack machine with a register-mapping trick." That debuggability is a
   first-class reason the fast approach won.

The microJIT inherited every one of these benefits when the fast single-pass
method became the compiler product.
