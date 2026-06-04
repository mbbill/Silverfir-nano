# Stay stack-based

Keep the Wasm stack model instead of converting to a register machine. In a stack
machine `sp[top]` is a compile-time constant the C compiler sees through, so
fusing several stack operations lets the compiler optimize across them as if they
were one expression — which is what makes automatic fusion tractable.

Staying stack-based is also what makes the TOS window and the L0/L1/L2 hot-local
cache expressible as a linear number of depth-specific handler variants rather
than a register permutation: static Wasm verifiability gives the compile-time
stack height at every point.

## In practice

Must:
- Keep operands implicit on the Wasm operand stack, addressed as `sp[top-k]` so
  the compiler treats stack offsets as compile-time constants and optimizes across
  fused operations.
- Use the statically-known compile-time stack height at each instruction to emit
  depth-specific handler variants (e.g. `i32_add_D2`), so the TOS window and
  hot-local cache cost a linear number of variants per handler.

Must not:
- Rewrite the bytecode into virtual-register form whose operands are loaded from
  the instruction stream (that reintroduces the aliasing barrier that defeats
  automatic fusion).
