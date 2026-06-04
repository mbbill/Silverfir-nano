# Fixed 32-byte (4×64-bit) instruction encoding

Every instruction is four 64-bit words: one handler pointer plus three immediate
slots (local indices, constants, branch targets, memory offsets, TOS-encoded
offsets). On its face wasteful — most ops need zero or one immediate — but fusion
makes the width dense rather than wasteful: any pattern whose combined immediates
exceed the 3 slots is rejected at discovery time, so every fused pattern fits and
longer fusions pack the slots tightly.

The payoff is a branchless fixed-stride decode and cache-line alignment: two
instructions fit exactly in a 64-byte cache line.

## In practice

Must:
- Encode every instruction as exactly four 64-bit words (32 bytes): one handler
  pointer plus three immediate slots.
- Reuse the same ≤3-immediate-slot budget as the fusion-discovery reject filter,
  so every fused pattern is guaranteed to fit the fixed encoding (see
  facts/fixed-32-byte-encoding-fusion-budget-cache-line.md).
- Keep the decode branchless and fixed-stride, advancing pc by 32 bytes per
  instruction, so two instructions land per 64-byte cache line.

Must not:
- Admit a fused pattern whose combined immediates exceed the three slots.
- Use a variable-length encoding for the interpreter instruction stream (the fixed
  width is what gives the branchless decode and cache-line alignment).
