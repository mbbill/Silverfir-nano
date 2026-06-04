---
status: abandoned
---
# Static two-bank register split

The split between local-cache registers and transient stack-value registers is
fixed once per function at compile time and never changes. Each register class
has two banks of fixed size: ARM64 had 13 GP local-cache registers (x23–x28,
x9–x15) plus 9 GP transient registers, the transient bank a sliding window over
the operand stack.

## In practice

While in force this entailed:

Must:
- Partition each register class into a fixed local-cache bank and a fixed
  transient bank, sized once per function and never resized.
- Use the transient bank as a sliding window over the Wasm operand stack.
- Pin the function-wide set of cached locals into the local-cache bank.

Must not:
- Reuse idle transient registers for local caching within a function.
- Grow the local-cache set beyond the function-wide fixed allocation, even when
  a hot loop wants another local resident.
