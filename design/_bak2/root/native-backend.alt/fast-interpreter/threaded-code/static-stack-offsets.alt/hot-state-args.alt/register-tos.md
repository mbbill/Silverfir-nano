- Top-of-stack values travel between handlers in a small register bank
  (`Regs`): four 64-bit lanes plus a depth word, passed by pointer. Handlers
  mutate the bank in place and return only the next instruction pointer.
- Values below the lanes live in an explicit in-memory value stack; lanes
  spill there when the window overflows.
