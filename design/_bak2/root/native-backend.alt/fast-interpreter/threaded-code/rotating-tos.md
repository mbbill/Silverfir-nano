- The top of the operand stack lives in a bank of four TOS registers
  (t0–t3); the mapping is circular and purely compile-time — the register
  holding the top is `(depth-1) mod 4` — so nothing physically rotates at
  runtime.
- Each handler exists in depth variants (D1–D4); the IR builder selects the
  variant from the static stack height. Register selection is never
  dynamic.
- Pushing past the bank spills the oldest value to the memory stack; fills
  are lazy and demand-driven (`spill_depth` tracks the register/memory
  split; popping never eagerly reloads).
- Before branches and block ends, spill state normalizes to the minimum so
  every control-flow path agrees on what is in registers.
- Compute is TOS-only: handlers read and write the register bank; the
  memory stack holds only the cold bottom.
- Calls and returns follow a register convention: arguments and results
  travel in TOS registers instead of staging through memory.
