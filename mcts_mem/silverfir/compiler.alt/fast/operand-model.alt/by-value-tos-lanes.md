- Each instruction handler takes the four top-of-stack lanes (t0..t3) and a
  depth byte as separate by-value register arguments and returns a Next struct
  bundling the next Instruction pointer together with the updated lanes and
  depth.

- The C wrapper unpacks the returned Next and tail-jumps into the next handler
  carrying the lanes forward by value.

## Moves

- 2025-08-13 (240fb3d8) replaced by [[register-window]]: passing the four
  top-of-stack lanes plus depth as separate by-value arguments forced per-call
  stack argument traffic on Win64 (only four GP argument registers) and
  zero-extend shuffles for the byte-sized depth; a single by-pointer register
  bank removes both (code).
