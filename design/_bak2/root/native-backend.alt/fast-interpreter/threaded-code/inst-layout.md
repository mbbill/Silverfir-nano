- Each instruction is a fixed 32-byte record — handler pointer, alt
  (branch/auxiliary) pointer, and op-specific immediate fields — with the
  size statically asserted on both the C and Rust sides.
- The fallthrough successor is implicit: instructions are contiguous and
  execution falls through to pc+1; only the alt target is a stored pointer.
- Variable-length immediates live in a per-function side blob built at IR
  time; the instruction array itself stays dense and uniform.
