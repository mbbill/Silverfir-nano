- The hottest local variables are promoted into dedicated registers (l0,
  l1) passed through the entire handler chain as arguments, alongside the
  TOS bank.
- Hot locals are chosen at compile time by a single pass over the bytecode,
  counting accesses weighted by loop nesting depth (×10 per level) — no
  CFG, no SSA.
- The chosen locals are remapped to fixed indices at function entry; the
  displaced locals take the freed frame slots.
- Local registers are orthogonal to the TOS window: they add handler
  arguments, never handler variants.
- Spill and fill fold into the existing call/return handlers (save before
  call, reload after return); no dedicated spill instructions exist.
- Local-register ops are first-class fusion participants: patterns
  containing l0/l1 ops are discovered and generated like any others, and a
  fused local access costs zero instructions.
