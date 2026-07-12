- The threaded interpreter carries three hot virtual registers (v0, v1, v2) as
  `preserve_none` handler arguments through the trampoline and tail-chain; the
  abstract register file passed across handlers is three wide.

- Lowering is stateless: physical registers R0/R1/R2 map one-to-one onto window
  slots v0/v1/v2 and the backend emits spills explicitly, but the register file
  is bounded at three slots, spilling values to numbered slots frequently.

- Per-permutation handlers are generated from a spec that enumerates every
  register permutation explicitly for each signature over the three slots.

## Moves

- 2025-11-29 (005fae86) replaced by [[register-model]]: three hot registers
  forced heavy spill/load traffic, and dispatch benchmarks showed passing eight
  registers through the preserve_none convention is nearly free per instruction
  even though the per-permutation handler count grows ~10x (to ~15K), so the
  interpreter was widened to eight hot registers to keep more values resident
  (code).
