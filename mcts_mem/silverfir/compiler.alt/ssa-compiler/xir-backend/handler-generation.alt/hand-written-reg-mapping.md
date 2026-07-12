- The LIR-physical-register to XIR-permutation mapping is a hand-written module
  of per-signature functions (sig_0_1, sig_2_1, ...), each a match over the
  concrete register indices returning the matching generated permutation enum
  variant.

- The mapping assumes LIR physical registers correspond one-to-one to XIR
  registers (R0->v0, R1->v1, R2->v2) and that the allocator only ever produces
  register combinations for which a permutation handler exists.

## Moves

- 2025-11-29 (11a2cdff) replaced by [[handler-generation]]: the hand-written
  mapping enumerated one match arm per register permutation for 3 registers and
  could not scale to 8 (e.g. 512 arms for 3-input signatures); generating the
  mapping functions in build.rs from the same arity_patterns.toml that generates
  the handlers keeps the LIR-register-to-XIR-permutation mapping in lockstep with
  the handler set at any register count (code).
