- Permutations are listed by opaque name strings in the spec; the build step
  decodes each name with a per-signature match (Sig_0_1 / Sig_1_1 / Sig_2_1 /
  ...) to compute which window registers map to the canonical implementation
  positions.

- The canonical C implementation writes its result to a fixed pointer (pv0); the
  handler signature has no explicit destination pointer; the output slot is
  not a free parameter and the result lands implicitly in the first input's slot.

## Moves

- 2025-10-24 (d097f067) replaced by [[handler-generation]]: the old generator
  named each permutation with an opaque string (V0V1_V0) and decoded it through
  per-signature hardcoded match arms that mapped window slots onto a fixed
  canonical layout (pv0=lhs, pv1=rhs, result implicitly to pv0), which could not
  express an output landing in a slot chosen independently of the inputs;
  declaring each permutation as explicit inputs[] and outputs[] slot vectors and
  giving every handler a uniform signature with an explicit pv_dst output pointer
  makes the result slot a first-class free parameter and removes the special-cased
  shuffling (code).
