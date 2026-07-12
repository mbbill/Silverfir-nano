- VIR instructions mirror SSA semantics but operate on virtual-register indices
  instead of SSA values, with types tracked separately in metadata.

- Similar SSA operations are merged in VIR: struct.get_s and struct.get_u lower
  to a single StructGet, and ref.cast_null / ref.test_null lower to RefCast /
  RefTest, with the extension and nullability distinctions left to be tracked
  elsewhere.

## Moves

- 2025-10-23 (a9655e3e) replaced by [[vir-two-stage-middle]]: the merged VIR
  instructions had no field to carry the sign-vs-zero extension and nullability
  distinctions, so packed-field gets and nullable cast/test lost their meaning
  during lowering; giving VIR its own per-variant instructions restores correct
  runtime behavior and pushes any fusion to the backend (code).
