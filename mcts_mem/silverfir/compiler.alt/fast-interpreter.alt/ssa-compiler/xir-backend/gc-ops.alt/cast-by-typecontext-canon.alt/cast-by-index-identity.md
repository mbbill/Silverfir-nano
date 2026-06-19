- A concrete GC cast target matches the object type only when their type indices
  are literally equal, walking the object type's supertype chain by index.

- Two structurally identical types declared under different indices are treated as
  distinct by the cast check.

## Moves

- 2025-10-14 (6f3e5855) replaced by [[cast-by-typecontext-canon]]: exact index
  identity matched only the literal type index and missed two structurally-equal
  concrete types declared at different indices; canonicalizing both sides through a
  TypeContext makes equal types match regardless of declaration index (code).
