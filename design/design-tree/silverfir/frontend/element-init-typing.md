- Element-init expressions carry the element segment's reference type, and the
  validator checks each init expression against that type (read through
  `Element::value_type`); element segments can hold funcref or externref
  values rather than funcref only.

## Facts

- 2025-06-22 (9e353093) conformance: applying an active element segment at
  instantiation type-checks each value against the target table's declared
  element type — a bare function-index element only into a funcref table, an
  init-expression element matching exactly (FuncRef into FuncRef, ExternRef into
  ExternRef) — rather than accepting any reference into any table (diff).

## Moves

- 2024-03-19 (0033e67d) replaced [[funcref-only-element-init]]: the funcref-only
  ElementInit::InitExprs(Vec<ConstExpr>) had no slot to carry a reference type,
  so it structurally could not express externref element segments (diff).
