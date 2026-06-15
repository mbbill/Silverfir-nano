- Element-init expressions are carried as ElementInit::InitExprs(Vec<ConstExpr>),
  which records no element reference type.

- The validator checks every element-init expression against ValueType::FuncRef;
  element segments can only hold funcref values.

## Moves

- 2024-03-19 (0033e67d) replaced by [[element-init-typing]]: the funcref-only
  ElementInit::InitExprs(Vec<ConstExpr>) had no slot to carry a reference type,
  so it structurally could not express externref element segments (diff).
