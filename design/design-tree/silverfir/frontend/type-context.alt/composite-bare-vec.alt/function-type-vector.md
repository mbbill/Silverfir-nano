- The module's type section is a vector of function types only
  (Vec<Rc<FunctionType>>); every type index resolves to a function signature.

- FunctionType (params, results) lives in module::entities and is the only
  composite type the type section can hold.

## Moves

- 2025-10-04 (d93a9392) replaced by [[composite-bare-vec]]: a vector of function
  types could not represent the 3.0 unified type/index space: struct and array
  composite types, subtyping (supertype indices and finality), or recursive type
  groups all share one index space with functions (diff).
