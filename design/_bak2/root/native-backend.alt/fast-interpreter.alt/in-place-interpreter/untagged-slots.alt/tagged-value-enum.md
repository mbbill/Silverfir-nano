- Operand-stack slots are a tagged Rust enum (`Value`) with one variant per
  wasm value type; the tag travels with every value at runtime.
