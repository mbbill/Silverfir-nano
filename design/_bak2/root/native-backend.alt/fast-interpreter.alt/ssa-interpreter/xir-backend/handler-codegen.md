- Handlers are not written by hand: a definitions file declares each
  operation once — its types, arity/signature pattern, and one canonical
  implementation (C for hot ops, Rust for the rest).
- Generation expands the definitions into the permuted wrappers (one per
  register assignment), the lookup tables, and the FFI declarations; manual
  FFI declarations do not exist.
- Semantics stay single-sourced in the canonical body; only the thin
  permuted wrappers multiply.
