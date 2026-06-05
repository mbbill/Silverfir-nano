- Multiple module instances link at runtime: imports/exports resolve across
  live instances.
- Shared entities are reference-counted (`Rc`), not copied.
