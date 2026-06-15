- A `Runtime` holds the registry of loaded modules ([[runtime-store/module-registry]])
  and of host-provided external functions, keyed by (module name, function
  name); it is shared behind reference-counted interior mutability.

- Instantiation is separated from this registry into a `Store` that owns the
  live instances (functions, tables, memories, globals, elements, data) and
  the GC heap, all keyed by global store index.

- Each instance kind is stored in its own flat vector; a module instance
  records a contiguous index range per kind, making a module's instances a
  slice of the global store.

- A function instance carries a back-pointer to its owning module instance.

## Facts

- 2024-02-09 (63de36fd) rationale: a function instance needs a back-pointer to
  its `ModuleInst`, but the `ModuleInst` can only be built after the function
  instances exist (it records their store ranges); the circularity is broken by
  holding the back-pointer in a `OnceCell` set in a post-pass once the
  `ModuleInst` is constructed (diff).

- 2024-02-15 (3906283c) rationale: table, memory, and global backing storage is
  held behind `Rc<RefCell<...>>` so a resolved import aliases the exporter's
  mutable backing store — cloning the instance shares the same cell rather than
  duplicating the data (diff).

## Moves

- 2026-02-14 (a8528504) replaced by [[runtime]]: the store is a single-module
  model — the module instance owns its memories, tables, globals and function
  specs directly, dropping Rc<RefCell<>> reference counting and borrow overhead,
  inter-module linking (LinkableData/LinkableInstance), and the per-lookup
  HashMap (Vec + linear scan, since modules have few imports and this avoids a
  hashbrown dependency) (diff).
