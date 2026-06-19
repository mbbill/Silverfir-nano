- Imports are resolved at instantiation by name against the runtime registry,
  binding host external functions into the same function-instance space as
  WebAssembly functions.

- Instantiation appends every instance kind to the store first, then
  initializes and module-binds them by reading each kind back out of the store
  as a per-module slice.

- Instantiation eagerly evaluates global, table, element, data, and start
  segments in spec order, with partial-success semantics where earlier applied
  segments persist when a later one fails.

## Facts

- 2024-02-09 (63de36fd) rationale: a resolved function import is built by
  cloning the exporting module's already-resolved function instance (which
  carries its module-instance back-pointer into the exporter) rather than
  synthesizing a fresh instance from the import's declared type, and is extended
  into the importer's own function range as a distinct duplicate store entry;
  tables, memories, and globals are created as fresh instances unconditionally
  (code).

- 2024-02-12 (83480c0c) conformance: resolving an import verifies the exported
  entity against the importer's declared type per spec 3.2.8 import subtyping —
  matching value/function type, table/memory limits within the imported bounds,
  and global mutability (the global value-type rule is itself mutability-branched,
  [[runtime-store/instantiation/global-import-check]]) (code).

- 2024-02-16 (01f2a6db) rationale: active element and data segments are applied
  at instantiation by evaluating their offset constexpr, bounds-checking against
  the target backing, copying the contents, and marking the segment dropped; all
  instances are computed before the store is mutated (code).

- 2024-03-07 (7c7e2a31) pitfall: at instantiation only locally-defined
  instances are bound to the module and have their globals/tables/memories/
  segments initialized — imported instances are already owned and bound by their
  exporting module, so the binding/init loops must filter on `!is_imported`, the
  negation of the natural-looking predicate (code).

- 2024-03-15 (34b6c16d) conformance: import type/size/mutability mismatches
  discovered at instantiation are reported with the dedicated `Unlinkable` error
  class (not `Invalid`), separating link-time failures from decode-time
  malformed and validate-time invalid so the three-way spec error partition is
  honored (code).

- 2024-03-15 (2577458b) pitfall: a memory import's minimum is checked against
  the live instantiated memory's current length, not the spec's declared
  minimum, because the providing memory may already have grown past its declared
  min by the time it is linked (code).

- 2025-06-22 (5ef58ebf) pitfall: an imported table's size compatibility must be
  checked against the exporting table's current runtime length, not its original
  declared minimum; a table that has grown past its declared min still satisfies
  an import whose required min exceeds that original spec min, and using the spec
  min wrongly rejects such a link (code).

- 2025-10-07 (f7febf40) rationale: carrying the declared type index lets import
  linking and `call_indirect` check function-type compatibility by isorecursive
  type equivalence rather than exact-Rc or pointwise-structural equality, so
  cross-module imports match when the exporting and importing function types are
  structurally equivalent within the exporting module's type context even though
  their module-local type indices differ (code).

- 2025-12-11 (8654e952) pitfall: baking callee entry pointers requires the
  callee to be compiled before the caller, so instantiation eagerly builds fast
  IR for all internal functions in definition order; this optimizes recursive
  and backward calls but mis-emits a forward call to a not-yet-built callee as
  `call_external`, which traps at runtime — the deficiency the later two-pass
  precompilation fixes (code).

## Moves

- 2024-03-15 (b22db5c9) replaced [[instantiation.alt/local-vector-processing]]:
  binding the module instance and evaluating segment initializers need the
  instances already live in the store, so cross-references resolve through the
  store's per-module slice rather than through pre-append local vectors (code).
