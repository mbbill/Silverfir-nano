- The store-rooted structure — module instance, GC heap, constant-expression
  evaluation, machine value encodings, and the per-invocation native caches —
  is the JIT engine's runtime world; the interpreter keeps its own flat
  per-instance state and never constructs a store (`Store`).

- What the engines share is the entity model (memory/table/global/function
  instances, aliased across linked instances), the public value model, and
  the cross-instance link registry that resolves pooled handles back to their
  values and carries each engine's minted identities.

- Frame slots are the canonical source of truth for locals, spilled deep-stack
  values, call arguments and results, and call-link records; registers are
  execution caches over that frame state, which is what makes the cache,
  local-call frame reuse, and boundary rules safe.

- Generated native code is written into executable code buffers whose backing
  is requested from an OS-abstracted executable-memory allocator; an unreachable
  module's whole executable arena is freed wholesale on drop, with its stale
  per-buffer state purged; the OS may reuse the virtual address.

- Runtime sizing is a value the embedder builds and hands to an engine,
  which copies it into every instance it creates ([[runtime-config]]).
  Hosted targets default it to the pre-config numbers; the bare-metal
  target defaults to zeros that engine construction rejects by name.

- The Wasm operand/call stack and native dispatch context are cached per Store
  and reused across exported invocations. `eval` temporarily takes ownership;
  exact module/table/function-registry revisions control cached dispatch-view
  refresh (`take_native_stack_cache`, `prepare_for_invocation`).

- Host/imported functions are clonable, potentially capturing `Fn` callbacks
  bound at instantiation and stored behind single-threaded shared ownership;
  plain `HostFn` pointers remain source-compatible inputs. Callbacks receive a
  `Caller`, typed params, and a caller-owned multi-value result buffer
  (`HostCallback`, `Import::func`).

## Facts

- 2026-07-22 measurement: forwarding the benchmark's real capturing
  `clock_ms` callback through `HostCallback` enabled genuine CoreMark execution;
  matched runs scored sf-nano 38,540.60, Wasmtime Cranelift 37,814.33, Wasmer
  Cranelift 37,674.24, and V8 34,736.29. This is distinct from the 3.323 ms
  startup/coremark instantiation benchmark, which does not exercise the host
  callback (sourced).

- 2026-07-22 measurement: reusing the native stack/context and validating
  cached views by revision reduced regex-redux from 30.393 to 19.199 us (36.8%)
  because short exported calls no longer allocate/free the stack or rebuild
  type, memory, table, and function dispatch views each time; see
  [[runtime/invocation-cache]] (sourced).

- 2026-04-09 (c329abab) rationale: the native artifact strips MachineIR after
  emission because nothing at runtime reads it on the native backends; the one
  exception is the emulator (Reference) backend, which interprets MachineIR
  directly and so keeps it, and the ir-dump path, consulted before the strip
  while MachineIR is still resident (code).

- 2026-04-26 (8dc01387) pitfall: dropping a `Store` leaves dangling entries in
  the shared cross-store function registry; `Store::drop` tombstones its own
  slots (nulling the owner pointer) and dispatch-view refresh emits an INVALID
  function view for a dead slot rather than filtering it out, so surviving
  handles keep their registry-index alignment (code).

- 2026-07-28 (fec5adb5) statement: the store-rooted layer's JIT ownership was
  made explicit after the interpreter-only feature build measured it as dead
  code (42 warnings spanning the store, GC heap, const-expr evaluator, machine
  encodings, and registries' store-rooted API): the interpreter never
  constructs a `Store`, and its only reach into one is resolving a pooled GC
  entry minted by the JIT (code).

- 2026-07-28 rationale: per the author, the original global-index store
  (see the [[runtime-store]] alternative) was deliberately spec-accurate — one
  unified linear space where every instantiated module's entities carry unique
  store-wide indices and linking records an index. The deeper forces behind the
  2026-02-14 single-module split were JIT-era: generated code caches raw
  addresses of entity cells, which a growing unified arena relocates, and
  whole-store `&mut` ownership fought the native re-entry path; per-entity
  `Rc` aliasing was the escape hatch, and raw-pointer cross-store identity was
  the collateral cost (sourced).

- 2026-07-28 (0d984f61) statement: direction agreed for the storage redesign
  (docs/RUNTIME_WORLD.md): restore spec-accurate *indexing* without spec-naive
  *storage* — module instances live boxed in generational slots of a
  linker-owned world, identity is a slot id plus generation instead of a store
  pointer, and flat u32 address tables are scoped to exactly what the type
  system lets escape at runtime (func/extern/exn/GC — the link registry's
  existing inventory); entities keep link-time `Rc` aliasing. Alternatives
  weighed and set aside in discussion: a pure spec store (unified arena growth
  relocates entity cells under the native caches, and it has no reclamation
  story for instance churn), wasmtime-style append-until-store-death (the spec
  suite's module churn and embedded RAM budgets require module instances to
  die inside a living app), and the status quo (pointer identity requires the
  drop-scan protocol and unchecked resolution) (sourced).

- 2026-07-28 statement: open measured decision for the redesign: interpreter
  funcref slots carry local function indices consumed directly by the
  private-table indirect-call fast path; a canonical global funcaddr would
  insert a translation there, so that step is gated on the performance CI
  rather than decided in design (sourced).

## Moves

- 2026-02-14 (a8528504) replaced [[runtime-store]]: the store is a single-module
  model — the module instance owns its memories, tables, globals and function
  specs directly, dropping Rc<RefCell<>> reference counting and borrow overhead,
  inter-module linking (LinkableData/LinkableInstance), and the per-lookup
  HashMap (Vec + linear scan, since modules have few imports and this avoids a
  hashbrown dependency) (code).

- 2026-02-14 (a8528504) replaced [[runtime-store/host-functions]]: the
  host-function hook is a bare `fn` pointer rather than a `dyn` trait object
  because that is zero-alloc and no_std-friendly and carries multi-value results
  through a caller buffer; WASI and other host capabilities are provided by an
  external crate passing function pointers in, not built into the core (code).
