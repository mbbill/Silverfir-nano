# Runtime World: retiring `*mut Store` cross-instance identity

**Status: PROPOSAL — not implemented. Stage 3 of the runtime-storage
refactor; needs design sign-off before any code moves.**

## Problem

The pre-JIT storage design survives in one load-bearing pattern: a raw
`*mut Store` is the identity token for anything that crosses an instance
boundary.

- `FunctionRegistryEntry { store: *mut Store, local_index }` and
  `RefRegistryEntry::Gc { store: *mut Store, gc_ref }` (`vm/link.rs`) hand
  raw pointers to whoever resolves the entry later.
- Correctness rests on `Drop for Store` linearly scanning both shared
  registries to null out entries (`vm/jit/store.rs`), and on every consumer
  honoring the poisoning contract at four separate `unsafe` deref sites —
  including one in the interpreter (`exec.rs`, ref type tests), which
  dereferences a pointer it did not create and cannot validate.
- The interpreter's authors rejected this design (see `FuncRefHost`'s doc)
  and built embedder delegation plus an `OpaqueInterpFunc` fallback instead
  — so the tree carries **two** cross-instance function-reference models,
  and the interpreter overloads `hostref` for published funcrefs. That
  overloading is why `ref.test (ref any)` answers differently per engine,
  and why `OpaqueInterpFunc` deliberately fails `ref.test (ref func)`.
- Of the crate's 589 `unsafe` tokens, 429 are in `vm/jit/runtime`, and
  roughly half of those sit on exactly this boundary (`context.rs` plus the
  two `entry.rs` marshalling files).

Goals, per the project owner: no hard embedder-API constraint; performance
and safety — ideally no `unsafe` on the cross-instance paths and no runtime
regression.

## Design

### One world, generational identity

A `RuntimeWorld` owns every live instance. An instance is addressed by a
generational id, never by pointer:

```rust
pub struct InstanceId { index: u32, generation: u32 }

pub(crate) enum WorldSlot {
    #[cfg(sf_jit)]
    Jit(Box<Store>),            // Box: address-stable for NativeContext
    #[cfg(sf_interp)]
    Interp(Box<InterpInstance>),
    Vacant { next_generation: u32 },
}

pub struct RuntimeWorld {
    slots: Vec<WorldSlot>,
    generations: Vec<u32>,
    registry: LinkRegistry,      // absorbed: the world IS the link registry
}
```

Registry entries store ids, not pointers:

```rust
enum RefRegistryEntry {
    I31(i32),
    Gc  { owner: InstanceId, gc_ref: GcRef },
    Func { owner: InstanceId, local_index: u32 },   // one model, both engines
    Exn(Rc<ExnInstance>),
}
```

Resolution is a safe bounds-plus-generation check:
`world.store(id) -> Option<&Store>`. A dangling reference is a generation
mismatch returning `None` — the same observable behavior the null-poisoning
gives today, without the `Drop` scan, without the contract, and without the
`unsafe` at any consumer. `Store::drop`'s registry walk is deleted; freeing
a slot bumps its generation.

### What the JIT keeps

`NativeContext` keeps its raw caches — memory bases, table views, global
cell pointers, the dispatch table — and the revision-epoch validation that
guards them. That part of the design is sound and hot-path-justified. Two
things change:

- `ctx.store` stays a raw pointer for the invoked instance (its `Box`
  address is stable inside the world slot), but every **cross**-instance
  resolution (`refresh_function_views`, GC type checks, linked calls) goes
  `id -> world` instead of dereferencing a stored pointer. The world
  pointer rides in `NativeContext` next to the store pointer; runtime
  helpers get `&mut RuntimeWorld` through the same accessor discipline as
  `current_store_mut` today.
- `FunctionRegistryEntry`'s `core::ptr::eq` store-identity comparison
  becomes an `InstanceId` equality.

Native-code ABI does not change; no emitted instruction is affected.

### What the interpreter gains

The interpreter registers its instances in the same world and mints
`Func { owner, local_index }` entries like the JIT does. Consequences:

- `published: Vec<(RefHandle, usize)>`, `OpaqueInterpFunc`, and the
  hostref overloading disappear. `FuncRefHost` narrows to what it is
  actually for — delegating calls to instances the embedder keeps *outside*
  this world — or is retired if no embedder needs that.
- The `ref.test` divergences dissolve: with provenance in the registry,
  one shared `ref_type_matches` (the deferred Stage 2d) serves both
  engines, and `hostref` answers `Any` uniformly.
- `ImportedFunction::Linked` stops being silently skipped at interp bind:
  a linked function is an `(owner, index)` the interpreter can drive
  through the world, either by predecoded call or by asking the owner
  engine to invoke.

### Embedder API

`Instance` becomes a handle: `{ world: Rc<RefCell<RuntimeWorld>>, id }` for
source compatibility, or — preferred — the embedder owns the
`RuntimeWorld` and passes `&mut` at call boundaries, making all mutation
explicit and keeping the crate free of hidden shared mutability:

```rust
let mut world = RuntimeWorld::new();
let a = world.instantiate(&engine, module_a, &imports)?;
let result = world.invoke(a, "run", &args)?;
```

The current `Instance::from_module` single-instance path remains as a thin
convenience that owns a private one-slot world, so simple embedders and
the CLI keep their shape. `LinkRegistry` as a public type is subsumed by
`RuntimeWorld`; `from_module_with_registry` becomes `world.instantiate`,
which also resolves the current asymmetry where the registry-aware
constructor exists only in JIT builds and its error type carries a
`JitInstance`.

### Encodings (deferred Stage 2c lands here)

With identity unified, `RefHandle` becomes a tagged index whose payloads
are registry indices or local function indices — never pointer-derived.
The three current encodings (host verbatim, TARGET32 wire form, interp
null-normalized slots) reduce to one slot encoding with an explicit 32-bit
wire mapping; the interpreter's null normalization (a slot means the same
thing on every width) becomes the rule rather than a third variant.

## What this deletes

- `impl Drop for Store` registry poisoning, and the O(registry) teardown.
- All four cross-instance `unsafe` deref sites (`context.rs:490`,
  `gc_type_check.rs` ×2, interp `exec.rs` Gc arm).
- `OpaqueInterpFunc`, the publication map, and the hostref overloading.
- The second funcref model, and with it the engine-visible `ref.test`
  divergences.

Remaining `unsafe` after this: the generated-code ABI itself (`preserved/
runtime_call` marshalling, signal handling, executable memory) and the
epoch-validated raw caches — the parts that are the JIT's actual job.

## Performance expectations

- Hot paths (in-instance execution, native dispatch, interp chain):
  untouched — same pointers, same epochs, same emitted code.
- Cross-instance resolution (linking, funcref publication, GC type tests
  across instances, linked calls): one bounds check + generation compare
  instead of a raw deref. These paths are already cold or already go
  through a registry borrow.
- Instance teardown gets cheaper: generation bump instead of scanning
  every registry entry.
- CI gates: performance-regression suite must stay flat; any measured
  regression on CoreMark/dispatch benchmarks blocks the stage.

## Migration plan

1. Introduce `RuntimeWorld` + `InstanceId` behind the existing public API
   (worlds of one instance; `LinkRegistry` internally becomes a world
   reference). No behavior change; all suites green.
2. Move the function registry to `Func { owner, local_index }`; JIT first
   (replaces `FunctionRegistryEntry`), then interpreter publication.
   Delete `Drop` poisoning when the last `*mut Store` entry is gone.
3. Unify ref type tests (the deferred Stage 2d) on registry provenance.
4. Collapse the encodings (Stage 2c) onto the tagged-index `RefHandle`.
5. Public API switch: `world.instantiate`/`invoke`, keeping
   `Instance::from_module` as the one-slot convenience.

Each step lands green on the full matrix (all engine configs, all targets,
both spec suites, WASI suite, performance gates).

## Open questions

1. `Rc<RefCell<RuntimeWorld>>` handle compatibility vs. explicit
   `&mut World` at the public API — the latter is safer and matches no_std
   single-threaded reality, but changes every embedder call site.
2. Should `FuncRefHost` survive as the escape hatch for out-of-world
   instances, or is in-world linking sufficient for every real embedder?
3. Entity sharing (`Rc<RefCell<MemBacking>>` etc.) is untouched by this
   proposal. A follow-up could move entities into world-owned arenas too
   (ids all the way down), which would also give the interpreter's shared
   tables/globals a chain-visible flat representation — measured decision,
   not assumed.
