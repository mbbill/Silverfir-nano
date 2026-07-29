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
  honoring the poisoning contract at each `unsafe` deref site. Those sites
  fall into two populations:
  - **GC-entry derefs**, roughly 13 of them in
    `jit/runtime/preserved/ops.rs`, all reached through exactly two
    resolvers — `resolve_struct_ref` and `resolve_array_ref`
    (`ops.rs:582-597`), each returning `(*mut Store, GcRef)` destructured
    straight out of `RefRegistryEntry::Gc` — plus `gc_type_check.rs:35`
    and the interpreter's own arm at `interpreter/exec.rs:1466`, which
    dereferences a pointer it did not create and cannot validate.
  - **Function-entry derefs**: `runtime_call/entry.rs:192`,
    `native_eval.rs:50`, `gc_type_check.rs:104`, and `context.rs:491`.

  The exact total is around nineteen and is not worth quoting precisely;
  what matters is that it is two populations with different owners, not
  the four sites this document originally claimed.
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
and safety, with no runtime regression.

On safety specifically, this design does **not** eliminate `unsafe` from
the cross-instance paths, and earlier drafts of this document were wrong to
imply it could. What it does is contain it: roughly nineteen unaudited
derefs, each depending on every consumer honoring the poisoning contract,
collapse into **one** audited primitive resting on **two** invariants,
checked in one place:

- **(a)** the generation matched at checkout, and
- **(b)** a checked-out slot cannot be freed.

Both are load-bearing. An earlier draft claimed a single precondition, which
was (a) alone, and (a) alone is weaker than what it replaces: the poisoning
contract re-validated on *every* use (`store.rs:252-254`, `ops.rs:610-611`),
whereas a checkout validates once and then holds a raw pointer for the whole
call. Invariant (b) is what buys that property back. With both stated, the
containment is the design's main win and it survives being stated
accurately.

## Design

### The framing: spec-accurate indexing, engine-real storage

The original store was deliberately spec-accurate — one unified linear space,
every instantiated module's entities carrying unique store-wide indices,
linking = record the index. It died in 2026-02 because the spec conflates two
roles a JIT must separate: a unified *index space* (good: context-free
identity) and a unified *storage arena* (fatal: growth relocates entity cells
while `NativeContext` holds their raw addresses, and the spec never frees
anything while real embedders drop and reload module instances). This design
restores the first role without the second:

- **Index layer, shared, spec-shaped**: flat `u32` address spaces —
  plural, one per escaping entity kind — for exactly what the type system
  lets escape at runtime as a first-class *reference*: functions,
  extern/host refs, exceptions, GC objects. That is `LinkRegistry`'s two
  **reference** arenas; the divergence kept the spec store's correct scope
  and only lost its safety. `LinkRegistry`'s third arena, the SIMD registry,
  is not a reference space at all — it is out-of-line payload storage for a
  value type that does not fit the 64-bit slot ABI — so it is carried across
  unchanged rather than restructured (see "Encodings" and "Filed
  separately").
- **Storage layer, owner-structured**: module instances stay whole, boxed in
  generational slots; entities stay owner-held and `Rc`-aliased at link time.
  Memories, tables, and globals get no global address space on purpose: wasm
  2.0 has no first-class values carrying them, so their sharing is fixed at
  instantiation and link-time aliasing is sufficient.
- **Generations, the price of mortality**: the addition every real engine
  needs once instances can die mid-app. Weighed and set aside: a pure spec
  store (relocation + no reclamation) and wasmtime-style
  append-until-store-death (the spec suite's module churn and embedded RAM
  budgets require instance death inside a living app).

### The state split: arenas stay side tables, the world owns instances

The single most important structural fact about the current design is one an
earlier draft of this document overlooked, and it constrains everything
below. `LinkRegistry`'s three arenas (`link.rs:198-205`) **own no
instances**. They are `Rc`-shared side tables, cloned into every `Store`
(`store.rs:30-36`) and every `InterpInstance` (`exec.rs:381`), which is
exactly why borrowing an arena can never overlap borrowing an instance —
and therefore why the deep runtime helpers compile at all.

A world that owned the instances *and* the arenas in one struct would
destroy that property: the container that must be reached would own the
reacher. Concretely, `resolve_array_ref` gets its arena through
`store.ref_entry_for_handle` from the `&Store` that `current_store(ctx)`
yields (`ops.rs:582-597`, `:28-34`), and `NativeContext` has nothing but
`store: *mut Store` and `current_module` (`context.rs:95-156`) — below
generated frames there is no world and no argument position to thread one
through. So the state is split by whether it owns instances:

- **The arenas keep the `LinkRegistry` shape.** The reference arena, the
  function address space, and the SIMD arena stay `Rc`-shared side tables
  cloned into every `Store` and `InterpInstance`. Their reachability is
  unchanged from today. What changes is only their *contents*: entries hold
  an `InstanceId` instead of a `*mut Store`. That is the point of the design,
  and it is independent of where the arena lives.
- **Only the instance table is world-owned**, because it is the only thing
  that owns instances.

```rust
pub struct InstanceId { index: u32, generation: u32 }

pub(crate) enum WorldSlot {
    #[cfg(sf_jit)]
    Jit(Box<Store>),            // Box: address-stable, and see "disjointness"
    #[cfg(sf_interp)]
    Interp(Box<InterpInstance>),
    Vacant,
}

pub(crate) struct FuncEntry { owner: InstanceId, local_index: u32 }

enum RefRegistryEntry {                 // in the Rc-shared arena, as today
    I31(i32),
    Gc { owner: InstanceId, gc_ref: GcRef },
    Exn(Rc<ExnInstance>),
}

/// The one world-owned structure, and the only route from an id to an
/// instance. The `RuntimeWorld` facade holds the sole STRONG reference;
/// instances hold `Weak` (see "Ownership direction").
pub(crate) struct InstanceTable(Rc<InstanceTableInner>);

struct InstanceTableInner {
    slots:       RefCell<Vec<WorldSlot>>,
    /// The single source of truth for slot generations, indexed by slot.
    generations: RefCell<Vec<u32>>,
    /// Live checkouts per slot. A count, not a flag — see "The seam".
    in_use:      RefCell<Vec<u32>>,
}

/// What every `Store` and `InterpInstance` carries, in the same place
/// `LinkRegistry` lives today. NON-owning, which is what keeps the graph
/// acyclic.
pub(crate) struct InstanceHandle {
    table:   Weak<InstanceTableInner>,
    /// This instance's own id. THIS FIELD IS WHAT `self as *mut Store`
    /// (`store.rs:225`, `:274`) AND `core::ptr::eq` (`context.rs:499`)
    /// BECOME. Both engines carry it; on the interpreter it replaces the
    /// operation `published`'s doc comment describes in prose
    /// (`exec.rs:391-394`).
    self_id: InstanceId,
}

impl InstanceHandle {
    /// The one resolution operation. `&self`, so it is callable from the
    /// `&Store` the deep helpers already hold.
    fn checkout(&self, id: InstanceId) -> Option<InstanceToken> {
        let table = self.table.upgrade()?;   // world gone -> None
        // bounds + generation check, in_use increment, &raw read,
        // borrows released before returning.
    }

    /// The two conversion primitives, both TOTAL and range-conditional,
    /// both resolving in the ARENA and consulting no slot. `self` is the
    /// handle of the instance owning the frame being read (`absolutize`) or
    /// written (`localize`). See "The conversion primitives" for the arms —
    /// this is the same pair, not a third declaration of it.
    fn absolutize(&self, value: u32) -> u32;
    fn localize(&self, value: u32) -> u32;
}
```

`checkout` takes `&self`, so a helper holding only `current_store(ctx)` can
reach a **second, runtime-chosen** instance — the one named by an
`InstanceId` it just read out of an arena. That is the operation the design
previously lacked: a token obtained before the call can only name an
instance chosen before the call.

**`InstanceToken` is engine-discriminated**, and this is not optional. An
earlier draft named `world.store(id) -> Option<&Store>` as the seam's one
resolution primitive, which cannot address a `WorldSlot::Interp` at all:
`Store` is JIT-only (`jit/store.rs:30`, imported under `#[cfg(sf_jit)]` at
`link.rs:26-29`), while the seam is claimed engine-neutral and is walked
through on the interpreter below. The token is therefore an enum over the
slot kinds, or carries per-engine accessors that return `None` for the other
kind. This holds regardless of the single-tier scope boundary below, because
`WorldSlot::Interp` exists in any interp build.
(`Instance::function_handle_at` returning `None` on its `Interp` arm,
`instance.rs:387-397`, is today's version of the same hole.)

The generation lives in `generations` and nowhere else: `checkout` validates
`id.generation` against `generations[id.index]`, so there is one place to
read it and one place to bump it. A dangling reference is a generation
mismatch returning `None` — the same observable behavior null-poisoning gives
today, without the `Drop` scan and without the contract. A slot whose
generation would wrap is **retired rather than reused** (saturate at
`u32::MAX`), which makes stale-id aliasing unreachable by construction for
one comparison on the free path and leaks at most one slot per 2^32 reuses.

`RuntimeWorld` is then a facade: it holds an `InstanceTable` clone plus
arena clones, and its `&mut` methods (`instantiate`, `free`, `invoke`) go
through the same table API the engine uses. It is not a container anything
has to reach into.

#### Why this is sound: disjointness

The soundness argument is about which memory a borrow covers, and it must be
stated rather than assumed:

- A `Ref`/`RefMut<Vec<WorldSlot>>` covers **the `Vec`'s buffer** — the slot
  discriminants and the `Box` pointers.
- An instance body is a **separate heap allocation** behind its `Box`.

Those two regions are disjoint, so a live `&mut Store` derived from
`ctx.store` never overlaps a borrow of the table. This is what makes the
whole scheme work, and it has two consequences that must be written down
because each is one edit away from silently breaking it:

1. **`checkout` obtains the pointer without ever forming a reference to the
   pointee** — `&raw const **b` / `&raw mut **b`, not `&**b as *const _`.
   Forming even a transient `&Store` inside the table borrow would put the
   instance body inside the borrow's provenance.
2. **`Vec<Box<Store>>` must never be flattened to `Vec<Store>`.** That single
   change would put instance bodies *inside* the `Vec` buffer, make the two
   regions overlap, and reintroduce the aliasing UB with nothing else in the
   design changing. This is a tripwire, not a preference.

For contrast, the shape that is unsound is the one an earlier draft implied:
an `Rc<RefCell<RuntimeWorld>>` whose `borrow_mut` yields a `RefMut` covering
a `Box<Store>` that `current_store_mut` has *also* handed out as `&mut Store`
through `ctx.store`. `RefCell` cannot detect that, because the second
reference never came in through the cell. Nothing panics; it is simply UB.
Decision 8 rejected `Rc<RefCell<World>>` for the embedder API because a
nested `borrow_mut` panics — a different and much friendlier failure.

#### Ownership direction, and why the back-reference is `Weak`

> The world owns the table. The table owns the instances. **The instances
> hold non-owning handles back.**

That is the same acyclicity today's tree has, and for the same reason: today
the arenas hold **non-owning** `*mut Store` (`link.rs:44-46`, `:154`) while
the owner is `JitInstance { store: Box<Store> }` (`jit/instance.rs:47-50`),
outside them. This design deletes the raw pointers and moves ownership into
the shared structure, so the back-reference has to become non-owning some
other way or the graph closes: `Rc<InstanceTableInner>` -> `Vec<WorldSlot>`
-> `Box<Store>` -> the store's own handle -> back. With N instances the
strong count would be N+1, and dropping the facade would reach N and stop.
Nothing collects it — `tracked_alloc`'s `Rc` is a reference count, not a
collector — so every instance, module, memory and compiled-code buffer would
leak. `Instance::from_module` keeps a private one-slot world, so that leak
would land on the CLI and on every in-core test, and `memprof` would report
every instance ever created as live.

**Tripwire: turning that `Weak` into an `Rc` reintroduces the leak silently**,
and also removes the only consumer of the `downgrade` that migration step 0
adds. It sits alongside the `Vec<Box<Store>>`-never-flattened tripwire above:
both are one-line edits that keep compiling.

`checkout` upgrading and finding the world gone returns `None`, which every
caller already handles — it is indistinguishable from a generation mismatch
and produces the same trap. "The world is gone" and "that instance is gone"
are the same answer to the same question, so this adds no error path.

#### The conversion primitives

**Two** operations, one per direction, each defined once. Naming both matters:
an earlier draft named only `localize` and described its counterpart in prose,
and two crossings then said "localize" for the outbound direction — where a
literal implementation is a **no-op**, because a value already in the local
form has no funcaddr to resolve. It would compile, run, and ship the local
form to the embedder, silently, into exactly the case `FuncRefHost`'s doc
comment warns about. A word that exists is easier to reach for than a phrase
that does not.

Both are **total** and range-conditional, returning a value rather than an
`Option`. That matters at the call sites, which all want "convert if
applicable, else pass through" — and it removes an `Option` whose `None` arm
was carrying an unstated meaning.

> `absolutize(store_owning_the_frame_READ, value) -> value`
> - `value` is **null** -> identity.
> - `value` is in the **absolute** range -> identity. It already names another
>   instance's function, and this store cannot speak for it.
> - `value` is in the **local** range -> resolve that store's local index to
>   its world funcaddr.
>
> `localize(store_owning_the_frame_WRITTEN, value) -> value`
> - `value` is **null** -> identity.
> - `value` is in the **local** range -> identity.
> - `value` is in the **absolute** range -> the local index if that store owns
>   it, **otherwise the value unchanged**.
>
> Both resolve **in the arena**, and both **run before `checkout`, never
> after, and consult no slot.**

Totality is not tidiness. Every absolutizing site reads storage whose contents
are mixed **by design** — the rule grants permission for the local form, it
does not require it, so a funcref owned by a third instance sits in those
slots already absolute. And the commonest input is neither form: **null** is
`u32::MAX` on the wire and `usize::MAX` on the host, outside both ranges, and
it is what every uninitialized table slot, every `ref.null func`, and every
unset funcref global produces. A partial definition would be asked to resolve
`u32::MAX` as a local index on the first null it met.

> **Rider on `localize`'s local-range identity arm, which is where it is
> easiest to get this wrong.** That arm is **forced** by the encoding limit —
> a local-range value is instance-relative, so there is nothing to check it
> against — and it is itself *default-local*: an absolute value that should
> have arrived and did not is indistinguishable from a local one that
> belongs. It is safe **only because `absolutize` is total at every outbound
> crossing.** A missed `absolutize` therefore does not degrade to a slow path;
> it degrades to **silent misdispatch on the far side**. Outbound
> completeness is load-bearing in a way inbound completeness is not.

The signatures are the discipline, not decoration. **Direction is checkable by
word; instance is checkable by argument.** That is what would have caught
`invoke_runtime_target` handing `owner_store` to a conversion operating on the
caller's frame — the type says "the store owning the frame", and `owner_store`
is not it.

The ordering is load-bearing, not stylistic. Running before `checkout`
keeps a self-reference off the slot entirely, which is what makes it legal
under the materialization invariant below *and* what makes the
vacant-during-initialization window work — during that window a `checkout`
would find the slot vacant and trap, while these never look.

`localize` has three consumers, and it is the same operation in all three:

1. **Shared-table self-dispatch** — an instance calling its own function
   through its own exported table (see "JIT side effect").
2. **The inbound boundary** — a reference arriving from outside is localized
   iff this instance owns it (see "The invariant is two-directional").
3. **Initialization** — a self-dispatch during the vacant window, where it is
   the only thing that works (see the migration plan, step 3).

**Same-instance calls are untouched.** A direct call within one instance never
crosses a frame owner, so it converts nothing and its emitted code is
unchanged — this is the hot path and it stays hot. The one same-instance case
that does pay is **indirect dispatch through the instance's own exported
table**, which stacks two costs on one shape: the dispatch regression (the
call leaves inline local dispatch, see "JIT side effect") *and* the
marshalling lookup, because the value is absolute in the table and absolute
across the call boundary. That stacking is recorded here because it changes
how the step-1 benchmark should be read, not because the two costs interact.

#### The concession

The `Rc` does not disappear, and this document should not imply it does.
What the world replaces is **raw-pointer identity for instances**: `Store`
addresses stop being the cross-instance name, `InstanceId` plus a generation
takes over, the `Drop` scan is deleted, and the poisoning contract every
consumer had to honor is gone. That is the win. A single `Rc` holding the
instance table, held strongly by the world and weakly by the instances, with
one resolution operation and its stated invariants, is what it costs.

### The seam: `checkout`

A shared borrow is enough for most resolution, but not for a call. A
cross-instance call puts two instances live on one call stack, and the
callee's execution cannot hold a borrow derived from the table while the
table must stay reachable for the next hop. The named primitive is
`InstanceTable::checkout(&self, id) -> Option<InstanceToken>`.

The bounds-plus-generation check is safe; the token carries the validated
raw pointer. **The table borrow is released at checkout** — it covered only
the `Vec` buffer for the length of the lookup — the token is what crosses
the call boundary, and a cross-instance call made from inside that callee
re-enters through a *fresh* checkout rather than nesting a borrow. This is
the one audited `unsafe` the containment statement above refers to, and it is
where the JIT's `*mut NativeContext` fits: that pointer becomes one instance
of the token, not the mechanism itself.

#### The safety invariant lives on the token

It is tempting to state the rule as "the table never yields a reference to an
instance". That is a useful **API rule** and should hold, but it is not the
safety property, and mistaking one for the other would leave the real
obligation unguarded. Aliasing is created by *materializing* a reference from
a token, and tokens are held by callers the table cannot see.

The obligation, therefore, is on every site that materializes:

> Materializing `&Store` / `&mut Store` from a token is **scoped**. It must
> not span a call, and it must not overlap another materialization of the
> same slot.

Two live tokens for one slot is not a bug to be prevented — it is a
**required state**, because decision 14's in-use count exists precisely so
that mutual recursion and self-recursion through a foreign hop can check out
one slot repeatedly. What must not overlap is the *materialization*, not the
tokens.

The tree already contains both shapes, which is why this is a discipline that
can be audited rather than a hope:

- **The load-bearing example.** `do_struct_set` (`ops.rs:628-658`) opens with
  a block that takes `current_store(ctx)`, extracts the field storage and
  resolves the reference, and **closes at `:643`** — before
  `value_from_machine(ctx, ...)` at `:645` and before the mutable
  materialization at `:646`. The shared borrow is deliberately dead by the
  time the mutable one is formed.
- **The anti-pattern shape.** `do_array_new_default` (`ops.rs:660-690`) takes
  `current_store_mut(ctx)` at `:665` and holds that `&mut Store` across the
  entire body. That is fine today because it touches exactly one store, and
  it is exactly the shape that becomes unsound the moment a second instance
  is reached from inside it.

The review question is "does this site hold a materialized reference across a
call that could reach another instance?", not "does it call the right
accessor".

**The audit scope is the invariant's own range, not a convenient subset.** An
earlier draft scoped it to "the ~13 GC sites", which cannot find the largest
violation in the tree: instantiation holds a materialized `&mut store` across
the entire start function (`jit/instance.rs:972-976`). An audit that lands
green while missing the longest-lived materialization in the engine is worse
than no audit, because it produces false confidence. The scope is therefore
**every site that materializes a reference from a token** — the GC resolvers,
the runtime-call path, the shared-table retag helper, and instantiation, which
is the one least likely to be recognised as belonging to "cross-instance
resolution" and so the one most likely to be skipped again.

`free`'s in-use check is narrower than this invariant and does not subsume
it: it protects only the slot being dropped.

Releasing the borrow is what makes re-entrancy work, and it is also what
gives up the static guarantee that used to make this safe for free:
`Instance::invoke(&mut self, ...)` (`instance.rs:233-238`) means the borrow
checker refuses to let an embedder drop an instance while a call into it is
on the stack. Once the table borrow is released, a later `free` — reached
through the world facade or through another checkout path — *could* drop a
slot a live token points into, and freeing is a named operation in this
design rather than a hypothetical. A generation bump does not help: the
token already holds the pointer, and nothing re-reads the generation. So
invariant (b) needs an enforcer:

- Each slot carries an in-use **count**, not a flag. Mutual recursion across
  instances and self-recursion through a foreign hop both legally check out
  the same slot more than once at a time, so a boolean would be wrong.
- `checkout` increments; `free` on a slot with a nonzero count returns an
  error instead of dropping the `Box`. One field, one branch, on a path that
  is already cold.
- The token is an **RAII guard** whose `Drop` decrements — not a value with a
  manual release. Traps, wasm exceptions and host errors all unwind through
  these frames, and a missed decrement would leave a slot permanently
  unfreeable on exactly the paths that are hardest to test.

**Requirement on the trap mechanism, recorded because RAII soundness depends
on it.** The guard-page signal handler does not jump past Rust frames: it
sets `trap_kind` in `NativeContext` and rewrites the signal frame's PC to the
faulting function's `body_local_error_label`, so JIT code returns normally
and `eval()` reads `trap_kind` afterwards to build the `WasmError`
(`trap_signal.rs:22-27`). Every intervening Rust frame therefore returns
normally and every `Drop` runs. If that redirect is ever changed to a
longjmp-style jump past frames, checkout counts leak silently and slots
become unfreeable — so this property must be preserved, or the count must be
replaced with something that survives a non-unwinding escape.

Two existing pieces of the tree are precedent, and they are precedent for
different halves — the plan should not overstate either:

- `InterpInstance::drive` (`exec.rs:1872-1888`) is a **true release**.
  `core::mem::take` moves the stack and return-stack out of `self`, so a
  re-entrant call finds them absent (`reentrant = stack.is_empty()`) and
  allocates its own pair. That is the discipline `checkout` generalizes.
- `jit/arch/common/eval.rs` shows where the current aliasing checkout
  **ends**, and why it is not sufficient on its own: the `NativeContext` is
  taken out of the store at `:109` and put back at `:186`, but the enclosing
  `&mut Store` borrow stays live across the whole native call and is used
  again at `:170-177`. It is take-and-restore *without* release. Making that
  boundary a real release is the work this design adds.

Two call sites need specific handling, and neither needs a disjoint-pair
accessor:

- The runtime-call type check (`runtime_call/entry.rs:201-226`) does not
  need `&mut` at all. Resolve the owner as `&Store` for the comparison, let
  that borrow end, then resolve again as `&mut Store` for the invoke; two
  shared borrows coexist on one slot without trouble. **`callee` must be
  re-derived after the second resolve** — it is currently obtained from
  `owner_store` through a raw pointer at `:196-200` and read at `:205`,
  `:220` and `:236`, so holding it across the gap would move the aliasing
  into `callee` instead of removing it.
- `do_array_copy` (`ops.rs:897-911`) needs nothing new. Its source borrow is
  already scoped to a block that closes at `:908` before the destination is
  taken at `:910`, with `to_vec()` at `:907` doing the decoupling. It is
  already resolve-use-release-resolve.

### What the JIT keeps

`NativeContext` keeps its raw caches — memory bases, table views, global
cell pointers, the dispatch table — and the revision-epoch validation that
guards them. That part of the design is sound and hot-path-justified. Three
things change:

- `ctx.store` stays a raw pointer for the invoked instance (its `Box`
  address is stable inside the world slot), but every **cross**-instance
  resolution (`refresh_function_views`, GC type checks, linked calls) goes
  `id -> world` instead of dereferencing a stored pointer.
- `FunctionRegistryEntry`'s `core::ptr::eq` store-identity comparison
  becomes an `InstanceId` equality.
- **The epoch keeps its invariant, and loses one of its events.** The
  invariant stands: every event that can invalidate a cached view must
  advance a revision the context snapshots. What changes is that instance
  death is no longer such an event, because after the re-indexing in
  "Encodings" the context caches no cross-instance data at all.

  Today the teardown edge is supplied accidentally — `Store::drop` calls
  `function_registry.borrow_mut()` (`store.rs:310`), which bumps the revision
  (`link.rs:72-75`), which is what `cached_views_are_current` compares at
  `context.rs:257`. That counter has exactly two readers in the crate, both
  inside `NativeContext` (`context.rs:237` snapshot, `:257` compare), and
  both guard the cross-instance view cache that the re-indexing eliminates.
  So `cached_function_registry_revision` and `Store::function_registry_revision`
  are **deleted outright** rather than replaced, and no world-owned
  counter takes their place.

  This is deliberate and it replaces an earlier draft that proposed a
  `RuntimeWorld::functions_revision`. That counter would have had no reader,
  and keeping it alive would have cost a duplicate: `NativeContext`'s only
  handle outside itself is `store: *mut Store` (`context.rs:116`), so
  reaching a world-owned counter would need either a raw world pointer in the
  hot context or an `Rc<Cell<u64>>` clone of the counter in the store — a
  second copy of a single number, with the desync risk that implies. Neither
  is worth paying for something nothing reads. What remains is
  `cached_module_revision` (already compared at `context.rs:256`) plus the
  per-table revisions, both local and both already correct.

  **This reasoning is about the counter and nothing else.** An earlier draft
  stated it as a general objection to shared handles, which was wrong and
  briefly foreclosed the shape the design actually needs: the `InstanceTable`
  clone in the store is a shared handle, is not a duplicate of anything, and
  is how the deep helpers reach a second instance at all (see "The state
  split").

  **If anyone ever re-widens `function_views` to hold cross-instance
  entries, the teardown edge comes back and so does the reachability problem
  above.** That is the condition under which this deletion stops being safe.

One note for whoever touches indirect dispatch later: a non-`LOCAL` view
branches to the runtime helper, which re-resolves through the registry
(`lower_module.rs:2055-2068`), and `fixed_call_entry_tables` caches native
entry addresses only for `LOCAL` views (`context.rs:580-594`). That routing
is what keeps a stale peer view a contract gap rather than a live defect.

The native-code ABI does not change. On emitted instructions the accurate
claim is narrower than earlier drafts stated: **no emitted instruction on the
performance-gate benchmarks is affected.** Outbound writes into a table
another instance can reach do gain a helper call — see "The invariant is
two-directional" — and that is a real change to emitted code for modules
that import or export a table.

### What the interpreter gains

The interpreter registers its instances in the same world and mints world
funcaddrs like the JIT does. Consequences:

- `published: Vec<(RefHandle, usize)>`, `OpaqueInterpFunc`, and the
  hostref overloading disappear. Localizing a reference the instance itself
  owns becomes an O(1) world lookup in place of today's linear scan
  (`exec.rs:3194`, `:3478`). `FuncRefHost` narrows to what it is actually
  for — delegating calls to instances the embedder deliberately keeps
  *outside* this world.
- The `ref.test` divergences dissolve: with provenance in the world, one
  shared `ref_type_matches` (the deferred Stage 2d) serves both engines, and
  `hostref` answers `Any` uniformly. This does not depend on the encoding
  choice — it depends on provenance existing at all.
- `ImportedFunction::Linked` gains an **identity**: it becomes an
  `(owner, index)` the interpreter can name and type-test, instead of being
  silently skipped at bind (`instance/interp_imports.rs:119`). **Driving the
  call is follow-up work, not part of this stage** — see below.

The seam applies here with no native frame anywhere on the stack. The
interpreter reaches it through host-throw payload validation:
`pending_from_error` takes `&mut self` (`exec.rs:1667`), runs from the driver
at `:2429`, and reaches a foreign store deref at `:1466` by way of
`host_value_matches_type` (`:1537`) and `ref_handle_matches_type` (`:1433`).
`drive`'s take-and-restore protocol already makes per-instance *execution
state* re-entrancy-safe, so `WorldSlot::Interp` inherits a working answer for
buffers; what the world adds is the *borrow* discipline, which the checkout
token supplies.

Stated plainly, because it is the honest form of the claim: in-world linking
**relocates** the cross-instance raw pointer from the embedder into the
engine rather than eliminating it. Today that pointer is the spectest
runner's `HashMap<String, *mut Instance>`, whose validity rests on a comment
about single-threaded test execution (`wast_test_runner.rs:269-282`). After
this change it is one generation-checked primitive the engine owns.

#### Scope boundary: a world is single-tier

**A `RuntimeWorld` holds instances of one engine. `instantiate` records the
tier of the first instance and rejects a later one of the other tier with a
named error.** This is a deliberate scope boundary, written down because
nothing else in the design prevents a mixed world and every other claim here
quietly assumes one.

The reason it needs saying is that unifying the funcref model changes what a
mixed world *looks* like. The mixed capability itself is not new: the tree
already has arms for both tiers in `Instance::from_module_with_registry`
(`instance.rs:106-135`), so a JIT instance and an interp instance can already
share one `LinkRegistry` today. What today's tree does in that case is
exactly the two mechanisms this design deletes — the interpreter mints an
`OpaqueInterpFunc`, which "Deliberately matches NOTHING, `func` included"
because "nothing can call it or say which function it names"
(`gc_type_check.rs:43-51`), or the embedder forwards through
`FuncRefHost::invoke`. **What is new is that a unified funcaddr space would
make such a reference look callable** while every consumer of it is JIT-typed
end to end: `call_runtime_by_handle` derefs into `&mut Store`
(`entry.rs:190-192`) and hands it to `invoke_runtime_target(..., owner_store:
&mut Store, callee: &FunctionInst, ...)` (`entry.rs:239-245`).

The cost of the boundary is zero on anything that exists: the spectest
harness builds every registry uniform-tier (one `Engine`, one
`LinkRegistry`, a single tier per run), and an embedder that genuinely wants
the two engines to call each other keeps precisely the mechanism it uses now
— two worlds, one per engine, forwarding between them through `FuncRefHost`.

That is also `FuncRefHost`'s **second** justification, and the plan should
carry both: it is the escape hatch for instances the embedder deliberately
keeps outside the world, *and* it is the cross-engine hop.

Making a foreign funcaddr callable across engines is named follow-up work,
not part of this stage. It needs a call path that is not JIT-typed, a
value-marshalling boundary between the two engines' raw slot forms, and its
own entry under "Performance expectations" — it is a new engine capability
rather than a refactor, and folding it into a stage whose goal is containing
`unsafe` without regression is how a stage stops being reviewable.

### Embedder API

The embedder-facing call keeps `&mut RuntimeWorld`. The engine-internal call
boundary does not: `world.invoke` checks out through the instance table,
lets its own `&mut` borrow of the world end, and runs the callee against the
token — so a cross-instance call made *from inside* that callee re-enters
through a fresh checkout on the shared table rather than nesting a borrow,
and never needs the `&mut RuntimeWorld` the embedder is holding. Both shapes
considered earlier fail without this: a held `&mut World` cannot yield a
nested `world.invoke`, and `Rc<RefCell<World>>` panics on the nested
`borrow_mut`.

```rust
let mut world = RuntimeWorld::new();
let a = world.instantiate(&engine, module_a, &imports)?;
let b = world.instantiate(&engine, module_b, &imports_naming(a))?;

// Flat case.
let result = world.invoke(a, "run", &args)?;

// Nested case — this is the one that has to work. `run_b` calls a funcref
// owned by `a`; the runtime resolves it to `(a, local_index)`, checks out
// `a` while `b` is mid-execution, and returns through `b` afterwards.
// Neither checkout nests: `b`'s world borrow ended when `b` was checked out.
let result = world.invoke(b, "run_b", &args)?;
```

`Instance::from_module` remains as a thin convenience owning a private
one-slot world, so simple embedders and the CLI keep their shape.
`LinkRegistry` as a public type is subsumed by `RuntimeWorld`, and
`from_module_with_registry` becomes `world.instantiate`.

That also resolves the asymmetry where the registry-aware constructor exists
only in JIT builds and its error type carries a `JitInstance` — but by
replacing the error rather than by the switch alone, which an earlier draft
asserted without saying how. `InstanceInstantiationError::Partial` carries an
`InstanceId` into a still-occupied slot; see migration step 8, which also
records why occupied-on-failure is a requirement rather than a choice.

This is also what makes retiring `FuncRefHost` defensible, and the reason is
sharper than "no embedder needs it". `FuncRefHost`'s objection is conditional
(`exec.rs:114-120`): "an engine-side registry could only manage it with a raw
pointer into storage **the embedder is free to move**." A `RuntimeWorld` owns
that storage — boxed instances in its own slots — so the antecedent is false
and in-world linking is sound exactly where the embedder's raw-pointer map is
not. `FuncRefHost` survives as the documented escape hatch for instances an
embedder deliberately keeps outside the world.

### Encodings (deferred Stage 2c lands here)

There are **two** address spaces, not one, and the distinction is
load-bearing:

- **The world function address space.** `RuntimeWorld::functions`, resolved
  to `(owner, local_index)`. *Foreign* funcrefs index this; local ones do
  not, for the reason in the next subsection.
- **The pooled reference arena.** `I31` / `Gc` / `Exn`, reached through
  `RefHandle::from_pool_index`, which sets `SPECIAL_TAG | pool_payload_tag`
  (`value.rs:95-97`) and therefore reports `is_special()`.

Keeping them separate is what lets **both** funcref forms stay untagged, and
that matters because four fast paths reject `is_special()` handles by
construction: `function_entry_for_handle` (`store.rs:250`),
`function_view_for_fixed_handle` (`context.rs:620-622`), the interpreter's
"not one of mine" test (`exec.rs:3186`, `:3471`), and `check_ref_type_match`
taking the pooled arm before `check_func_ref_type` is ever reached
(`gc_type_check.rs:26` vs `:63`). Folding functions into the pooled arena
would make every funcref pooled, drop every funcref in a `FixedLocalOnly`
table off the JIT's fixed-dispatch path, and cost the no-regression gate
directly. Neither funcref form sets any special-space bit, so the pooled
predicates keep meaning exactly what they mean today.

#### Provenance lives in the encoding

Two address spaces are necessary but not sufficient. The remaining question
is what an *untagged* funcref payload means, and the two engines answer it
differently today: in the JIT it is a shared-registry index — a
cross-instance address already, since `SharedFunctionRegistry` is cloned into
every linked store (`link.rs:51-54`) and `function_views` is indexed by
`handle.encoded()` into it (`context.rs:623`) — while in the interpreter it
is a *local* function index (`exec.rs:57-60`), consumed directly by generated
assembly. They get away with the disagreement only because the interpreter
does not participate in the function registry at all (`link.rs:9-12`).
Unifying the model is what forces a choice, and a context-free funcaddr in
the untagged form is the wrong one: it is a small plain integer that passes
every guard the interpreter's fast paths apply.

The rule is therefore:

- A funcref naming a function **of the instance holding it** is in the
  **local form**: its value is that function's local index — what
  `exec.rs:57-60` already documents and what every interpreter fast path
  consumes.
- A funcref naming a function **in any other instance**, or held by anything
  that is not an instance in this world, is in the **absolute form**: its
  value encodes a world funcaddr, which `RuntimeWorld::functions` resolves to
  `(owner, local_index)`.

"The instance holding it" means an instance **in this world**. A handle held
by the embedder, or in transit through embedder-owned memory, has no local
form at all, because "local" is meaningful only relative to an instance — see
the public-API boundary under "The invariant is two-directional".

The two forms are told apart by *range*, not by a tag bit — see below. The
words "retag" and "retagging" survive in this document as the name of the
conversion at a transfer boundary; no bit is being set.

**The limit of what the ranges can tell you.** They distinguish local from
absolute, and nothing more. **"Local" is instance-relative:** a value in the
local range says that it names *some* instance's own function; it never says
*whose*. A reader that finds a local-range value in a container it did not
write has no way to recover the writer's identity from the value — the
information is not there to recover.

That is why the absolute form has to **begin at the container boundary**
rather than being reconstructed at the read, and it is the whole reason the
normalization rule below is stated over containers instead of being replaced
by a cleverer check at the point of use. Any proposal to skip a conversion
and "work out the owner later" runs into this limit; there is no later.

No storage class is privileged for the *provenance* question. Operand slots,
private tables, shared tables and globals all hold whichever form the
reference is — which is what lets the invariant reach `Op::CallRef`, whose
callee comes from an operand slot with no table anywhere in the path
(`exec.rs:3182`, `:3231`) and, on the local arm, no type check at all. (The
separate *normalization* rule for shared containers, below, is about which
form a container may hold, not about how a form is recognized.)

**Scope: this rule governs reference-typed slots only.** The 64-bit slot ABI
also carries values that are not `RefHandle`s at all. A v128 rides as an
index into the shared SIMD arena — `value_to_raw_in_store` sends
`Value::V128` to `intern_v128` and `Value::Ref` to `from_ref` into the same
raw slot (`value_encoding.rs:124-135`) — so a v128 slot holds a small integer
that is disambiguated by static type. The local/absolute range rule below is
a statement about reference-typed slots and says nothing about that case;
in particular, a v128 arena index may fall in either funcref range and means
neither.

**There is no provenance bit. The two forms occupy disjoint ranges of one
value space.** Define a single width-independent constant:

```rust
const FUNCADDR_TOP: usize = (1 << 28) - 2;
```

- A **local** funcref encodes as its local function index, allocating upward
  from 0 — unchanged from today.
- A **foreign** funcref encodes as `FUNCADDR_TOP - funcaddr`, allocating
  downward from the top.

Every value in play is therefore `<= (1 << 28) - 2`, and that one fact makes
all four masks the identity function:

| mask | value | result |
| --- | --- | --- |
| 64-bit `payload()`, `(1<<60)-1` (`value.rs:28-30`) | `<= 2^28-2` | identity |
| 32-bit `payload()`, `(1<<28)-1` (`value.rs:21-30`) | `<= 2^28-2` | identity |
| TARGET32 encode, `(1<<28)-1` (`value_encoding.rs:79-81`) | `<= 2^28-2` | identity |
| TARGET32 decode, `(1<<28)-1` (`value_encoding.rs:104-106`) | `<= 2^28-2` | identity |

This is what a bit-tagged scheme could not do. A tag in bits 30-31 is free in
the 32-bit *word* but unreachable *through the encoders*: on 32-bit
`payload()` would strip it and hand a clean, in-range index to
`function_entry_for_handle` (`store.rs:244-255`), which returns a live entry
for the wrong function, and the TARGET32 round trip would mask it out in both
directions so the generated-code test could never fire.

`is_special()` stays **false** for both forms at both widths (bit 60 / bit 28
clear), which is correct rather than a compromise: both *are* plain funcrefs,
so `is_pooled()`, `is_host()` and `pooled_index()` all keep exactly their
current meanings.

One consequence looks alarming and is not, so it is written down rather than
left to be rediscovered: on 32-bit, `pool_payload_tag()` is `1 << 27`
(`value.rs:32-34`), so **any funcref value `>= 2^27` sets that bit**. It is
harmless because every pooled accessor gates on `is_special()` first —
`is_pooled()` is `self.is_special() && (self.0 & pool_payload_tag()) != 0`
(`value.rs:82-84`) — and `is_special()` is false for a funcref. The bit is
set and never consulted.

`(1 << 28) - 1` is reserved and never minted, as a plain guard rail at the
top of the range. (It is *not* adjacent to the null sentinel: null is
`u32::MAX` on the wire and `usize::MAX` on the host, far above this range.)

Add the `ref_to_machine_raw`/`machine_raw_to_ref` round trip over **both
range endpoints at both widths** to the migration step's test list. That is
where the two widths are reconciled, and a future edit to any of the four
mask constants would otherwise break this scheme silently.

#### The invariant is two-directional

Tagging on arrival is only half of it. A funcref of *this* instance that
escapes to another one must be **retagged at the transfer boundary**, or the
receiver reads an untagged payload as one of its own local indices — the same
silent wrong-function dispatch, in the opposite direction.

The rule is structural, and it is stated by **property**, over the closed set
of places a local form is permitted — not as a list of the places it is
forbidden. Earlier drafts did the latter twice and the enumeration was
incomplete both times.

> A reference in instance `I`'s **local** form may appear only in storage
> observable solely by `I`:
>
> 1. a reference-typed **operand or frame slot of an activation of `I`**, or
> 2. a container **statically proven unreachable by any instance other than
>    `I`** — a private table or a private global.
>
> **Everywhere else it is always absolute.** Conversion happens at the
> crossing, in every direction.

Both clauses are indexed by the **owning instance**, and that is the whole
content of the rule: **`I`'s local form lives only where only `I` can see
it.**

An earlier draft wrote clause (1) as "a reference-typed operand or frame
slot", unqualified, and that omission was structural rather than clerical.
Both endpoints of a cross-instance wasm call are frame slots, so a value
moved from a caller's frame to a callee's frame went from a permitted-local
container to a permitted-local container **belonging to a different
instance**, with no container in the path the rule forbade. The rule reasoned
about *where a value sits* and never about *who is looking at it* — which is
the same gap the encoding limit above describes from the other side, since a
local-range value never says *whose* local index it is. Binding the clauses
to an owner makes the call boundary a conversion site by construction.

**"May", not "is".** A foreign reference has no local form at all, so an
operand slot legitimately holds absolute values much of the time. The rule
grants permission for the local form in two places; it does not require it
anywhere.

#### The enforcement invariant

The rule above says where each form may live. This says how it is kept true,
and it is what lets the cross-instance call be handled without any bespoke
conversion code:

> A **frame slot** holds its **owner's** local form.
> A **`Value`** holds the **absolute** form.
> `absolutize` and `localize` convert between them **against the store that
> owns the frame being read or written** — an argument the call site must
> pass, and which is *not* always the store that owns the callee.

That last clause is the one with teeth. `invoke_runtime_target` reads the
**caller's** frame at `entry.rs:260-275` and writes it at `:337-347`, but
passes `owner_store` — the **callee's** store — to both conversions. Today
that is harmless because the `Ref` arms ignore the store; the moment they
become instance-relative it is a silent misdispatch in both directions, and
in the absolute form outbound, which is then resolved confidently everywhere
it subsequently travels. The fix is to pass the caller's store at those two
sites, not to add a conversion.

Two consequences worth stating, because they make the change smaller than it
looks:

- **The callee side is already correct.** `runtime::eval` reaches
  `arch/common/eval.rs:92-95`, which writes the *callee's* frame with the
  *callee's* store — the frame's owner — so it becomes right for free once
  the arms are instance-relative. `native_eval`'s `Linked` arm
  (`native_eval.rs:46-60`) likewise passes `&[Value]` and the new callee's
  store, and needs no conversion of its own: under `Value` = absolute it is
  already carrying the neutral form.
- **Crossings 1, 2 and 4 stop being three arms** and become three instances
  of this one invariant, since all three are `Value` boundaries. Crossings 3
  and 5 remain explicit because they carry raw slots and never form a
  `Value`.

#### Why the rule is stated this way round

The set of containers that can hold the local form is **closed** and
statically decidable. The set that must hold the absolute form is
open-ended — earlier drafts of this document missed the public API, both host
boundaries, `FuncRefHost::invoke`, shared globals, and GC heap fields, each
time by omission from a list rather than by disagreement about the principle.

The direction matters because the two failure modes are **asymmetric**:

- Absolute in a container that was really private: an unnecessary
  `localize`. **Slower, correct.**
- Local in a container that was really shared: **silent wrong-function
  dispatch.**

Default-absolute fails safe; default-local fails unsafe. A rule whose
omissions cost performance can survive being incomplete, which — given the
history above — is the property this one needs. It also dissolves the
producer question instead of answering it case by case: "which form do I
mint?" becomes "is my destination an operand slot or a statically private
container?", which is answerable at every producer without a judgement call
about what the container is *for*.

Two consequences that are easy to miss:

- **`module.function_handles[]` is not a container the rule ranges over.** It
  is a lookup table holding mixed forms by design (see below), consulted to
  *produce* values rather than to store them for another instance to read.
- **Passive element segments hold the absolute form**, since nothing proves
  them private. So `table.init` into a *private* table is an **inbound**
  conversion — absolute to local — which is the opposite direction from the
  active-segment case. The rule gets this right without a special arm, which
  is the point of stating it by property.

This does not contradict the reasoning that rejected a storage-class
invariant for `call_ref`. An operand slot's provenance is genuinely dynamic
and must ride in the encoding; a *container's* reachability is static. Both
mechanisms are needed and they are not alternatives.

#### Why the local form cannot escape: the three channels

The rule is only worth stating by property if the property is exhaustive. A
value leaves the region only `I` observes by exactly three channels:

1. **Aliasing** — into a container another instance can reach. Clause (2).
2. **Control transfer** — into another instance's activation. Clause (1),
   newly, and the channel the previous formulation had no term for at all.
3. **Escaping the engine** — to the embedder or a host, which is no
   instance's storage. The crossings list.

Five candidate fourth channels were tried and refuted, which is recorded
because "we could not think of another" is weaker than showing the search:

- **A shared `Module`.** Refuted by value ownership: `Store` holds
  `module: ModuleInst` by value (`jit/store.rs:30-31`), and `ModuleInst` owns
  its `function_handles: Vec<RefHandle>` by value (`jit/entities.rs:182-186`).
  No two instances share one.
- **Unwinding** (traps, wasm exceptions). Covered twice over — an exception
  payload is a `Vec<Value>`, hence absolute by representation (below), and the
  frames it unwinds through belong to a single instance each.
- **Persistence.** The engine serializes no reference; there is no channel.
- **Reference-typed linear memory.** Wasm has none: memory holds bytes, and a
  reference has no byte representation to store there.
- **Arena residency** — a reference held in the shared ref arena, e.g. an
  `Exn` payload. This one is real, and it is **not** covered by clause (2)'s
  static fact: the arena is shared **unconditionally**, so no analysis makes
  it private. It is covered instead by `Value` = absolute, since everything
  the arena holds is a `Value`. Worth distinguishing precisely: clause (2)
  is about containers whose sharing is *decidable*; the arena's sharing is
  *given*.

#### Absolute by representation

`Value` = absolute is not only a convention at the boundaries — it decides
two containers outright, because their payloads *are* `Value`s:

- **Exception payloads:** `ExnInstance { fields: collections::Vec<Value> }`
  (`link.rs:34-40`).
- **GC struct fields and array elements:** `GcStruct { fields:
  collections::Vec<Value> }` (`gc_heap.rs:27-36`).

So both hold the absolute form by representation rather than by a rule
someone must remember. This independently confirms the GC conclusion reached
earlier by derivation, and it closes a question the JIT side left open: the
exception-payload sites need no engine-specific arm, because there is no
representation in which they could hold a local form.

The ownership form of the rule needs the same qualifier: "of the instance
holding it" means **an instance in this world**. When the holder is the
embedder, or the value is in transit through embedder-owned memory, the
absolute form applies. Read without that qualifier, the rule decides the
public boundary backwards — `Instance::function_handle_at` is called on the
owner, so the function literally *is* "of the instance holding it", and the
handle would be minted local and then installed into a different instance's
imports, where it reads as that instance's local index. That is the silent
wrong-function dispatch the range encoding exists to prevent, and it would be
a regression: today's handle is unambiguously absolute, because
`register_local_function` returns `RefHandle::new(registry.len())`
(`store.rs:224-238`), an index into a registry cloned into every linked
store.

**This is a conversion site, not a documentation change.** The instance's
`module.function_handles[]` array holds **mixed forms by design** — it is a
lookup table, not a container the rule ranges over. What each producer reads
out of it depends on **where the value is going**, and an earlier draft got
this wrong by keying on the producing *operation* instead:

- `do_ref_func` (`ops.rs:255-261`) produces into an **operand slot** ->
  local. It is the only one of the three earlier drafts listed that was
  right.
- **Element-segment materialization** produces into a table that may be
  **imported**: `materialize_element_init` resolves through
  `store.module().function_handle(idx)` (`jit/instance.rs:1444-1458`) and the
  `Element::Active` arm copies the result into `store.table_mut(*table_index)`
  with **no import filter** (`:907-925`). The table-*initializer* loop one
  above it does filter — `if table.is_import() { continue; }` (`:881-883`) —
  so the same file treats imports as special in one loop and not the other.
  This producer must consult its destination.
- **Const-expression `ref.func`** produces into a global cell that may be
  **exported**: `StoreResolver::func_ref` returns
  `module.function_handle(func_idx)` (`expr_eval.rs:20-28`), written into the
  cell at `jit/instance.rs:871-878` — and for an exported global that cell is
  the one importers bind through
  `GlobalInst::from_shared(state.global.clone_shared_cell(), ..)` (`:745-749`).
  Same fix.
- The public `function_handle_at` (`jit/instance.rs:1251-1253`) looks up or
  mints the **world funcaddr** and returns that.

**`linking.wast:592-611` is the executable case for the element-segment
half, and it fails silently rather than loudly.** `$Ms` (`:579-589`) exports
its table and has two functions of the *same* type `$t = (func (result i32))`
— local 0 is `get memory[0]`, local 1 is `get table[0]`, whose body is
`call_indirect (type $t) (i32.const 0)` through its own table. The trapping
module (`:593-606`) imports that table and writes `(elem (i32.const 0) $f)`
where `$f` is *its* local index 0. If the local form reaches the table, `$Ms`
reads `0`, finds it inside its own local range, and dispatches its own
function 0 — and the type check cannot catch it, because both functions are
`(type $t)`. `(invoke $Ms "get table[0]")` then returns **104**, the value
asserted for `get memory[0]` at `:609`, instead of the `0xdead` asserted at
`:610`. Signature-compatible, silent, wrong function, in a test that passes
today. It is named as a regression case in migration step 3.

**The static fact is needed at instantiation, before lowering exists.** Both
producers above run during instantiation, so the per-container "reachable by
another instance" fact cannot be a lowering-time input only. It is derivable
from the parsed module (`is_import() || !export_names().is_empty()`), so this
is a sequencing requirement rather than an obstacle.

#### Keeping the static fact true: the accessor surface

Clause (2) grants the local form on the strength of a container being
*statically proven* unreachable. The public API can make that proof false at
runtime with no `unsafe` on the embedder's side, so the proof has to be
defended rather than asserted.

The hole: `shared_table_state_at` / `shared_global_state_at`
(`instance.rs:483-508`) index tables and globals with **no export or import
filter** — the JIT arm is `store.module().tables.get(idx).cloned()`
(`jit/instance.rs:1265-1289`) — and the clone aliases, because `TableInst`
derives `Clone` over `elements: Rc<RefCell<Vec<RefHandle>>>`
(`entities.rs:169-176`) and `GlobalInst` over `cell: Rc<GlobalCell>`. From
there `Import::table_with_state` binds it into a peer through
`TableInst::from_shared`. So a table with no imports and no exports — private
by the fact, inline store, local forms permitted — can be read by a second
instance that resolves those values against *its* function list. Same silent
misdispatch as `linking.wast:592-611`, through supported public API.

**This is created by the design.** Today's payloads are shared-registry
indices, so the alias is harmless, and coherence is already handled:
`elements_mut` bumps a shared revision unconditionally, with a comment saying
exactly why (`entities.rs:172-175`). Only the local/absolute rule makes it
unsound. (`FixedLocalOnly` is *not* separately broken today for the same
reason — the revision bump invalidates the cached views. But it reads the same
`private_local` predicate, so both consumers must move together.)

**The fix: the sharing accessors' precondition becomes the stored fact.**
They return `Some` iff the container is already classified reachable, `None`
otherwise. The API and the fact become one predicate. This costs nothing
in-tree — both callers already pass export-derived indices
(`wast_test_runner.rs:1592-1595`, `:1462-1465`) — and it is arguably right
independently of this design, since handing out a container the module does
not export is the embedder reaching past its own declared interface.

**Filtering those two is necessary and not sufficient**, because the sharing
API is not the only door. A second, entirely `pub` route exists:
`Instance::as_jit` -> `JitInstance::store` (`jit/instance.rs:1082`) ->
`Store::table(idx)` (`jit/store.rs:100`) -> `&TableInst` ->
`clone_shared_elements` (`entities.rs:221`) -> `TableInst::from_shared` ->
`ImportedTableState`, whose fields are `pub` (`imports.rs:25-28`). Globals
likewise via `Store::global` and `clone_shared_cell`. So:

- **`JitInstance::store`/`store_mut` become `pub(crate)`.** There is exactly
  one out-of-crate caller across all workspace members
  (`wast_test_runner.rs:2458`), and it needs one boolean: it matches on
  `functions[func_index]` with **three** arms — `Local` asserts
  `spec.has_native_code()`, `Host` and `Linked` both panic. Serve it with
  `JitInstance::function_has_native_code(idx) -> Option<bool>`, where `None`
  absorbs `Host`, `Linked` and out-of-range. The diagnostic gets slightly
  blunter — one `None` where there were two distinct panics — which is a fair
  price rather than an improvement, and worth saying so plainly.
- **The interpreter's accessors are filtered identically.**
  `InterpInstance::table_state_at` (`exec.rs:1715`) and `global_state_at`
  (`:1413`) are `pub` and bypass the wrapper entirely through `as_interp`.
  This is the more important half, not the lesser one: under the single-tier
  scope boundary the all-interpreter world is a *common* configuration, so
  filtering only the JIT side would fix the rarer engine and leave the
  commoner one open. It costs nothing — no out-of-crate caller, and the CLI's
  `as_interp` use is statistics only (`sf-nano-cli/src/main.rs:334-344`).

**The fact must be stored, because it cannot be re-derived.** `TableInst` is
`{ elements, revision, limits, .. }` with no import or export information, so
an accessor holding only `&TableInst` has nothing to test. Both engines keep
it per container, on the precedent already in the struct —
`ModuleInst { .., pub(crate) table_dispatch_modes: Vec<TableDispatchMode>, .. }`
(`jit/entities.rs:191`), computed once at instantiation:

- **JIT:** `table_reachable: Vec<bool>` and `global_reachable: Vec<bool>` on
  `ModuleInst`, computed beside the dispatch modes.
- **Interpreter:** the same two beside `tables` and `shared_globals`. This
  *completes* a computation it already half-does — `if
  g.export_names().is_empty()` at `exec.rs:819-826` — subject to the rider
  that it must key on `is_import() || exported` rather than exports alone.

> **Tripwire, ongoing rather than one-time — beside
> `Vec<Box<Store>>`-never-flattened and `Weak`-never-`Rc`.** No `pub` API
> yields an aliasing handle to a container the stored fact marks private. That
> is a property of the **audited accessor surface**: adding a `pub` accessor
> that hands out an entity — a `TableInst`, a `GlobalInst`, or anything
> containing one — re-opens the hole silently, with no compile error. It is
> checkable rather than aspirational because the step-3 audit enumerates that
> surface mechanically (below).

**Honest scope.** This makes the fact unfalsifiable through the crate's public
API *as it stands*. It does not make it unfalsifiable in principle — which is
why the tripwire is an ongoing obligation.

**Ordering: all of this lands in step 3, with the fact itself.** The
narrowing, the two accessor filters, and `function_has_native_code` are
**prerequisites for clause (2) being true**, not consequences of it. Leaving
`store()` public through steps 3-8 would be exactly the intermediate state
this plan forbids elsewhere: a step boundary where a stated invariant does not
hold.

Import binding genuinely needs no change: `jit/instance.rs:861-867` installs
a `linked_handle()` unchanged, which is right precisely because the value
arrives absolute and stays absolute. The conversion at **this** boundary is
one-directional — `function_handle_at` only returns, so nothing comes back
in through it. That is a property of this one API and **not** of the boundary
class; an earlier draft generalized it and was wrong. Every other crossing
below is bidirectional.

#### The six crossings

The boundary is **not** "where a `Value` crosses". Enumerating by `Value` was
still enumerating by representation, and it missed a crossing that carries a
`RefHandle` and bare `u64` slots. The boundary is:

> wherever a reference leaves the instance that can interpret it — whatever
> it is carried in.

**`absolutize` outbound, `localize` inbound.** Both operations are named, and
both carry a type signature that pins the *instance* as well as the
direction:

> `absolutize(store_owning_the_frame_READ, value) -> value`
> `localize(store_owning_the_frame_WRITTEN, value) -> value`

Both are **total** — identity outside the range they convert — so every
crossing below can apply them unconditionally to its funcref-typed positions
without a per-site case analysis. See "The conversion primitives" for the
arms, and for why outbound completeness is load-bearing in a way inbound
completeness is not.

Direction is then checkable by **word** and instance by **argument** — which
is the discipline that would have caught `invoke_runtime_target` passing
`owner_store` while operating on the caller's frame. Neither operation
consults a slot; both resolve in the arena, and both run before any
`checkout`.

**Crossings 1, 2, 4 and 6 are instances of the enforcement invariant** rather
than independent arms: each is a place a `Value` meets a frame, and the
invariant already says which form each side holds. They are listed
individually because the *sites* still have to be found and fixed.

1. **`Instance::function_handle_at`** — outbound only, as above.
2. **Host callback arguments and results, JIT.** Outbound at
   `try_machine_raw_to_value_in_store`'s `Ref` arm (`value_encoding.rs:169`),
   inbound at `value_to_machine_raw_in_store`'s `Ref` arm (`:144`), reached
   from `entry.rs:260-275` / `:337-347` and from `native_eval.rs:38-43` on the
   root path.

   **The conversion surface is four functions, not two**, and an earlier draft
   named only the two on this path. `value_encoding.rs` has a `Ref` arm in
   `value_to_raw_in_store` (`:124`, arm at `:132`),
   `value_to_machine_raw_in_store` (`:138`, `:144`),
   `try_machine_raw_to_value_in_store` (`:162`, `:169`) and
   `try_raw_to_value_in_store` (`:182`, `:196`) — plus the composer
   `normalize_machine_raw_in_store` (`:202-212`). State the fix over **the
   functions carrying a `Ref` arm**, not over a list, and let the compiler
   find them.

   The two that were omitted are not incidental: they are the **public API's**
   conversions. The instance getters and setters use them, including an
   unlisted crossing of their own — the funcref-global getter/setter at
   `jit/instance.rs:1097-1131`, where `global_at` returns a `Value` to the
   embedder through `try_raw_to_value_in_store` and `replace_global_at` /
   `set_global` take one back through `value_to_raw_in_store`.

   `normalize_machine_raw_in_store` needs a **stated output form**. It
   composes machine-raw -> `Value` -> host-raw, passing the *same* store to
   both halves (`:208-211`), which is correct only when one store owns both
   sides. Under the invariant its contract has to say which frame each half
   belongs to; today it composes correctly by luck.

   Note also that "these functions take `&Store`, so the instance is in scope"
   — an earlier justification — is **not sufficient**. The instance in scope
   is whichever the call site passed, and `invoke_runtime_target` passes the
   callee's store while operating on the caller's frame. The signature
   discipline above is what makes that visible.
3. **Host callback arguments and results, interpreter.** **Not** a match arm.
   `value_to_raw_for_interp`/`raw_to_value_for_interp` (`exec.rs:411-433`) are
   free functions with no instance, and the `Value` conversion happens inside
   the closure `interp_imports::bind` builds — which is constructed from
   `&Module` and `&[Import]` *before the `InterpInstance` exists*, so it has
   no instance to localize against at any point. Localize instead in
   **`call_host`** (`exec.rs:2239-2244`), which holds `&mut self` and already
   destructures `Self { module, host, memories, .. }` disjointly
   (`:2249-2254`) for exactly this kind of borrow reason — `self_id` and the
   handle join that destructure. **Absolutize** the funcref-typed raw `u64`
   slots before handing them to the dispatcher and **localize** them on the
   way back, using the funcref-typed positions from the import signature
   `call_host` already looks up. `HostDispatch` keeps its signature: it
   deliberately carries only std types 'so external callers stay
   feature-independent' (`exec.rs:105-108`).
4. **`Instance::invoke` / `call`** — they take `&[Value]` and return
   `Vec<Value>` (`instance.rs:233-238`), so an embedder can pass a funcref
   straight in.
5. **`FuncRefHost::invoke`** — `Box<dyn FnMut(RefHandle, &[u64], &mut [u64])
   -> Result<(), WasmError>>` (`exec.rs:129-131`): a handle and raw frame
   slots, straight to the embedder. Its call sites are `Op::CallRef`
   (`exec.rs:3218`) and `Op::CallIndirect` (`:3504`), **neither of which goes
   through `call_host`**, so (3) does not cover it. Both are inside `exec_ins`
   with `&mut self` in scope, so it gets the same raw-slot treatment:
   **absolutize** the funcref-typed argument slots outbound and **localize**
   the result slots inbound, leaving the hook's signature untouched for the
   same reason as (3) — the boxes are `alloc`'s so an embedder can build one
   without this crate's allocator (`exec.rs:122-126`). The `handle` argument
   is already absolute by construction, being what the embedder was handed.
   Decision 18 keeps this path deliberately, for out-of-world instances and as
   the cross-engine hop, so it is not legacy.
6. **The wasm-to-wasm cross-instance call** — the only crossing that is not an
   embedder or host boundary, and the one the unqualified clause (1) could not
   see. It needs **no new conversion code**: under the enforcement invariant
   it is already handled, provided `invoke_runtime_target` passes the
   **caller's** store at `entry.rs:260-275` and `:337-347` instead of
   `owner_store`. The callee side is already right
   (`arch/common/eval.rs:92-95` writes the callee's frame with the callee's
   store), as is `native_eval`'s `Linked` arm (`native_eval.rs:46-60`), which
   carries `&[Value]` — the neutral form — into the new callee's `eval`.

   The defect is **not** limited to `Linked`: a handle can name a host import
   in another module, after which the same two lines read the caller's frame
   with the foreign store. It is a property of `invoke_runtime_target`
   conflating "the store that owns the callee" with "the store that owns the
   frame", not of linked calls.

   **The interpreter half is follow-up work, named rather than scheduled.**
   `interp_imports.rs:116-119` skips the bind today ("calling it is what
   fails"), so making linked functions drivable adds a second cross-instance
   marshalling path *and* its conversion obligation. Under the single-tier
   scope boundary a `Linked` import in an interp world names another *interp*
   instance, while the existing cross-instance call path is JIT-typed end to
   end (`entry.rs:184-245`) — so this is a new interp-to-interp invoke path,
   not wiring for an existing one. Decision-wise it sits exactly where the
   cross-engine hop sits, and for the same reason: folding a new engine
   capability into a stage whose goal is containing `unsafe` without
   regression is how a stage stops being reviewable.

   **Nothing in this stage depends on it, and that is worth checking rather
   than asserting.** The deletions this stage claims — `OpaqueInterpFunc`, the
   publication map, the `ref.test` divergences — rest on funcrefs having real
   provenance, which step 3 delivers, not on calls being drivable. An
   interpreter's cross-instance *call* continues to go through `FuncRefHost`,
   which the scope boundary keeps.

   **Name the deferred state rather than leaving it half-working.** In an
   interp world a foreign funcref will **type-test as a function** — that is
   the `ref.test` win this stage does deliver — and **calling it traps with a
   stated error**, the way a mixed-tier `instantiate` is rejected with a named
   error. This is a deliberate split: today's `OpaqueInterpFunc` fails the
   test *and* the call together, and the design separates them, so the
   intermediate state must be explicit or it reads as a bug.

   **This is created by the design, not pre-existing.** Today every JIT
   funcref payload is a shared-registry index — absolute already — so copying
   raw slots between two frames is correct. Permitting the local form in frame
   slots is what creates the hazard.

`FuncRefHost`'s own doc comment states the invariant at the boundary that was
missed, and is worth quoting rather than paraphrasing: *"A funcref is a local
index and means nothing outside the instance that wrote it, so one entering a
SHARED table needs a global name"* (`exec.rs:115-116`). The tree already knew.
What went wrong was restating the principle as a storage-class rule and then
enumerating sites by representation.

One documentation fix belongs in this work rather than after it:
`interp_imports.rs:10-13` still claims "A host import taking or returning a
reference is rejected at bind time", which the same file's `raw_kind`
contradicts at `:281-289`. A module header that contradicts its own file is
part of why this boundary read as closed for four review passes.

Two consequences to record. First, this is a **public-API behavior change**:
`function_handle_at`'s returned value is no longer a local index, so an
embedder interpreting the number rather than treating the handle as opaque
sees different values; the doc comment should say the handle is opaque and
absolute. Second, the checklist below gains a **round-trip test** — export a
handle from one instance, install it into a second through
`Import::linked_func_typed_with_context_and_index`, call it, and assert the
callee is the exporter's function. That is the spectest path
(`wast_test_runner.rs:1501-1522`) in miniature, and its absence is why this
boundary survived a full review pass.

The site list is then a checklist against that rule, not the definition of
it — every entry is a place the rule is applied, and each is derived by
asking the same question about its destination.

On the interpreter, at minimum: `TableSet` (`exec.rs:3251-3263`), `TableGrow`
(`:3272-3288`), `TableFill` (`:3289-3305`), `TableCopy` in both directions
(`:3306-3341`), `TableInit` (`:3342-3370`, whose values come from
`elem_value` at `:1395-1398`) — note that this is the **inbound** direction
when the destination is private, since passive segments hold the absolute
form. **Both** element-segment arms (`:1065-1080` and `:1082-1087`); the two
arms of one match disagree today, which the inverted rule resolves by giving
both the same question rather than by listing one of them. The table
initializer (`:950-958`). `Op::GlobalSet` and `Op::GlobalGet`
(`exec.rs:3141-3155`), which already branch on shared-versus-private and so
need the conversion added to an existing branch rather than a new one. And
the exception payload sites: `canonicalize_exception_fields` outbound
(`:1576-1595`) and `localize_exception_field` inbound (`:1605-1614`) — the
latter worth preserving as written, since it localizes only the value
installed in the catching frame, so `ref.eq` against a fresh `ref.func` in
the source instance still holds while other instances keep the absolute name.

**GC heap fields and array elements** need no arm of their own, and this is
worth stating as a derivation rather than a list entry. A GC object's
reachability is **dynamic** — any instance holding the handle resolves the
owner through the shared arena — so no GC field can ever be *statically
proven* unreachable by another instance. It therefore fails clause (2) of the
rule by construction and holds the absolute form always. The conversion sites
are the struct/array helpers, which step 4 (the `Gc` entry conversion)
already opens:
absolute-on-write in `do_struct_set` (`ops.rs:628-658`) and its array
siblings, `localize` on read in `do_struct_get` (`:599-626`) before the value
reaches the reader's frame. The crossing is explicit there today —
`do_struct_set` decodes with the **writer's** store (`:645`, through
`current_store(ctx)` at `:57-68`) and stores into the **owner's** heap, while
`do_struct_get` returns `value_to_machine(origin_store, ..)` (`:625`) into the
**reader's** frame. Producers feeding GC containers ask the same destination
question: `array.init_elem`, and const-expression `struct.new`/`array.new`
through `ConstResolver::alloc_struct`/`alloc_array` (`const_eval.rs:31-42`).

Under the world these retag through O(1) lookups, replacing the linear scans
over `published` at `exec.rs:3194`, `:3478` and `:1605-1614`.

The checklist also covers all six crossings above. Import binding
(`jit/instance.rs:861-867`) is on the list only to record that it needs **no**
change.

> **Process note.** Six times in this review a rule was stated over
> engine-internal structure and turned out to have no arm for a boundary *out*
> of it: the reachability reasoning that described what the world owns without
> asking how anything below a generated frame would reach it; the
> storage-class partition that covered four kinds of engine-owned slot and not
> the public API; the host-call boundary on both engines;
> `FuncRefHost::invoke`; and the wasm-to-wasm cross-instance call. Each was
> found by reading the rule against a call site rather than against its own
> terms.
>
> Two of them sharpened the check, and the second sharpening is the one to
> keep. The fifth defeated the fix for the fourth: enumerating by "where a
> `Value` crosses" was still enumerating by *representation*, and
> `FuncRefHost::invoke` carries a `RefHandle` and bare `u64` slots. The
> sixth was the first where the blindness was **structural** rather than an
> omission — both endpoints of a cross-instance call are containers the rule
> permitted, so no enumeration of containers could ever have found it. What
> closed it was naming the channel instead of the container: **control
> transfer**, alongside aliasing and engine escape.
>
> So the standing check is not "find the `Value` boundaries", and not even
> "find every boundary": it is **name the channels by which a value can leave
> the region the rule protects, and show the list is exhaustive.** For this
> design that list is three, and the argument for its exhaustiveness — with
> the refuted candidates — is recorded above rather than assumed.
>
> A later pass then found two failures of a *different* kind, and they are
> worth naming separately because the channel check does not catch either:
>
> - **A sound rule resting on an unsound predicate.** The channel argument was
>   complete; clause (2)'s *proof* was not, because a public accessor could
>   falsify it at runtime (see "Keeping the static fact true"). Checking a
>   rule's channels says nothing about whether its premises survive contact
>   with the API. The corresponding check: **for every static fact a rule
>   relies on, name what could make it false at runtime, and close it.**
> - **Dangling internal cross-references.** Renumbering the migration steps
>   left five references pointing at real-but-wrong steps, one of them a
>   safety precondition stated by number. A stale name fails loudly; a stale
>   number fails silently. Hence: cross-references carry **names alongside
>   numbers**, and preconditions are stated as **properties of the code**
>   ("deletable when the ref registry holds no raw pointers") rather than as
>   step references.

**On the JIT the write is inline emitted code**, for tables *and* globals
alike, so the boundary has to be decided statically or not at all.

- `lower_table_set` bounds-checks and stores the source register through
  `lower_table_access_continuation` with no runtime helper
  (`lower_leaf_special.rs:831-873`), and tables are genuinely aliased —
  `TableInst { elements: Rc<RefCell<Vec<RefHandle>>> }` (`entities.rs:170-171`)
  bound through `TableInst::from_shared` (`jit/instance.rs:528-533`).
- `lower_global_set` (`lower_leaf_special.rs:602-612`) falls through to the
  scalar emit, which ends in a plain `MachineInstKind::Store` into the cell
  address (`:640-650`); the GpI64 path reaches the same emit
  (`lower_i64_gp64.rs:131-137`). Globals are aliased the same way —
  `GlobalInst { raw_ptr: *mut u64, cell: Rc<GlobalCell>, .. }`
  (`entities.rs:351-356`), bound through
  `from_shared(state.global.clone_shared_cell(), ..)` (`jit/instance.rs:745-749`),
  so both instances address the same `u64`. This path is *more* committed to
  being inline than the table one: the per-global raw pointers are cached in
  the `NativeContext` tail specifically to keep each access at two machine
  loads with no dedicated register (`context.rs:10-27`).

The resolution is **one static fact over both container kinds** — "reachable
by another instance", i.e. `is_import() || !export_names().is_empty()` —
computed per table and per global at instantiation and plumbed to lowering.
A write into a private container keeps its inline store unchanged; only a
write into a reachable one lowers through a retagging helper. Writing it as
one fact over two kinds is also what stops the next container from needing a
third copy of it.

**The global fact needs a type guard the table fact does not:** `reachable &&
the global is funcref-typed`. A funcref table is funcref-typed throughout,
while globals are typed individually, so without the guard the helper cost
would land on every shared `i32` global in every module.

`TableDispatchMode::Generic` is **not** a usable proxy for this and must not
be reused as one. `compute_static_table_dispatch_modes` (`jit/instance.rs:260-292`)
forces `Generic` for a growable table, a non-null initializer, non-local
element segments, declared subtyping, or *any* module-wide table mutation —
so keying off it would push private-table writes onto the helper path in
precisely the modules the performance gate measures. The new fact is
narrower and independent, and nothing else feeds it.

**The interpreter already computes this fact and already implements the
exported-global arm**, which is the precedent to build on rather than a
parallel invention. At instantiation it publishes a funcref global's value
when the global is funcref-typed and a resolver exists (`exec.rs:806-814`),
and it decides sharing from export status with the rule in its own comment —
"A global another instance can reach must BE the shared cell, so an
importer's writes are visible here and ours there. A private one stays in the
array the chain reads" (`exec.rs:816-826`). At runtime both `Op::GlobalGet`
and `Op::GlobalSet` already branch on that decision (`exec.rs:3141-3155`). So
the JIT's const-expression half is a **divergence gap**, the same shape as
the element-segment arms — one engine implements the rule and the other does
not.

Two riders on reusing it:

- **Re-establish the arm on the new mechanism when `published` dies.** The
  interpreter's version publishes through `funcref_host` into the
  publication map, both of which this design deletes. The *decision* it
  encodes survives; the mechanism underneath must be swapped, not inherited.
- **Key the fact on `is_import() || exported`, not on "has a shared cell".**
  The interpreter currently tests `g.export_names().is_empty()` alone
  (`exec.rs:819`), which is the representation talking rather than the
  property. Imported globals are equally reachable, and keying on the
  property is what keeps the fact identical for tables and globals.

#### The `indirect_info` bound check is load-bearing

The native handlers must bound-check the function index against the
`indirect_info` length before scaling it by 24. Today nothing does: the
table-length compare bounds the table *index*, not the slot *value*, and the
"plain slot value is `< funcs.len()`" invariant holds only by construction.
Add the length to `EnterState` and test it.

Under the range encoding this check **is** the interpreter's provenance test,
because `indirect_info` is built `(0..self.funcs.len())` (`exec.rs:1206-1218`)
— so `fi < indirect_info_len` is exactly "this value is in the local range".
It is not a belt-and-braces addition on any backend, and the earlier framing
of it as uniquely load-bearing on RV64 was wrong in a way that invited
removing it elsewhere as redundant. **All three backends that implement
native calls depend on it equally:**

- x86_64 tests `shr rdx, 32` / `jnz slow` (`x86_64.rs:669-671`) and arm64
  tests `lsr x13, x12, #32` / `cbnz x13, slow` (`arm64.rs:652-654`). Both
  reject only values with bits at or above 32. Every value under the range
  encoding is `<= 2^28-2`, so **both tests pass local and foreign funcrefs
  alike** — they are blind to this scheme, and equally blind to any scheme
  whose discriminator lives below bit 32.
- RV64 tests only equality with the null sentinel (`riscv.rs:1856-1866`), so
  it never had a high-bits guard to begin with.

Without the bound check, a foreign funcaddr therefore flows unbounded into
`info + fi*24` on **all three**. The check must not be removed as redundant
anywhere.

#### JIT side effect

If untagged means "local index", then `function_views` — today indexed by
shared-registry index (`context.rs:487-489`, `:623`) — is re-indexed by local
function index. That is mostly a simplification:

- `refresh_function_views` iterates `store.module().functions` instead of the
  whole world registry, so every entry is local by construction.
- The `INVALID` dead-peer arm and the `core::ptr::eq` owner comparison
  (`context.rs:499`) both disappear — the latter is the identity test that
  was going to become an `InstanceId` equality anyway.
- The per-call refresh cost drops from O(world) to O(local functions), which
  is the cost flagged under "Performance expectations" at `entry.rs:335`.

**The JIT's discriminator already exists, and adds no instruction.** Under
the re-indexing, `function_views_len == module.functions().len()`, so
`value < function_views_len` is exactly "this value is in the local range" —
the same test the interpreter's `indirect_info` bound performs. That compare
is already emitted: `build_call_ref_validate_block`
(`lower_module.rs:2798-2828`) contains **loads only** — the ref slot and
`function_views_len` — and the unsigned `Ge` comparison with its two edges
lives in the caller (`lower_module.rs:2084-2101`), where `then_edge`
currently targets `trap_invalid_ref` and `else_edge` targets `type_check`.

The change is to re-target that `then_edge` from `trap_invalid_ref` to the
runtime helper. That is **edge plumbing, not a one-word substitution**: the
trap edge carries no arguments (`args: collections::Vec::new()`), while an
edge into the runtime helper must carry the block's parameters the way the
sibling dispatch branch does (`args: carried_args.param_values()`,
`lower_module.rs:2060-2063`).

**No comparison and no load is added — but that is not the same as "no
cost", and an earlier draft stopped one sentence too early.** It holds for a
table holding the local form. For a table holding the **absolute** form the
call does not merely pay an extra instruction; it leaves the inline path:

- Today a self-dispatch through an instance's own **exported** table takes
  the inline local branch. The generic indirect path branches `kind != LOCAL`
  -> runtime helper, else -> `local_prepare` (`lower_module.rs:2144-2160`),
  and `refresh_function_views` marks the view `LOCAL` via
  `core::ptr::eq(owner_store, current_store) && matches!(func, Local { .. })`
  (`context.rs:498-503`).
- Under the encoding the slot holds an absolute value near `FUNCADDR_TOP`, so
  it fails `value < function_views_len` before any view is consulted and is
  re-targeted to the helper — **every iteration, for the life of the
  instance**.

Two scoping notes, because both are easy to get wrong in either direction.
The **fixed**-dispatch tables are *not* involved: `refresh_fixed_call_table_views`
skips any table whose mode is not `FixedLocalOnly` and `continue`s before its
`INVALID` arm (`context.rs:571-574`), and `FixedLocalOnly` requires
`private_local = !table.is_import() && table.export_names().is_empty()`
(`jit/instance.rs:277-290`) — an exported table is `Generic` today and has no
fixed entry table at all. And the bullet above about the `INVALID` dead-peer
arm at `context.rs:499` stands unqualified; it is a different arm.

`linking.wast`'s `$Mt` is the executable case — a module that exports a table,
fills it with its own `$g`, and `call_indirect`s through it
(`linking.wast:269-281`, asserted at `:303`).

**Correctness on this path needs `owner == self` before checkout**, not
after. The helper resolving its own id and then materializing would overlap
the caller's live materialization on an ordinary dispatch path — decision
14's in-use count permits the second *token*, correctly, but what must not
overlap is the materialization. `localize` consults the arena and no slot,
which is why the ordering is part of its definition.

**Cost, honestly.** CoreMark exports no table, so the gate stays flat and the
narrowed claim under "What the JIT keeps" survives. That is exactly the
problem: **the gate structurally cannot see this shape.** Decision 10 costed
and gated the outbound half of the same trade; the inbound half lands on
exported tables shared between modules, which is the workload this design
exists to serve, and no benchmark in the suite exports a table. "The gate
stays flat" is true and uninformative until one does — hence the benchmark
prerequisite in migration step 1.

**The optimization is a measured decision, not a design commitment.** The
candidate is a per-instance contiguous funcaddr block plus `self_lo`/`self_hi`
in `NativeContext`, turning the self case into a register range test that
recovers the inline path; it composes with the downward-from-`FUNCADDR_TOP`
encoding because a contiguous funcaddr block maps to a contiguous encoded
range. Its real cost is that the escapable-set filter (step 5, "Bound
registration at the source") breaks
offset-equals-local-index, so it needs a small per-instance index map and a
load. Land the benchmark, measure, then decide.

Two alternatives are **rejected** rather than left open. Keeping the local
form in exported tables the owner itself reads reintroduces exactly the
misdispatch the absolute-always rule exists to prevent — a peer reading that
slot cannot know it is local. Re-localizing on every table read moves an arena
lookup onto the load path, which is worse than the helper call it replaces.

#### The 32-bit budget

`RefHandle` has 28 payload bits untagged and 27 pooled (`value.rs:21-38`),
and the TARGET32 wire form matches (`value_encoding.rs:13-16`). The two
ceilings have **different** bounds, and an earlier draft justified both with
one clause about the escapable-set filter, which is wrong:

- **The function address space** is what the escapable-set filter (step 5,
  "Bound registration at the source") bounds. With it, the space is bounded by live instances rather than
  by churn.
- **The pooled arena is not bounded by live instances at all.** Its entries
  are minted per *operation*, not per instance: one per exception object
  (`alloc_exn_in`, `link.rs:184-196`, reached from `ops.rs:143`, `:177`,
  `exec.rs:1646-1647`, `:1694-1695`) and one per escaping GC object with no
  dedup (`register_gc_ref`, `store.rs:273-285`; `register_i31` at least
  linear-scans first). Nothing reclaims either — exception entries need no
  teardown precisely because their `Rc` owns the object (`store.rs:315-317`),
  which is why they accumulate. The real bound is total such allocations over
  the world's lifetime. A long-running embedder throwing in a loop consumes
  the space monotonically. The world inherits this property; it neither
  creates nor worsens it.

At the pooled ceiling the behavior is **silent aliasing, not a trap**:
`from_pool_index` masks with `host_payload_mask()` (`value.rs:95-97`), which
is `(1<<27)-1` on 32-bit (`value.rs:21-38`), so entry `2^27` yields
`2^27 & (2^27 - 1) == 0` — a handle resolving to pooled entry 0, a live and
differently-typed object, with no diagnostic. `pooled_index()` masks
identically on the way back (`value.rs:86-89`), so the round trip is
consistently wrong. See "Filed separately".

## What this deletes

- `impl Drop for Store` registry poisoning, and the O(registry) teardown.
- Both populations of cross-instance `unsafe` deref, replaced by the single
  `checkout` primitive: the ~13 GC-entry derefs behind
  `resolve_struct_ref`/`resolve_array_ref`, and the function-entry derefs at
  `entry.rs:192`, `native_eval.rs:50`, `gc_type_check.rs:35` and `:104`,
  `context.rs:491`, and interp `exec.rs:1466`.
- `OpaqueInterpFunc`, the publication map and its linear scan, and the
  hostref overloading.
- The second funcref model, and with it the engine-visible `ref.test`
  divergences.
- The function-registry revision counter — `cached_function_registry_revision`
  (`context.rs:138`, `:233`, `:237`, `:245`, `:257`, `:781`) and
  `Store::function_registry_revision` (`store.rs:143`) — together with the
  cross-instance view cache it guarded. No world-owned counter replaces it;
  see "What the JIT keeps".

Remaining `unsafe` after this: the generated-code ABI itself (`preserved/
runtime_call` marshalling, signal handling, executable memory), the
epoch-validated raw caches, and the one checkout primitive — the parts that
are the JIT's actual job.

## Performance expectations

- Hot paths (in-instance execution, native dispatch, interp chain):
  untouched — same pointers, same epochs, same emitted code.
- Cross-instance resolution (linking, funcref publication, GC type tests
  across instances, linked calls): one bounds check + generation compare
  instead of a raw deref. These paths are already cold or already go
  through a registry borrow.
- **Two items go under the gate, not one.** *Inbound*: the funcref
  provenance test on the by-handle dispatch path
  (`build_call_ref_validate_block`), whose exact cost depends on open
  question 2. *Outbound*: the retagging helper on writes into a table another
  instance can reach. The outbound item lands only on modules that import or
  export a table, which is what makes it acceptable — every benchmark the
  gate runs keeps its inline store — but it is a real cost and earlier drafts
  omitted it by counting only the inbound side.
- Instance teardown gets cheaper: a generation bump instead of scanning every
  registry entry. This is a real but small win, and it should not be read as
  the headline: the larger cost on this surface is the **per-call** refresh.
  `invoke_runtime_target` calls `ctx.refresh_cached_views()` unconditionally
  after every runtime call, host callbacks included (`entry.rs:335`), and
  `refresh_function_views` iterates the entire shared registry and allocates
  a fresh view vector (`context.rs:470-541`). That cost is O(all registered
  functions) and grows with instance churn. It is pre-existing — this design
  neither creates nor worsens it — but a design that makes instance death
  cheap should not leave the per-call cost of instance birth unbounded, so
  the migration addresses it directly.
- **The inbound cost is real, spans three containers, and is currently
  unmeasurable.** Reading a funcref in the absolute form and calling it leaves
  inline local dispatch for a helper call, every iteration (see "JIT side
  effect"). It arises from an exported table, a shared funcref global, and a
  GC field — and the gate has a benchmark for none of them, because none of
  those shapes appears in it. Migration step 1 adds coverage for the shape
  *before* the encoding lands. Until then, "the gate stays flat" is true and
  uninformative about the workload this design exists to serve.
- **Outbound, the same three containers take a retagging helper on write.**
  Tables and globals decide it statically, so a module that shares neither
  pays nothing and its emitted code is unchanged; GC fields have no static
  answer available, since reachability there is dynamic.
- CI gates: performance-regression suite must stay flat; any measured
  regression on CoreMark/dispatch benchmarks blocks the stage. Note that the
  gate does not build `memprof`, so the per-`checkout` stack capture that
  feature adds never enters a measurement — but a profiling run must not be
  read as one either.

## Migration plan

0. **Prerequisite, in `tools/tracked-alloc`, before any world work.** The
   `Weak` back-reference does not compile under `memprof` as the crate stands.
   Under that feature `tracked_alloc::rc::Rc` is a wrapper
   (`lib.rs:2995-2998`) whose entire API — `new`, `from_alloc_rc`, `clone`,
   `get_mut`, `ptr_eq`, `as_ptr` (`:3000-3053`) — has no `downgrade`, and
   `inner` is private, so a call site cannot reach `alloc::rc::Rc::downgrade`
   either. This is a hard gate, not a latent problem: `feature_args` maps
   `"all"` to `--all-features` (`ci/correctness.py:139-144`), the workspace
   diagnostic check runs it (`:246-252`), and `sf-nano-core/Cargo.toml:87`
   declares `memprof = ["tracked_alloc/memprof"]`.

   Add `downgrade` to the wrapper, returning the already re-exported
   `alloc::rc::Weak` (`inner::Rc` *is* `alloc::rc::Rc`, so
   `inner::Rc::downgrade` type-checks against it). Two call-site constraints
   are silent if missed: **use associated-function syntax**
   (`Rc::downgrade(&x)`, `Rc::clone(&x)`) so one source compiles against both
   `cfg` arms, and **never name `alloc::rc::Weak` directly** — go through
   `tracked_alloc::rc::Weak`.

   The profiler accounting is forced by how the wrapper already works, not
   open: tracking is **per clone handle** (`from_alloc_rc` retains with a
   captured stack, `:3010-3031`; `Drop` releases, `:3055-3059`). So
   `downgrade` must **not** release, `upgrade` **does** retain by going
   through `from_alloc_rc`, and a `checkout`'s temporary retains and releases
   within the call — net zero. **Add a test asserting a
   downgrade/upgrade/drop cycle leaves live bytes unchanged**; an asymmetry
   here would surface as a slow leak in every profile taken afterwards and be
   attributed to the world.

   Caveat to carry forward: under `memprof` every `upgrade` — so every
   `checkout` — captures a stack. That is fine for a development aid and
   wrong as a measurement. The performance gate does not build `memprof`, so
   nothing is at risk mechanically; the risk is a human reading a profiling
   run and concluding cross-instance calls are expensive.
1. **Coverage baseline — the measurement prerequisite.** Step 0 is a compile
   prerequisite; this is the measurement one. Nothing here needs the world:
   the benchmark is a wasm module plus a harness entry, the test is a spectest
   of two linked modules, and **both are written against the unmodified tree,
   run there, and their results recorded** — that recording *is* the baseline.
   A coverage step taken after the change it measures has no baseline; it can
   report an absolute number but never a delta.

   **Gated on this step:** step 3's discriminator re-targeting (which is where
   the dispatch cliff arrives) and step 4's GC retag sites. Those are the two
   steps whose regressions are *silent* — wrong value, no trap, suite green —
   so neither may land until the coverage exists.

   Two items, and both exist because a correct design cannot depend on the
   suite happening to look:

   *A benchmark for the **shape**, not for one instance of it.* The costed
   shape is **an absolute-form funcref read, then called** — and it now
   arises from three containers: an exported table (`linking.wast`'s `$Mt`), a
   shared funcref global, and a GC field. None has a gate benchmark, so the
   perf gate cannot observe the inbound cost described under "JIT side
   effect". Cover the shape once rather than writing three benchmarks:
   read a funcref in the absolute form and call it in a loop.

   **Reading the result: this benchmark measures two stacked costs on one
   shape.** An indirect dispatch through the instance's own exported table
   pays the dispatch regression (the call leaves inline local dispatch) *and*
   the cross-instance marshalling lookup. They are independent changes landing
   on the same code, so the first regression reading must not be attributed
   wholly to whichever one happens to be under review. If they need separating,
   a same-instance *direct*-call variant isolates the marshalling half, since
   direct calls convert nothing.

   *A cross-instance GC test, which the suite does not have at all.* Two
   linked modules; one writes a funcref into a `(array (mut funcref))` or a
   struct field, the other reads it and calls it. **Assert the returned
   value**, not the absence of a trap: the failure mode here is
   signature-compatible silent misdispatch, so a test that only checks for a
   trap would pass against the bug. The suite has ref-typed GC containers
   (`array_fill.wast:29`, `array_init_elem.wast:33`) but never passes one
   between instances, which is exactly why this survived five review passes.
2. Introduce `RuntimeWorld` + `InstanceId` behind the existing public API
   (worlds of one instance; `LinkRegistry` internally becomes a world
   reference). No behavior change; all suites green.
3. Move the function registry to the shared function address space (JIT
   first, replacing `FunctionRegistryEntry`, then interpreter publication)
   **and re-index `function_views` by local function index in the same
   step.** These are one change seen from two sides — the encoding says a
   funcref payload is a local index, and the view array is what consumes that
   payload — and splitting them is what created an earlier draft's hazard:
   replacing `FunctionRegistryEntry` makes the function-registry half of
   `Store::drop` (`store.rs:307-314`) uncompilable, which deletes the
   revision bump that was the epoch's teardown edge, while the replacement
   was scheduled a step later. Doing both here removes the edge *and* the
   cache that needed it at the same instant, so no intermediate state is ever
   missing an invalidator.

   This step also carries the JIT's provenance discriminator, which under the
   range encoding is the re-targeting of an existing branch edge rather than
   an added test (`lower_module.rs:2084-2101`; see "JIT side effect"). An
   earlier draft made this step wait on an unresolved tag placement; that
   dependency is **discharged** — the tag test is withdrawn, not deferred,
   because the range encoding needs no tag. Step 3 is still the large one and
   should not be read as a one-line registry swap.

   **The accessor narrowing lands here too, with the fact it protects.** The
   two sharing-accessor filters, the `pub(crate)` on
   `JitInstance::store`/`store_mut` with `function_has_native_code` replacing
   its one out-of-crate caller, and the interpreter's two filters — all of it
   is a *prerequisite* for clause (2) being true, not a consequence (see
   "Keeping the static fact true"). Also delete `table_elements_at` and
   `replace_table_elements_at` (`jit/instance.rs:1201-1233`) here rather than
   at the public-API step: the conversion obligation is live from this step,
   the functions have no caller anywhere in the workspace, so deleting them
   early is free and keeps this step's own promise that no intermediate state
   is missing an invariant.

   **The audit for this step is two-armed, and both arms are greps:**
   - *(i) Value crossings* — every `pub fn` on `Instance` or `JitInstance`
     whose signature mentions `RefHandle` or `Value`.
   - *(ii) Aliasing crossings* — every `pub fn` yielding a container or
     storage handle: `TableInst`, `GlobalInst`, `ImportedTableState`,
     `ImportedGlobalState`, `&Store`.

   Arm (ii) is the one that finds the clause-(2) class, and it is the arm an
   earlier draft did not have: `shared_table_state_at` and `store()` mention
   neither `RefHandle` nor `Value`, so arm (i) alone cannot see them. The two
   arms answer different questions — *what form does this carry* versus *what
   can this alias* — and only the second decides whether the static fact
   survives.

   **The instantiation window closes here too.** `instantiate` reserves the
   `InstanceId` and its generation up front — so the registration loop at
   `jit/instance.rs:861-867` can mint `FuncEntry { owner: id, .. }` and
   element-segment materialization at `:907-938` keeps working unchanged — but
   leaves the slot **`Vacant`** until `init_result` returns, moving the
   `Box<Store>` in at that point. Without this, initialization holds a
   materialized `&mut store` across the start function (`:972-976`) into a
   slot that is already checkout-able from `:867`, which is the longest-lived
   materialization in the engine and a direct violation of the invariant.

   Behavior change, stated rather than assumed: re-entry into a
   still-initializing instance now traps, because `checkout` finds the slot
   vacant. Verified against the suite — only `linking.wast`, `linking3.wast`
   and `ref_func.wast` combine a start function with `call_indirect`, and none
   self-references through the world during initialization. A self-dispatch
   during this window works only because `localize` runs **before** `checkout`
   and consults no slot; if that ordering is ever relaxed, this window breaks
   silently.

   **Named regression case: `linking.wast:592-611`.** This step introduces the
   two forms, so it is the step that could send the local form into an
   imported table. The test passes today and would keep "passing" in the sense
   of not trapping — `(invoke $Ms "get table[0]")` would return `104` instead
   of the `0xdead` asserted at `:610`, because both candidate functions share
   `(type $t)` and the type check cannot separate them. Watch the **value**,
   not the exit status.
4. Convert `RefRegistryEntry::Gc { store: *mut Store, gc_ref }` to
   `Gc { owner: InstanceId, gc_ref }` and re-point `resolve_struct_ref` and
   `resolve_array_ref` (`ops.rs:582-597`) at the world. This is **one
   signature change covering all ~13 call sites**, not the thirteen-site
   slog the raw count implies — each site destructures the pointer straight
   out of the resolver's return type.
   **Delete `impl Drop for Store` at the end of this step, not earlier.**
   **State the precondition as a property of the code, not as a step
   number:** `impl Drop for Store` is deletable exactly when the ref registry
   no longer holds raw pointers — checkable by reading `store.rs`. Step 3
   (function registry) leaves `store.rs:318-325` live, so this step is where
   the property first becomes true.
5. Bound registration at the source: register only functions that can
   escape, rather than every function of every instance
   (`jit/instance.rs:861-867`). The escapable set is statically known
   because the validator already rejects `ref.func x` as "undeclared
   function reference" unless `x` is declared (`validator/functions.rs:1026-1054`).
   **Caveat, and it is load-bearing:** the `ElementInit::InitExprs` arm sets
   `is_declared = true` unconditionally for any index, without inspecting
   the expressions (`functions.rs:1045-1048`), so element segments alone
   over-approximate to "everything" on any module using expression-form
   segments. The **code-section `ref.func` scan is the authoritative input**;
   element segments and exports supplement it. In the same step, guard the
   per-call refresh at `entry.rs:335` behind the revision check
   `prepare_for_invocation` already uses (`context.rs:201-203`) — and verify
   first that `cached_views_are_current` (`context.rs:250-319`) covers
   everything a callee can mutate, rather than assuming it.
6. Unify ref type tests (the deferred Stage 2d) on world provenance.
7. Collapse the encodings (Stage 2c). Include the
   `ref_to_machine_raw`/`machine_raw_to_ref` round trip in this step's test
   list: that is where the 64-bit and 32-bit forms are reconciled.
8. **Failed instantiation: the slot stays occupied and the error carries an
   `InstanceId`.** `InstanceInstantiationError::Partial { id, error }`
   replaces the `JitInstance`-carrying variant (`jit/instance.rs:52-58`,
   constructed at `:981-987`). Three reasons, and the first is a hard
   requirement rather than a preference:

   *It is exercised.* `linking.wast:592-611` has a module write its `$f` into
   an imported table, trap in its start function, and the suite then invoke
   through that table expecting `0xdead`. An occupied slot with an unbumped
   generation preserves that by construction — where freeing the slot would
   bump the generation and break it. The harness's own comment states the
   requirement: an instance "may publish a funcref into a table another module
   holds, and that reference has to stay callable afterwards -- including when
   this instantiation traps" (`wast_test_runner.rs:1369-1373`).

   *It is engine-neutral*, which removes the `vm/instance.rs:136` divergence
   where the interpreter discards its partial while the JIT preserves one, and
   genuinely resolves the `#[cfg(sf_jit)]` error-type asymmetry rather than
   asserting it away.

   *It is implementable*, where handing back a `JitInstance` by value is not:
   the `Box<Store>` lives in a slot, and removing it is the one thing the
   retention requirement forbids.

   **Dependency, stated in both directions:** occupied-on-failure is only safe
   because step 0's `Weak` fix makes dropping the world reclaim slots. Without
   it, "occupied" means "stranded forever". Neither may be changed in
   isolation.

   Consumer to update: `retained_failed_instances: Vec<JitInstance>`
   (`wast_test_runner.rs:685-688`, pushed at `:1436-1444`) becomes a list of
   ids or disappears — the harness kept those values alive only to keep stores
   alive, and under the world the world does that.
9. Public API switch: `world.instantiate`/`invoke`, keeping
   `Instance::from_module` as the one-slot convenience.

Each step lands green on the full matrix (all engine configs, all targets,
both spec suites, WASI suite, performance gates).

## Open questions

1. **Where does the world borrow release, and what carries identity across
   the gap?** Answered in principle by `checkout` above — the same
   answer on both engines, with the JIT's `*mut NativeContext` as one
   instance of the token rather than the mechanism. Two directions have to be
   covered and only one is settled. *A token outliving its slot* is answered
   by the in-use count and the RAII guard in "The seam". *A safe reference
   outliving the token* is the remaining open work: the worked call path, and
   the exact points at which each engine takes and returns a token.
2. ~~Funcref encoding~~ — settled by "Provenance lives in the encoding":
   disjoint ranges in one value space rather than a provenance bit, local
   funcrefs upward from 0 and foreign ones downward from
   `FUNCADDR_TOP = (1<<28)-2`, retagging at every transfer boundary in both
   directions, and the `indirect_info` bound check mandatory on every backend.
   This was a correctness question, not a performance one — a context-free
   funcaddr in the local range silently passes every guard the interpreter's
   fast paths apply. The discriminator on both engines is a bounds compare
   that already exists, so the inbound side adds no instruction.
3. ~~`Rc<RefCell<RuntimeWorld>>` vs. explicit `&mut World`~~ — settled by
   the checkout API above: `&mut RuntimeWorld` at the embedder boundary,
   released at checkout internally.
4. ~~Should `FuncRefHost` survive?~~ — settled: yes, narrowed to
   out-of-world instances, for the reason given in "Embedder API".
5. Entity sharing (`Rc<RefCell<MemBacking>>` etc.) is untouched by this
   proposal. A follow-up could move entities into world-owned arenas too
   (ids all the way down), which would also give the interpreter's shared
   tables/globals a chain-visible flat representation — measured decision,
   not assumed.

## Filed separately

Three pre-existing items this design touches but does not fix. They share a
shape: each masks or degrades silently where it should fail loudly, and if
any is ever addressed, the fix is the same — error at the ceiling instead of
wrapping into a live, wrongly-typed value.

**RV64 `call_indirect` guard.** `interp_gen/riscv.rs:1856-1866`: the handler
tests the table slot for equality with the null sentinel and nothing else,
while the comment above it (`:1853-1855`) states that tagged handles are
filtered by having high bits set. x86_64 and arm64 do implement that test;
arm32 and RV32 are unaffected because both route every call flavour to the
slow stub. No failing case has been constructed — reaching it requires a
special-tagged handle inside a *private* table — so this is an unverified
divergence between comment and emitted code, pre-existing and independent of
this design. It concerns only the *existing* `hostref`-style handles the
comment describes; it is not the reason the `indirect_info` bound check is
mandatory. That check is mandatory on all three backends for the separate
reason given under "The `indirect_info` bound check is load-bearing": no
backend's high-bits test can see a discriminator below bit 32.

**The SIMD arena's growth.** `SharedSimdRegistry::intern` (`link.rs:107-129`)
linear-scans for a duplicate and pushes otherwise: append-only, O(n) on
insert, with no reclamation, and its own TODO says so. The world carries the
arena across unchanged, so it inherits that property. This is the same class
of unbounded growth that step 5 ("Bound registration at the source") removes
from the function address space, and a design whose stated motivation is instance mortality should name
it rather than inherit it silently.

**Pooled-index overflow.** `from_pool_index` masks rather than failing
(`value.rs:95-97`), so on a 32-bit target pooled entry `2^27` aliases entry 0
with no diagnostic, and `pooled_index()` masks identically on the way back
(`value.rs:86-89`). Combined with the per-operation, never-reclaimed minting
described under "The 32-bit budget", this is a reachable ceiling on a
long-running embedder rather than a theoretical one.
