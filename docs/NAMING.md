# Runtime type naming

**Status: waves 1 and 2 landed; waves 3 and 4 are undecided.**

Waves 1 and 2 are done — `71659c61` (the `Handle` family) and `ff8e482d`
(the body/lease swap). Waves 3 and 4 are not renames and are not scheduled;
they need decisions recorded below.

The first draft of this document was reviewed against the code and failed on
eleven counts; the review is why this one is smaller. What changed and why is
recorded at the end.

## Method: extract the convention, do not invent one

The first draft invented a taxonomy and imposed it. That is why it produced
names like `ModuleEntities` for a type that also holds config, a type context,
and executable memory, and `FuncArena` for a vector that contains locations
rather than functions. A name that asserts something false is worse than a
vague one, because it will be believed.

This draft starts from the opposite end. **The codebase already distinguishes
these things correctly in three places.** The convention's job is to state
those patterns and apply them where the code violates its own practice.

### Pattern A — identity, permission, scope

```rust
InstanceId     // a name for a slot; resolving it is generation-checked
InstanceToken  // resolved permission; while it lives the slot cannot be freed
InstanceLease  // a token held for a scope, released on drop
```

Three tiers: **name**, **permission**, **scoped permission**. This is the
pattern the whole refactor was built on and it is already right.

### Pattern B — keeps-alive handle versus payload owner

```rust
MemInst    { backing: Rc<RefCell<MemBacking>>, limits }  // handle + metadata
MemBacking { data: Vec<u8>, .. }                         // the payload

GlobalInst { raw_ptr, cell: Rc<GlobalCell>, .. }         // handle + cached ptr
GlobalCell { raw: UnsafeCell<u64> }                      // the payload
```

`Inst` is the `Rc` handle that keeps a payload alive and carries the type's
metadata; `Backing` / `Cell` is the allocation itself. Two tiers, already
consistent. The first draft's `MemoryEntity` would have collapsed both into
one word and lost the distinction.

### Pattern C — engine body versus lease

```rust
InterpInstance      // the body: entity state, execution scratch, arena clones
InterpInstanceLease // a lease over one
```

The interpreter already names this correctly.

## The population this governs

`sf-nano-core/src/vm/` defines **413** types. The overwhelming majority are
compiler and backend internals — `Arm64Reg`, `CfgBlock`, `EdgeStub`,
`BlockPlan` — and no storage-and-identity convention should govern them.

**This convention governs the instance storage and identity types**: those
declared in `vm/link.rs`, `vm/instance.rs`, `vm/entities.rs`, `vm/tag.rs`,
`vm/value.rs`, `vm/jit/instance.rs`, `vm/jit/runtime/mod.rs`, plus the engine
bodies and their leases. Roughly forty types. Everything else is explicitly
out of scope; a compiler IR node is free to be called whatever the compiler
finds clearest.

## The question a name must answer

Not "can resolving it fail" — the first draft's axis, which broke because
fallibility is a property of the **operation**, not the type. Resolving a
`RefHandle` to an arena entry cannot fail; resolving the same handle to a live
owning instance can. One suffix cannot encode both.

The question that every type in the population does answer is:

> **What does this type do about the lifetime of the thing it refers to?**

| Role | Suffix in use | Meaning |
|---|---|---|
| Owns the payload | `Backing`, `Cell`, `Heap`, or a plain noun | the bytes are here |
| Keeps alive | `Inst` | holds an `Rc`; extends the payload's life |
| Names | `Id` | identifies a slot; resolution is checked |
| Locates | `Entry`, `Location` | records where something is; owns nothing |
| Grants access | `Token`, `Lease`, `Access` | a capability, with its guarantee |
| Raw access | `Pointer` | unchecked; `unsafe` to follow |
| Identity only | *(see below)* | compared, never resolved |
| Borrowed view | *(no suffix needed)* | the `<'a>` parameter already says it |

Scope prefixes are a **closed set**: `World`, `Jit`, `Interp`. Anything else
before the suffix is the subject. This resolves the first draft's grammar
ambiguity, where `InstanceTable` could be read as scope `Instance` + subject
`Table` — it is subject `Instance` + `Table`, and `InstanceId` is likewise an
id *of* an instance, not an id *scoped to* one.

## What actually needs to change

Only where the code violates its own patterns.

### 1. The JIT body and lease are swapped (Pattern C)

| Today | Proposed |
|---|---|
| `Store` — the JIT body | `JitInstance` |
| `JitInstance` — a lease | `JitInstanceLease` |

`Store` has not been the spec's store since the 2026-02 split; it is the JIT's
instance body, and `JitInstance { lease: InstanceLease }` owns nothing.

**This rename alone does not achieve its goal**, and the reason is not
visibility. The two engines expose *different access patterns*:

| | escape hatch | hands back | public methods |
|---|---|---|---|
| JIT | `as_jit()` / `as_jit_mut()` | `&JitInstanceLease` — the token wrapper | 28 |
| Interp | `with_interp(f)` / `with_interp_mut(f)` | nothing; lends `&InterpInstance` for a scope | 26 |

Which is why the visibility is mirrored: the JIT publishes its lease because
that is what it hands out, and the interpreter publishes its body because
that is what its closure lends. Each is internally consistent. `lib.rs:44-49`
calling them counterparts is what is false.

The two shapes are not equally safe. The closure form is the containment
pattern this refactor established — materialization is bounded by a scope the
caller cannot escape. The lease form is safe today only because
`JitInstanceLease`'s 28 methods each materialize internally and return
nothing borrowed.

Three coherent resolutions:

- **A — make both closure-scoped.** Add `with_jit(f)`, retire `as_jit`.
  Consistent with the containment design and makes the counterpart claim
  true. Breaks any embedder calling `as_jit`.
- **B — make both lease-returning.** Publish `InterpInstanceLease` with
  forwarding methods. More work, and it spreads the weaker shape.
- **C — keep both, correct the comment.** Free and honest: document that the
  JIT hatch is a token wrapper and the interpreter's is a scoped accessor,
  because that is how each engine materializes.

**Recommendation: A**, on the grounds that the closure shape is the one the
rest of the runtime is built on; **C** if the API break is not worth it.
This is a decision about the public API, not about spelling.

### 2. `Handle` carries four roles

| Today | Role | Proposed |
|---|---|---|
| `RefHandle` | a reference value carried in guest data | `RefValue` |
| `TagHandle` | minted identity, never resolved | `TagIdentity` |
| `InstanceHandle` | weak back-reference plus own id | `InstanceBackref` |
| `WorldHandle` | invocation capability for a chosen peer | `WorldAccess` |

`RefValue` deliberately does not promise infallible resolution — it names what
the type *is* (the reference representation inside a `Value`), which is the
only claim that survives contact with the code. `TagIdentity` avoids `Id`
precisely because nothing resolves it.

### 3. `RefRegistryEntry` must split — the one forced design change

```rust
enum RefRegistryEntry {
    I31(i32),                    // an inline payload
    Gc { owner, gc_ref },        // a locator
    Exn(Rc<ExnInstance>),        // a strong owner
}
```

Three variants, three lifetime roles. No suffix is honest about this type, and
that is the convention working as intended: the name cannot be fixed because
the *type* conflates three things. Splitting it is a design improvement the
naming exposed rather than caused.

**This is the only change here that alters structure rather than spelling.**

### Explicitly not renamed

`InstanceId`, `InstanceToken`, `InstanceLease`, `InstanceTable`, `WorldSlot`,
`MemInst`, `MemBacking`, `GlobalInst`, `GlobalCell`, `TableInst`,
`FunctionInst`, `GcHeap`, `NativeContext`, `StoreAccess`,
`InterpInstanceAccess`, `Caller`, `Func`. Each already satisfies the rule.
A convention that renames names which are already honest earns resentment and
teaches nothing.

`ModuleInst` also stays: the first draft's `ModuleEntities` narrowed it
falsely, and no better name has been established.

## Sequencing

| Wave | Scope | Notes |
|---|---|---|
| 1 | the `Handle` family (§2) | **Landed `71659c61`.** Four independent renames. The tree lint caught a first attempt that rewrote wrapped continuation lines inside committed Facts entries. |
| 2 | body/lease swap (§1) | **Landed `ff8e482d`.** Not mechanical: `Store` is also an opcode (140 `MachineInstKind::Store`, 9 `Fam::Store`), so the type surface was bounded and verified opcode-free before renaming, with rustc confirming the bound. |
| 3 | the visibility decision (§1) | A design change. Needs a decision before it can be planned. |
| 4 | `RefRegistryEntry` split (§3) | A design change, independent of the others. |

Waves 3 and 4 are not renames and should not be scheduled as though they were.

No rename in waves 1–2 changes behavior. A wave that alters a test's expected
*value* means something went wrong.

## What the first draft got wrong

Recorded so the same mistakes are not re-proposed:

1. **`Id` versus `Ref` as fallible versus infallible.** False in both
   directions: `SharedFunctionRegistry`'s own comment states that absolute
   handles may outlive their owner, and `TagHandle` has no resolver at all.
2. **"Owns bytes" as a binary.** Direct payload ownership, strong
   lifetime extension, and access capability are three different things.
3. **A rule claimed total over an undefined population.** 413 types, of which
   ~40 are in scope.
4. **Names asserting false things**: `ModuleEntities`, `FuncArena`,
   `V128Arena`, `EscapeArenas`, `WorldBackref`.
5. **Suffix bans as ratchets.** "No name ends in `Inst`" would have destroyed
   Pattern B, which is one of the three things the code already gets right.
6. **A five-wave, ~900-site plan** where the demonstrable defects are four
   renames and two design decisions.

## Investigated and settled: the `FunctionSpec` native-code cache

`module/entities.rs`'s `FunctionSpec` carries two JIT-only fields behind
`UnsafeCell` — `native_code` and `native_cache` — with every caller in
`vm/jit/`. That is engine-only state living in the shared module layer.

The stated risk to moving it was that the cache is attached to the module and
might therefore be shared by all instances of that module, so relocating it
into the JIT subtree would lose compiled-code reuse. **That risk does not
exist.** `Module` derives only `Debug` — it is not `Clone` and never held
behind `Rc` — and `FunctionSpec` is likewise neither, and is *moved* into
`FunctionInst::Local` at instantiation (`jit/instance.rs:345`). Each instance
owns its own specs, so the cache is already per-instance and no cross-instance
sharing can be lost.

The move is therefore mechanical. Every caller already holds both the local
function index and the JIT's `ModuleInst` (`build.rs:1285,1385`,
`native_eval.rs:55`), so a side vector on `ModuleInst` indexed by function
index reaches all of them.

One thing it does **not** buy: the `UnsafeCell` does not disappear.
`ensure_module_compiled` takes `&JitInstance` and still writes the cache, so
interior mutability is required wherever it lives. The gain is that the
shared module layer stops carrying `unsafe`, not that the `unsafe` goes away.
