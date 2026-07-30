- An instance's identity across instance boundaries is a slot index plus a
  generation, not a pointer. Instances live boxed in generational slots of a
  linker-owned table, and a stale identity fails its generation check and
  resolves to nothing (`InstanceId`).

- Resolving an identity to its instance goes through one audited primitive that
  validates the generation, increments a per-slot in-use count, and yields a
  token. The token is an RAII guard: a slot with live tokens cannot be freed,
  and the last token to be released reclaims it (`checkout`).

- Two live tokens on one slot are a required state; two overlapping
  materializations of `&Store` from those tokens are not. A materialization is
  scoped, never spans a call, and never overlaps another for the same slot. The
  invocation boundary carries identity, not a borrow.

- A slot stays vacant for the whole of its instance's initialization, and the
  instance moves into it only once initialization returns.

- A function reference names an instance and a local index. Its wire form is
  two disjoint ranges of one integer space with no tag bit: an instance's own
  functions count upward from zero, foreign ones downward from a reserved top.

- The local form is a permission, not a default: it may appear only where the
  owning instance alone can observe it — its own frame slots, and containers
  statically proven unreachable by other instances. Everywhere else, including
  every value handed to an embedder, the absolute form is required. Defaulting
  to absolute fails safe; defaulting to local misdispatches silently.

- Conversion between the two forms is total and range-conditional in both
  directions, resolves against the shared function arena and per-instance index
  map without consulting any slot, and runs before any checkout. Its argument
  is the instance owning the frame being read or written, which is not always
  the instance owning the value.

- A failed instantiation keeps its slot occupied and its generation unbumped.

- Both engines answer reference type tests from one implementation over this
  identity, and both share one slot encoding.

## Facts

- 2026-07-30 rationale: a slot is kept occupied on a failed instantiation, and
  a vacant window is held for the whole of a successful one, for the same
  reason: initialization may already have published references into another
  instance's storage, while a start function that re-enters its own instance
  must find nothing to check out (code).

- 2026-07-30 (bc7cbb03) statement: the two invariants above are one change, not
  two. Holding a materialization across a call and occupying a slot during
  initialization are the same defect seen from two ends: an A -> B -> A call
  produced two overlapping `&mut Store(A)` until the invocation boundary
  carried tokens, and the vacant-window rule is what makes the initializing
  case sound without a second mechanism (code).

- 2026-07-30 rationale: the design's own framing is containment, not
  elimination. Scattered cross-instance dereferences became four accessors in
  one file behind a generation check; what remains unsafe is the generated-code
  ABI, the epoch-validated raw caches, and the checkout primitive — the parts
  that are the JIT's actual job (sourced).

- 2026-07-30 (bc7cbb03) pitfall: a passing aliasing test proves nothing until
  the violating order has been shown to fail. Injecting a deliberate
  double-materialization made Miri reject it under both Stacked Borrows and
  Tree Borrows, which is what makes the passing suite evidence rather than
  decoration (code).

- 2026-07-30 (bc7cbb03) pitfall: two bugs this exposed were silent misdispatch,
  not traps — an exported funcref global whose local index the importer read as
  its own function, and a funcref in a GC array that made the reader dispatch
  its own decoy. Both are the default-local direction; a regression test for
  this class must assert a returned value, because a trap-only assertion passes
  against it (code).

- 2026-07-30 statement: the engine-native interpreter-to-interpreter call path
  is deferred. An installed host funcref hook drives cross-instance calls on
  the interpreter; without one the call traps by name, and a test pins that
  state so it cannot rot into unreachable behaviour (code).

## Moves

- 2026-07-30 (bc7cbb03) replaced [[pointer-identity]]: raw store pointers made
  cross-instance identity unfalsifiable — correctness rested on a Drop scan
  poisoning registry entries and on every dereference honouring that contract,
  nothing at all handled aliasing, and the interpreter refused the model and
  grew a second funcref identity beside it (code).
