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

- 2026-07-30 measurement: the migration made the interpreter faster, not
  slower. Against the pre-migration baseline on x64: counter-global +61%,
  regex_redux +13%, tiny_keccak +11%, argon2 and nbody about +9%. The one
  benchmark the gate flagged as a regression, fibonacci-tail, is 5.43% faster
  when its upstream module is run locally with both binaries built on the same
  toolchain, and x86_64 emits no return_call handler at all, so its tail calls
  take the same shared path every other target does (code).

- 2026-07-30 pitfall: the performance gate reports false regressions. A
  calibration run comparing one commit against itself flagged bzip2 at -1.91%
  with 99.999998% stated confidence after twenty paired samples. Treat a single
  flagged metric as a prompt to reproduce locally, not as a finding; a direction
  that repeats across independent runs is the weakest signal worth acting on
  (code).

- 2026-07-30 pitfall: a local A/B between this branch and main compares two
  toolchains unless told otherwise, because main pins an older rustc in
  rust-toolchain.toml while the branch tracks stable. Set RUSTUP_TOOLCHAIN for
  both sides; the same confound reached CI before it was fixed there (code).

- 2026-07-30 statement: the engine-native interpreter-to-interpreter call path
  is deferred. An installed host funcref hook drives cross-instance calls on
  the interpreter; without one the call traps by name, and a test pins that
  state so it cannot rot into unreachable behaviour (code).

- 2026-07-30 statement: the deferred interpreter cross-instance path became
  sound only after the invocation boundary stopped carrying a token-derived
  materialization across the call. Its earlier soundness at the one checkout
  site was an accident of scope, not a general property: a second,
  runtime-chosen checkout requires ending the caller's materialization first
  (sourced).

- 2026-07-30 pitfall: the earlier negative-control record needs this
  append-only correction: the plain double-materialization control is rejected
  by Stacked Borrows and accepted by Tree Borrows. The two Miri steps are not
  interchangeable; for this control, Stacked Borrows supplies the
  load-bearing rejection (sourced).

- 2026-07-30 pitfall: migrating a real consumer exposed a public-API gap:
  `RuntimeWorld` host callbacks could not call a runtime-chosen
  `(InstanceId, function_index)` back into their world. `HostFn` has nowhere
  to carry state, while a thread-local raw world pointer would alias the
  embedder's live mutable world borrow. The resulting `WorldHandle` is an
  opaque, cheaply cloned weak table reference; each call performs a fresh
  generation-checked checkout, so expired worlds and stale ids return errors
  without a strong reference cycle (sourced).

- 2026-07-31 statement: the runtime access types were renamed to say what they
  do about lifetime rather than sharing one `Handle` suffix across four roles:
  `RefHandle` to `RefValue`, `TagHandle` to `TagIdentity`, `InstanceHandle` to
  `InstanceBackref`, `WorldHandle` to `WorldAccess`. Facts and Moves entries
  above this line predate the rename and keep the old spellings, since the log
  is append-only (sourced).

- 2026-07-31 measurement: the migration cost `call_indirect` through a table
  that crosses the instance boundary. On arm64-darwin, funcref-exported-table
  fell from 541M to 6M calls/s on the JIT and from 21.0M to 16.1M on the
  interpreter; CI failed all eight native jobs on both engines. Direct calls
  were unaffected on both. The blast radius is exactly tables that are
  exported or imported: of the nineteen benchmark modules only funcref.wasm
  has one, and the same module rebuilt without `-Wl,--export-table` shows no
  regression at all (1210 vs 1222 M/s) (code).

- 2026-07-31 pitfall: for a reachable table the JIT's inline dispatch is not
  slow, it is unreachable. The slot holds the absolute form, the inline guard
  tests `value < function_views_len`, and the arena asserts the absolute and
  local ranges never overlap -- so the guard can never pass and every call
  takes the runtime helper. The helper did not get slower; `function_views` is
  byte-identical across the change and the pre-migration build simply never
  entered it. A fix must restore an inline path; making the helper cheaper
  cannot close the gap (code).

- 2026-07-31 pitfall: compilation metrics do not prove the emitted code is
  unchanged. Baseline and candidate report identical function, SSA, MIR and
  code-size figures across this regression, because the change is a branch
  TARGET: it moves no node and need not move a byte. Comparing those four
  numbers eliminates nothing (code).

- 2026-07-31 measurement: the interpreter's share of that regression was two
  unrelated causes, and the larger one is not about funcrefs. `Effect` and
  `StepExit` reached 144 bytes because `ExternalCall` carried a `PreparedCall`
  inline; both are returned by value from the slow path, so every slow exit
  copied 144 bytes. Boxing that variant takes them to 32 and recovers about
  three quarters of the loss, and it taxes every slow-exit-dominated workload
  rather than `call_indirect` alone. The remainder is the funcref cause: a
  self-owned absolute handle can never satisfy the local-range early-out, so
  each call fell through to the shared arena until each instance carried the
  reverse of its own identity map (code).


## Moves

- 2026-07-30 (bc7cbb03) replaced [[pointer-identity]]: raw store pointers made
  cross-instance identity unfalsifiable — correctness rested on a Drop scan
  poisoning registry entries and on every dereference honouring that contract,
  nothing at all handled aliasing, and the interpreter refused the model and
  grew a second funcref identity beside it (code).
