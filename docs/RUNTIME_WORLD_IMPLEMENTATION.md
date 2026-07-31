# RuntimeWorld — Implementation Playbook

**Audience: the agent (or human) implementing the RuntimeWorld migration with
no prior context.** This document tells you why the refactor exists, how to
work in this repository, and how to execute and *verify* each migration step.
It deliberately contains no design rationale of its own — the design and every
decision behind it live in two files you must treat as the source of truth:

- **`docs/RUNTIME_WORLD.md`** — the reviewed design (1997 lines). Every step
  below references its sections. When this playbook and that document seem to
  disagree, the design document wins; report the discrepancy rather than
  improvising.
- **`docs/decisions.json`** — 37 recorded decisions from a seven-pass
  adversarial review (35 issues, all resolved on file:line evidence). When you
  are tempted to "simplify" something that looks over-engineered, search this
  file first: the shape you are about to re-invent was probably tried and
  refuted, with the refutation recorded.

Longer-horizon design history lives in `mcts_mem/silverfir/runtime.md` (and
its `.alt/runtime-store.md` — the original spec-accurate store this design
partially restores). Read it if you want to know why the codebase is shaped
the way it is; you do not need it to implement.

---

## 1. Why this refactor exists (rationale and intention)

**The problem.** The runtime's cross-instance identity token is a raw
`*mut Store`. Registry entries in `vm/link.rs` hand raw store pointers to
whoever resolves them later; correctness rests on `Drop for Store` scanning
the shared registries to null dangling entries, and on ~19 `unsafe` deref
sites each honoring that poisoning contract. The `Drop` scan handles the
dangling case; **nothing handles aliasing** — a deref can manufacture a
`&Store` while a `&mut Store` to the same store is live, which is undefined
behavior no test can detect. Separately, the interpreter refused this design
and built a second, incompatible funcref model (`FuncRefHost` +
`OpaqueInterpFunc` + hostref overloading), which is why `ref.test` answers
differently per engine.

**The intention.** The project owner's goals, verbatim in spirit: *minimize
`unsafe` while keeping performance — no regression on the JIT hot path or
interpreter dispatch.* The design achieves the safety half by **containment,
not elimination**: the ~19 unaudited derefs collapse into one audited
primitive (`checkout`) resting on two invariants checked in one place —
generation matched at checkout, and a checked-out slot cannot be freed. The
performance half is achieved by construction: `NativeContext`'s raw caches
and revision epochs are kept verbatim, same-instance calls are untouched, and
every new cost lands on paths that are already cold or already helper-routed
— with the two exceptions the design names and gates (see step 1).

**The historical arc, in three sentences.** The original store was
deliberately spec-accurate — one unified index space, linking = record the
index — and was split in 2026-02 because a unified *storage arena* relocates
entity cells under the JIT's cached raw pointers. The split's escape hatch
(`Rc` aliasing) worked, but its collateral was pointer-based identity and the
two-engine divergence. RuntimeWorld restores the spec's *index* role (flat
address spaces for everything that escapes as a reference) without repeating
its *storage* mistake (instances stay boxed and owner-structured; generations
make mortality safe).

**What is deleted at the end** (design doc, "What this deletes"): the `Drop`
poisoning scan, both populations of cross-instance unsafe derefs, the second
funcref model (`OpaqueInterpFunc`, the publication map, hostref overloading),
the engine-visible `ref.test` divergences, and the function-registry revision
counter. What honestly remains: the generated-code ABI's own `unsafe`, the
epoch-validated caches, and the one checkout primitive.

**A fourth category, added by this migration and worth naming so a future
reader is not misled by the list above.** `Caller` used to hold
`Option<&'a mut [u8]>` — borrow-checked, zero `unsafe`. That shape forced
`&mut Store` to span the host call, which is exactly the anti-pattern this
design exists to remove, so `Caller` now holds an `Rc` handle and builds its
slices with `from_raw_parts`/`from_raw_parts_mut`
(`vm/entities.rs:121`, `:134`), guarded by a `host_callback_borrowed: Cell<bool>`
test-and-set. The protocol is sound today: both engines preflight the flag
before running guest code (`jit/runtime/native_eval.rs`,
`interpreter/exec.rs`), the borrow errors on double entry, and `Caller::drop`
releases it on every path including unwind. But it is held together by two
hand-placed preflights that nothing forces to stay in sync, where the old
shape had the borrow checker doing it for free. Treat any change near it as
touching an invariant, not an implementation detail.

---

## 2. Ground rules for working in this repository

These are enforced by CI and by the project owner. Violating them wastes a
review round at best.

1. **Read `CLAUDE.md` and `AGENTS.md` at the repo root before your first
   edit.** The two rules that bite hardest:
   - **`reason = "..."` is never a fix.** `ci/lint_policy.py` findings are
     work items — delete the dead code, restructure, or gate precisely. You
     do not author `reason=` attributes or `ci/lint_suppressions.toml`
     entries on your own judgment, ever. A finding you cannot properly fix is
     left red and reported.
   - **cfg follows meaning.** `sf_jit`/`sf_interp` mark the engine *gates*
     only (module declarations, engine selector, instance dispatch, exports,
     `vm/link.rs`); engine-only code found in a shared file is *moved* into
     the engine's subtree, never gated in place. `sf_backend_*` is the arch
     layer's vocabulary; `sf_has_*` are target capabilities; `sf_ir_dump`
     etc. are JIT debug tooling — each may appear wherever its meaning
     genuinely applies.
2. **When implementation hits a structural obstacle, stop and report** with
   "the plan assumed X, the code actually does Y, here are the options" —
   do not silently work around it. The design survived seven adversarial
   passes precisely because divergences were surfaced, not smoothed over.
3. **Every claim about emitted code or runtime behavior needs captured
   evidence**, not code-reading narrative (see CLAUDE.md's root-cause rule).
   `docs/DEBUG.md` documents the dump mechanisms (`SF_NATIVE_DUMP_DIR`,
   `SF_JITDUMP`, `--interp-stats`).
4. **Commit hygiene**: work on a `dev/**` branch (current work lives on
   `dev/runtime-storage-rescope`, PR #18). One migration step = at least one
   commit; never batch two steps into one commit, because each step's green
   state is the rollback point. Dev-branch CI soft-fail is **not** a pass —
   inspect job steps.

---

## 3. The verification harness

### 3.1 The full matrix (run after every step)

All of these must produce **zero warnings and zero errors** unless listed in
§3.3. Rust warnings fail correctness CI unconditionally; there is no lenient
mode.

```sh
# Engine feature matrix, host:
cargo check --release -p sf-nano-core --no-default-features --features jit
cargo check --release -p sf-nano-core --no-default-features --features interp
cargo check --release -p sf-nano-core --no-default-features --features jit,interp
cargo check --release -p sf-nano-cli  --no-default-features --features jit
cargo check --release -p sf-nano-cli  --no-default-features --features interp
cargo check --release -p sf-nano-cli  --no-default-features --features jit,interp
cargo check --release --workspace
cargo check --release --workspace --all-features   # compiles memprof, jit-debug, call-trace

# Cross targets (toolchains installed on the reference machine):
for t in thumbv8m.main-none-eabihf riscv32imac-unknown-none-elf \
         armv7-unknown-linux-musleabihf riscv64gc-unknown-linux-musl \
         x86_64-unknown-linux-musl; do
  cargo check --release -p sf-nano-core --target $t --no-default-features --features jit,interp
done
cargo check --release -p sf-nano-core --target armv7-unknown-linux-musleabihf \
  --no-default-features --features jit,thumb2-test

# Unit + integration tests (default features = both engines):
cargo test --release -p sf-nano-core          # all green; ~440 tests

# Spec suites (the WAST driver needs the jit feature; jit,interp runs both tiers):
cargo build --release -p sf-nano-spectest --no-default-features --features jit,interp
./target/release/sf-nano-spectest --backend native   # JIT tier — expect 100.0%
./target/release/sf-nano-spectest --interp           # interp tier — expect 100.0%

# Lint policy and formatting:
cargo fmt --all --check
python3 -m ci.lint_policy      # needs Python >= 3.11; see §3.3 for the expected findings

# CI's own unit tests (run when you touch anything under ci/):
python3 -m unittest ci.test_lint_policy ci.test_correctness
```

Baseline numbers on this branch at handoff: spectest **257/257** (JIT) and
**174/174** (interp). The counts may grow if the testsuite pin moves; the
criterion is 100%, not the absolute number.

The WASI suite and QEMU cross-execution run in CI (`ci/correctness.py`; see
its module docstring for the job partitioning). The performance gates run in
`.github/workflows/performance-regression.yml` on every push;
`ci/performance.py` classifies regressions with a Bonferroni-corrected
directional test — a perf regression **blocks the step**.

### 3.2 Definition of done, per step

A migration step is done when: (a) its work items are complete; (b) the full
matrix above is green (modulo §3.3); (c) the step's own success criteria
(below) hold; (d) the perf gate on the pushed branch shows no regression;
(e) the commit message names the step.

### 3.3 Known-red items you must NOT chase or "fix" in passing

| Item | State | Rule for you |
|---|---|---|
| `python3 -m ci.lint_policy` reports exactly **7 findings**: `devices/*` (5: two `os_shim.rs` CodeArena fields, three kernel modules), `sf-nano-core/src/vm/jit/middle/ssa_ir/ir.rs` (SsaInstView), `sf-nano-core/src/vm/link.rs` (ExnInstance.fields) | Deliberately red, awaiting the project owner's item-by-item decision | Any **new** finding is yours; these 7 are not. Do not add reasons or manifest entries for them. |
| `cargo test -p sf-nano-core --no-default-features --features interp` fails 8 `tests/array_ops.rs` cases ("interp: GC is not supported") | Pre-existing; CI never runs tests in that feature combination | Ignore. Do not run interp-only `cargo test` as a gate. |
| CI cross job `cross-riscv32-linux` fails on nightly-toolchain linker stderr ("ignoring deprecated linker optimization setting", missing rustlib probe) | Pre-existing on `main`; environment, not code | Ignore unless the owner directs otherwise. |
| Interp fast-path guard comments vs emitted code (see §5, side task) | Pre-existing divergence, unverified | Handle only as the scheduled side task, not ad hoc. |

---

## 4. The implementer's card — invariants you must never violate

Tape this up. Every one of these was purchased with a found bug; the design
document has the receipts.

1. **Two containment invariants**: (a) generation matched at checkout;
   (b) a checked-out slot cannot be freed (per-slot in-use **count**,
   RAII-guard token whose `Drop` decrements).
2. **Materialization rule**: materializing `&Store`/`&mut Store` from a token
   is scoped, must not span a call, and must not overlap another
   materialization of the same slot. Two live *tokens* per slot is a required
   state; two live *materializations* is UB. Model: `do_struct_set`'s scoped
   block (good) vs `do_array_new_default`'s whole-body `&mut` (the
   anti-pattern shape).
3. **`checkout` takes the raw pointer via `&raw const/mut **b`** — never form
   a reference to the pointee inside the table borrow.
4. **Tripwires — one-line edits that keep compiling and break everything**:
   `Vec<Box<Store>>` must never become `Vec<Store>`; the instance's `Weak`
   back-reference must never become `Rc`; no `pub` API may yield an aliasing
   handle (`TableInst`, `GlobalInst`, `ImportedTableState`,
   `ImportedGlobalState`, `&Store`) to a container the stored reachability
   fact marks private.
5. **The encoding**: local funcref = local index (upward from 0); foreign =
   `FUNCADDR_TOP - funcaddr` (downward from `(1<<28)-2`); no tag bit;
   `(1<<28)-1` reserved. Both forms keep `is_special()` false. Any change to
   the four mask constants must run the round-trip test at both widths.
6. **Conversion primitives**: `absolutize(store_owning_frame_READ, v)` /
   `localize(store_owning_frame_WRITTEN, v)`; both total (identity outside
   their range); both resolve in the arena, **before** any checkout, never
   consulting a slot. Direction checkable by word, instance by argument — the
   argument is *not* always `owner_store`.
7. **The partition**: instance `I`'s local form lives only where only `I` can
   see it — `I`'s own operand/frame slots, or containers statically proven
   unreachable by others (private tables, private globals). Everywhere else:
   absolute, always. Default-absolute fails safe; default-local is silent
   misdispatch.
8. **Trap mechanism property (RAII soundness)**: the guard-page handler
   redirects the signal frame's PC to a normal error return so every Rust
   frame unwinds normally and `Drop` runs. If anyone changes that to jump
   past frames, checkout counts leak. Preserve it.
9. **The `indirect_info` bound check is mandatory on all three native-call
   backends** (x86_64, arm64, RV64) — the high-bits tests are blind below
   bit 32 and cannot substitute.

---

## 5. Side task, before step 3: the fast-path guard divergence

*(Tracked as an open investigation; independent of this design but touching
the same code step 3 edits.)*

The interp `call_indirect` handler comments claim tagged (special) handles
are filtered by high-bits tests. The emitted code disagrees: x86_64 and arm64
reject only bits ≥ 32 (`interp_gen/x86_64.rs:669-671`,
`interp_gen/arm64.rs:652-654`), and RV64 tests only the null sentinel
(`interp_gen/riscv.rs:1856-1866`). Whether this is exploitable **today** is
unverified.

Work items: (1) attempt a reproduction — a module whose *private* funcref
table receives a special-tagged handle (e.g. via `table.set` of a published
foreign funcref through `FuncRefHost`, or an externref smuggled by type
confusion if any path allows it) and then `call_indirect`s through that slot,
run on RV64 under QEMU (`ci/correctness.py cross riscv64` shows the
invocation pattern); (2) whatever the outcome, fix the comments to state what
the code does; (3) if reproducible, report before fixing — the fix interacts
with step 3's bound check. Success criterion: comments match emitted code on
all four backends, and a written verdict (reproducible or not, with the
attempt preserved as a test).

---

## 6. The migration, step by step

Each step below gives: goal, the design sections that specify it, work items,
and success criteria. **Read the referenced design sections before coding the
step** — they contain the line-anchored specifics and the reasoning this
playbook deliberately does not duplicate. Steps must land in order; 0 and 1
have no dependencies on each other but both gate everything after them.

---

### Step 0 — `tracked-alloc`: `downgrade` under `memprof`

*Design: "Migration plan" step 0.*

- **Goal**: `tracked_alloc::rc` supports `Weak` in both cfg arms.
- **Work**: add `downgrade` to the memprof `Rc` wrapper in
  `tools/tracked-alloc/src/lib.rs`, returning the re-exported
  `alloc::rc::Weak`. Accounting is forced: `downgrade` does not release;
  `upgrade` retains via the existing `from_alloc_rc`; add a test asserting a
  downgrade/upgrade/drop cycle leaves live bytes unchanged.
- **Call-site discipline (applies to all later steps)**: associated-function
  syntax (`Rc::downgrade(&x)`), and never name `alloc::rc::Weak` directly.
- **Success**: `cargo test -p sf-nano-tracked-alloc` green in both arms
  (`--features memprof` and without); workspace `--all-features` check green;
  the new live-bytes test passes.

### Step 1 — Coverage baselines, against the UNMODIFIED tree

*Design: "Migration plan" step 1; "Performance expectations"; "JIT side
effect" (cost section).*

- **Goal**: the two silent-regression shapes become measurable **before**
  anything changes. Steps 3 and 4 may not land until this exists.
- **Work**:
  1. A benchmark for the shape *absolute-form funcref read, then called* —
     one benchmark covering exported-table indirect dispatch (model:
     `linking.wast`'s `$Mt`), integrated into the perf harness
     (`benchmarks/`, `ci/performance.py` — follow the existing benchmark
     registration pattern and `ci/bench_metrics.py`). Include a
     same-instance *direct*-call variant to isolate marshalling cost from
     dispatch cost later.
  2. A cross-instance GC spectest: two linked modules, one writes a funcref
     into `(array (mut funcref))` or a struct field, the other reads and
     calls it, **asserting the returned value** (a trap-only assertion would
     pass against the bug class this exists to catch).
- **Success**: both run green on the unmodified tree; the benchmark's
  baseline numbers are recorded in the perf pipeline (this recording *is*
  the baseline); the GC test is in the suite run. Record in the step's
  commit message which later steps are gated on these (3 and 4).

### Step 2 — `RuntimeWorld` + `InstanceId` behind the existing API

*Design: "The state split"; "Why this is sound: disjointness"; "Ownership
direction"; "The seam: checkout"; "The safety invariant lives on the token".*

- **Goal**: the world exists — one-instance worlds behind unchanged public
  API — with the full checkout mechanism, but nothing yet uses cross-instance
  ids.
- **Work**: `InstanceTable` (slots/generations/in_use, all the sketch's
  shapes), `InstanceBackref { table: Weak<...>, self_id }` carried by `JitInstance`
  and `InterpInstance` where `LinkRegistry` is carried today,
  engine-discriminated `InstanceToken` as an RAII guard, `checkout` with
  `&raw` pointer extraction, generation retire-at-`u32::MAX`, `free` erroring
  on nonzero in-use count. `Instance::from_module` wraps a private one-slot
  world. `RuntimeWorld` facade with `instantiate`/`free`/`invoke` over the
  same table.
- **Success**: full matrix green; **zero behavior change** (all suites
  identical); `memprof` build green (exercises step 0); a new unit test for
  each invariant: generation mismatch → `None`; free-while-checked-out →
  error; two tokens on one slot coexist; world drop with zero instances
  reclaims (live-bytes assertion under memprof proves no cycle).

### Step 3 — The function address space, the encoding, and the accessor narrowing

*Design: "Encodings" in full (all subsections), "What the JIT keeps",
"What the interpreter gains", "Keeping the static fact true", "The six
crossings", "JIT side effect", "Migration plan" step 3. This is the large
step; the design text for it is the largest share of the document. Do not
attempt it as one commit — it is one **step** with one green gate at the
end, but implement it as the sequence below.*

- **Goal**: funcref identity becomes world-based on both engines, with the
  range encoding, the partition, and the conversion surface — and the static
  privacy fact both computed and made unfalsifiable.
- **Work sequence**:
  1. `FUNCADDR_TOP`, the two total conversion primitives on
     `InstanceBackref`, and the encoding round-trip test over both range
     endpoints at both widths (`ref_to_machine_raw`/`machine_raw_to_ref`).
  2. Replace `FunctionRegistryEntry` with `FuncEntry { owner, local_index }`
     **and** re-index `function_views` by local function index in the same
     change; delete `cached_function_registry_revision` and
     `Store::function_registry_revision` outright.
  3. Re-target the by-handle validate branch's `then_edge` from
     `trap_invalid_ref` to the runtime helper (edge plumbing with carried
     args — see design for the sibling-branch pattern). Add the
     `indirect_info` bound check to `EnterState` + all three native-call
     backends.
  4. The per-container reachability facts (`table_reachable`,
     `global_reachable`, funcref-type guard on globals), computed at
     instantiation on both engines; retag helpers for writes into reachable
     tables/globals (JIT: helper-routed lowering only for reachable
     containers — `TableDispatchMode::Generic` is NOT a proxy; interp: the
     listed sites). Fix the two divergent producers (element-segment
     materialization, const-expr `ref.func`) to consult their destination.
  5. The six crossings: the four `Ref`-arm conversion functions become
     instance-relative (pass the store owning the **frame** — fix
     `invoke_runtime_target`'s two sites to pass the caller's store); native
     evaluation returns absolute `Value`s and deletes the owner-relative
     `normalize_machine_raw_in_store` composer; interp localization in
     `call_host` (raw u64 slots) and the two
     `FuncRefHost::invoke` call sites; `function_handle_at` becomes a
     conversion site minting the absolute form; fix the stale
     `interp_imports.rs` header.
  6. The accessor narrowing: sharing accessors' precondition = the stored
     fact; `JitInstance::store`/`store_mut` → `pub(crate)` with
     `function_has_native_code` for the one external caller; interp
     accessors filtered; delete `table_elements_at`/`replace_table_elements_at`.
  7. The instantiation window: id reserved up front, slot `Vacant` until
     `init_result` returns.
  8. Interp side: delete `published`, `OpaqueInterpFunc`, hostref
     overloading; `ImportedFunction::Linked` gains identity; **calling** a
     foreign funcref in an interp world traps with a stated error (named
     deferred state — do not half-implement the call path).
  9. The two-armed mechanical audit (both greps from the design) — run it,
     fix what it finds, and include the grep outputs in the PR description.
- **Success criteria**:
  - Full matrix green; both spec suites 100%.
  - **`linking.wast` watched by value**: the `:609/:610` assertions (104 and
    0xdead) still pass — this is the named regression case for the local-form-
    in-imported-table bug; a "pass" that trapped differently or a changed
    value is a failure even if the suite reports green overall.
  - Step-1 benchmark re-run: the exported-table dispatch cost change is
    *measured and reported* (a regression here is expected and was accepted
    as a gated trade — report the delta; the gate's flat benchmarks must
    stay flat).
  - New round-trip test green at both widths; the embedder round-trip test
    (export handle → import into second instance → call → assert callee
    identity) green.
  - Grep audits attached; `ref.test` parity test between engines (same
    module, same answers) green.

### Step 4 — GC entries to `InstanceId`; delete `Drop for Store`

*Design: "Migration plan" step 4; "The safety invariant lives on the token"
(audit scope); "Absolute by representation".*

- **Work**: `RefRegistryEntry::Gc { owner: InstanceId, gc_ref }`; re-point
  `resolve_struct_ref`/`resolve_array_ref` at checkout (one signature change,
  ~13 sites); GC read-side `localize` in `do_struct_get` and array siblings
  per the design; run the materialization audit over **every** site that
  materializes from a token (the invariant's own range — including
  instantiation, the runtime-call path, and the retag helpers, not just the
  GC resolvers). Delete `impl Drop for Store` **only when** the property
  holds: the ref registry no longer contains raw pointers (checkable by
  reading `store.rs`).
- **Success**: full matrix green; step-1 GC cross-instance test green **by
  value**; `grep -rn "\\*mut Store" sf-nano-core/src/vm/link.rs` returns
  nothing; the four legacy unsafe deref sites in
  `gc_type_check.rs`/`exec.rs`/`context.rs`/`entry.rs`/`native_eval.rs` are
  gone (grep for `\.as_ref()`/`as_mut()` on entry stores); unsafe-token count
  in `vm/link.rs` + resolution paths is zero outside `checkout` itself.

### Step 5 — Bound registration at the source

*Design: "Migration plan" step 5; "The 32-bit budget".*

- **Work**: register only escapable functions (code-section `ref.func` scan
  authoritative; the `InitExprs` validator arm over-approximates — see the
  design's load-bearing caveat); guard the unconditional per-call
  `refresh_cached_views` at the runtime-call return behind the existing
  revision check — after first verifying `cached_views_are_current` covers
  everything a callee can mutate.
- **Success**: full matrix green; spec suites 100%; perf gate flat or
  improved (the per-call refresh guard should show up as an improvement on
  call-heavy benchmarks — report it); a test that a non-escapable function
  is absent from the address space while `ref.func` on a declared one works.

### Step 6 — Unified ref type test (Stage 2d)

*Design: "What the interpreter gains" (the `ref.test` bullet); mcts_mem
runtime.md facts on the divergences.*

- **Work**: one shared `ref_type_matches` over world provenance serving both
  engines; resolve the recorded divergences deliberately (hostref answers
  `Any` uniformly; the null rule stays at call sites as today; concrete-func
  matching via the owner's type context through checkout).
- **Success**: matrix green; suites 100%; an engine-parity test asserting
  identical `ref.test`/`ref.cast` answers across JIT and interp for the same
  module set, including the previously-divergent cases.

### Step 7 — Collapse the encodings (Stage 2c)

*Design: "Migration plan" step 7; "Encodings".*

- **Work**: retire the interpreter's separate null-normalization
  (`ref_to_slot`/`slot_to_ref`) in favor of the unified slot encoding; the
  TARGET32 wire form reconciles via the round-trip test from step 3.
- **Success**: matrix green including **all cross targets**; suites 100% on
  both engines; the round-trip test extended to the interp slot path.

### Step 8 — Failed instantiation carries an `InstanceId`

*Design: "Migration plan" step 8; "Embedder API".*

- **Work**: `InstanceInstantiationError::Partial { id, error }` replaces the
  `JitInstance`-carrying variant; slot stays occupied on failure (generation
  NOT bumped); interp path stops discarding its partial; update the spectest
  harness's `retained_failed_instances`.
- **Success**: matrix green; `linking.wast:592-611` (`0xdead` through the
  trapped module's table) green **by value**; the error type is
  engine-neutral (compiles in interp-only builds — this removes the
  `#[cfg(sf_jit)]` on the registry-aware constructor).
- **Dependency check before landing**: world drop reclaims occupied-failed
  slots (step 2's memprof live-bytes test extended to this case).

### Step 9 — Public API switch

*Design: "Migration plan" step 9; "Embedder API".*

- **Work**: `world.instantiate`/`world.invoke` public; `LinkRegistry` as a
  public type subsumed; `Instance::from_module` kept as the one-slot
  convenience; single-tier enforcement at `instantiate` with a named error;
  CLI and harnesses migrated.
- **Success**: full matrix green; both suites 100%; WASI suite green (CI);
  perf gate flat; `grep -rn "pub use vm::link::LinkRegistry" sf-nano-core/src/lib.rs`
  reflects the new surface; the nested-invoke shape from the design's
  Embedder API example exercised by a test (b calls a funcref owned by a,
  mid-execution).

---

## 7. After the migration

- The engine-native interp-to-interp `Linked` call path is complete.
- Follow-ups deliberately **not** in scope (do not implement without owner
  sign-off): cross-engine funcaddr calls; the per-instance funcaddr block
  optimization (measure first — step 1's benchmark exists for exactly this
  decision); entity storage into world arenas ("ids all the way down"); the
  three "Filed separately" fail-loudly items.
- Record the completed migration in `mcts_mem` (the `runtime.md` node's
  Items must be updated to describe the new live design, with Moves entries
  for what was replaced — see the mcts-mem skill/README conventions; the
  tree must never silently disagree with the code).
- Update `docs/RUNTIME_WORLD.md`'s status header from "approved for
  implementation" to "implemented at <commit>", and prune nothing else — the
  document is the design record.

## 8. Current branch state at handoff (2026-07-29)

- Branch `dev/runtime-storage-rescope`, PR #18. Stages 1-2 of the wider
  refactor are already landed there: the Store and its satellites are
  JIT-owned (`vm/jit/store.rs`), `vm/link.rs` is the registry meeting point,
  `ExnHeap` is deleted (exceptions are registry-owned), one shared
  const-expr evaluator (`vm/const_eval.rs`), engine cfgs at the gates only,
  and the lint-policy machinery is hardened. **The RuntimeWorld itself is
  not implemented — no step of §6 has been started.**
- CI on the branch: correctness green on all host + bare jobs; the lint job
  red with exactly the 7 findings of §3.3; `cross-riscv32-linux` red on the
  pre-existing toolchain noise; performance green.
- The reference verification numbers in §3.1 were taken on this branch.
