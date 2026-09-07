# Public API and first registry release

Base: main 0983d9e4. Working branch: codex/public-api-release.

For a compact view of the proposed release, scoped verification evidence and
remaining decisions, start with [the release review checklist](RELEASE_REVIEW.md).

## Accepted direction

Keep the existing Wasmtime-style embedding model: configuration, engine,
module, instance, callable handles and host callbacks. Preserve both execution
engines and low-memory targets. Reduce exposed implementation, not embedding
capability. No additional benchmark-specific optimization belongs in this work.

The author approved the following work on 2026-09-07:

- `memprof` is an internal optimization tool. It must not change embedding
  types, signatures or trait contracts in any supported engine configuration.
  The author subsequently authorized simplifying or deleting it: retain basic
  statistics only where inexpensive, and do not substantially change valuable
  runtime code to accommodate the profiler. Remove container wrappers inside
  the tool and retain existing phase/runtime-buffer hooks where possible.
- Public API changes require review, including additions. Updating a generated
  snapshot is not review approval. Agents must present the actual API diff and
  obtain explicit human acceptance before accepting a new stable baseline.
- Hide runtime import/type machinery and internal exception transport.
- Define validated/prevalidated input, handle identity and lifetime,
  memory borrowing and reentrancy contracts.
- Associate WASI context with the instance/host state.
- Prepare crate boundaries, packaging, documentation and downstream validation
  for crates.io and wasmi-benchmarks.

## Work and acceptance evidence

- [x] Capture current public surface with pinned rustdoc/API tooling.
- [x] Implement feature-specific candidate captures and core memprof parity checks;
      include both published packages in review evidence.
- [x] Implement explicit human review, including same-account PR authors.
- [ ] Configure and verify remote review protection; accept the initial API baseline.
- [x] Use standard public allocation types and close implementation-type leaks.
- [x] Enforce callable/reference ownership and lifetime at the public boundary.
- [x] Define validation and prevalidated input contracts.
- [x] Replace ambient WASI context at the embedding boundary.
- [x] Prepare both registry packages and verify unpacked downstream consumers;
      keep CLI and development tools non-publishable.
- [x] Document features, target support, MSRV, embedding/memory examples and
      current engine limitations; provide the wasmi-benchmarks adapter migration.
- [ ] Resolve the reported interpreter-only and bare-metal warning ownership
      questions, and finish those configuration gates without suppressions.
- [ ] Resolve the full-startup regression and finish release-wide correctness,
      differential performance and final packaged-source verification.
- [ ] Present final API diff, reviewable PR and exact release artifacts.

Publication is a final action against a concrete version and package set.
Nothing has been published by this work yet.
The evidence below is a chronological work record: earlier measurements, API
sizes and resolved failures describe their named stage, not the final source.
Successful local configurations do not certify the unfinished release gates.

### API review workflow implementation

The new public-api workflow extracts five x64 profiles, always checks their
memprof counterparts, and compares the actual base and head API once review
tooling exists on main. The initial introduction requires review of the entire
candidate surface. A protected public-api-review environment owns explicit
human acceptance; edits to snapshots or labels cannot approve a change. The
owner may review PRs opened by their own account. The final required-status job
checks capture/review results and rejects a stale PR head. Workflow/policy
changes also require review. See PUBLIC_API_POLICY.md for the configuration and
the ordinary trust boundary of an in-repository CI workflow.

Local evidence: 131 CI unit tests pass, actionlint 1.7.11 accepts the workflow,
and a fresh five-profile capture remains warning-free with exact memprof parity.
The default candidate is still 880 lines. An attempted historical pre-policy
main capture reported private rustdoc links; it is not a passing baseline, and
no warning suppression was added. The initial policy introduction explicitly
reviews the full candidate instead of claiming an accepted historical API.

Remote enforcement is **not enabled yet**. Public repository metadata confirms
environment reviewers are supported, but the environment listing contains only
github-pages. The current gh session is unauthenticated and CUA reports no
available browser; no remote environment or branch-protection setting has been
changed. A real PR run and human acceptance are still required.

### Package boundary preparation

Prepare two registry packages: sf-nano-core and its small internal dependency
sf-nano-tracked-alloc. Keeping that helper avoids changing runtime call sites
solely to remove a profiling dependency. CLI, spec/WASI harnesses, foldsim and
the HTML report tool are non-publishable. The core's path dependency now also
carries its registry version. Both packages have readmes, repository metadata
and copies of the already-declared MIT/Apache-2.0 license texts. The core package
includes its interpreter build generator, source/tests/examples, and excludes
the temporary source audit document.

Cargo's multi-package local-registry verification packages and builds both
archives without warnings: helper 11 files / 11.6 KiB compressed; core 308 files
/ approximately 1.2 MiB compressed. These are dirty-tree preparation artifacts,
not approved release artifacts. The new core readme's embedding example runs
as a doctest; it and the unsafe-constructor compile-fail test pass (three older
documentation tests remain ignored). MSRV, clean release packaging and an
independent downstream consumer remain to be verified. crates.io availability
queries currently return HTTP 403, so no name availability or published-version
baseline has been asserted.

An independent application outside the repository compiles the unpacked crate
sources and calls both engines successfully, with WASI and memprof also enabled.
It uses version dependencies with temporary patches pointing only to the
unpublished extracted packages; this verifies packaged consumer behavior, not
availability on crates.io. Cargo's own package verification uses its local
staging registry to resolve the two packages together.

The supported compiler floor is now Rust 1.94. ARM64 all-feature contract tests
pass all 16 cases; x64 all-feature checking passes, both without warnings. The
same contract command is a lightweight correctness/msrv job using the existing
warning-failing runner. 132 CI unit tests pass and actionlint accepts both
changed workflows. API feature evidence also records edition and rust-version,
so changes to either require review.

Rust 1.89 was tested first: ARM64 and the unpacked consumer passed, but x64
rejected three existing CPUID calls because that compiler's intrinsic signature
was still unsafe. No runtime compatibility branches or warning suppressions
were introduced to support it. Version 1.94 is the supported/tested floor, not a
claim that every intermediate older release was exhaustively ruled out.

Remote setup remains pending. Automatic approval review rejected a proposed
noninteractive use of Git's credential helper for GitHub API authentication as
unintended credential access without specific authorization. That command did
not execute; no workaround was attempted. The author has been asked to choose a
normal gh login, explicit narrowly scoped authorization, or deferral of remote
settings. Unrelated local preparation continues.

The final capture after packaging metadata changes again passes all five
profiles and memprof comparisons without warnings; printed signatures/traits
are unchanged and the feature contract now records edition 2021 / Rust 1.94.
Pinned nightly reported a new manifest warning for explicit readme fields that
Cargo already infers, so those redundant fields were removed. The readmes stay
included; no lint setting or runtime code was changed for this fix.

## Initial findings

`cargo package -p sf-nano-core --no-verify --offline` fails because its mandatory
path dependency `sf-nano-tracked-alloc` lacks a registry version requirement.
The package description still names WebAssembly 2.0/interpreter-only operation;
repository/documentation metadata and package-level licensing/readme files
need attention. The file list includes a temporary source-tree audit document.

The upstream wasmi-benchmarks adapter at the start of this work uses Git tag
0.5 and root exports Config, Engine, Tier, Instance, Import, Caller, Value and
WasmError. It explicitly selects JIT/guard-pages or interpreter and disables
parallel compilation. Keep those measured operations equivalent when migrating
the dependency to a registry version.

Public container leakage includes invocation results, FunctionType constructors,
Import fields, host exception arguments and module/type builders. Changes to a
single return signature are insufficient to establish memprof API parity.

Func currently carries only an index and arities. Its association with an
instance must be enforced before dispatch. Import state and internal HostThrow
transport currently expose implementation types. WASI currently uses a thread-
local context setter. These are concrete contracts to resolve before release.

## Implementation evidence, 2026-09-07

The initial API capture found feature-dependent container signatures and trait
contracts in all five profiles. Replacing the profiler's custom containers with
direct `alloc` re-exports removes every tracked-allocation API leak and makes
all five memprof comparisons identical (default, JIT, interpreter, both, WASI).
The candidate capture passes without compiler/rustdoc warnings. An accepted
release baseline and enforced review workflow are still outstanding.

Compatibility conversions previously drafted in invocation methods and the
FunctionType constructor have been withdrawn. Engine execution/compilation code
does not need to change for the profiler. Existing phase and runtime-buffer
hooks remain. The core test binary installs the diagnostic allocator so the
existing isolated world-lifetime test measures real heap releases, rather than
becoming vacuous after wrapper removal.

The simplified tool retains allocator live/peak bytes, traffic/counts, separate
explicit runtime memory, and bounded phase records; it drops type attribution,
backtraces and historical timeline reconstruction. Global allocator tests cover
standard containers, recursion exclusion and concurrent workers; direct tests
cover failed realloc, pause, reset and stale diagnostic handles. The workspace
test run with core/CLI memprof enabled passes, including 581 core unit tests and
the isolated world-lifetime assertion, with no compiler warnings.

Independent public-contract fixes bind Func to its originating instance and
InstanceId to its originating world. External tests reproduce and reject
cross-instance, cross-world and reused-slot handles. Broader reference ownership,
input validation and WASI contracts remain work in progress.

Additional single-engine checks are not all green. Pure-interpreter
`--test array_ops` fails all eight GC tests because GC is unsupported there;
pure-JIT `--lib` emits dead-code warnings for the interpreter-only test hook
`Decoder::predecode_fast_disabled` and `disable_predecode_fast_for_test` in the
shared decoder. Both findings reproduce on an untouched checkout of main
0983d9e4. Leave them visible for the engine/test-ownership work; do not suppress
or mix them into the profiler change. The no-feature allocator tests pass.

A real CLI compile-only smoke run produces an HTML report with nonzero heap
counts and phase records. Basic heap/runtime peak counters have dedicated tool
tests, including preservation after the live allocation is released.

### Opaque import/error boundary

Import fields and the ImportValue/ImportedFunction/ImportedTagState variants are
now private to the crate. ImportedTableState and ImportedGlobalState retain
opaque shared identities; their underlying tables/global cells and type-context
fields are private. Named imports expose only module()/name() accessors.

WasmError is an opaque object with the existing lowercase constructors and
classification/exit accessors, new exception()/exception_tag()/exception_tag_name()
accessors and core::error::Error support. The old inbound HostThrow representation
is crate-private. Host code still uses Caller::throw. The wasmi-benchmarks adapter
must change WasmError::Trap(message) to WasmError::trap(message); no timing boundary
change is required for this migration.

Seven external-contract tests and the full workspace test run pass without
warnings. The first five-profile opaque-API capture removed 56 listed API lines
per profile and retained memprof parity; candidates remain unapproved.

Visibility contraction exposed additional single-engine ownership findings:

- Interpreter-only: TableInst's limits/from_shared/clone_shared_elements/size,
  GlobalInst's from_shared/clone_shared_cell, ImportedTableState.type_ctx, and
  ImportedGlobalValue.linked_function are not read in that configuration.
  Do not solve these by restoring public fields or adding engine cfgs in shared
  files. Resolve them with the remaining opaque export/linking API and engine
  ownership work. These five warnings were unfinished at this capture; the
  subsequent export/ownership pass below resolves them without suppressions.
- JIT-only: the host imported function's source type_index was ignored at link
  time. A new test proved the JIT incorrectly accepted equal signatures at
  different positions in a recursive type group. Consuming the existing index
  in the existing contextual type check fixes the failure. Both engines now
  reject the mismatched position and accept the matching position.

Source type-context constructors and module implementation introspection were
still public at that point. The following pass replaces cross-module binding
construction; module parsing/introspection still needs contraction.

### Opaque exported bindings and engine ownership

Instance::get_export(name) and Instance::exports() now produce opaque Extern
objects, consumed by Import::new(module, name, value). They carry source-world
identity and private type metadata. Importing one into an unrelated world fails
before instantiation. Shared storage remains aliased, while invoking a function
whose source instance was freed fails safely. New external tests cover host
function re-exports, tags, shared memory/global/table state, growth between
export capture and linking, stale function owners and foreign worlds.

The WAST runner now binds registered exports through this API. Its fixed 128
forwarding slots and about 500 lines of manual export scanning/binding code have
been removed. Public constructors accepting raw source TypeContext/type indices
and shared-state wrappers are removed. Metadata-forging negative fixtures live
in internal test support. The obsolete synthetic-host-function global import
is replaced by an ordinary host function exported through a shared global,
tested on both engines. Public shared-memory/table/global implementation getters
are removed as well.

Two reproduced linking failures are fixed: the interpreter now checks current
shared memory/table sizes at link time, and reference containers compare types
through both modules' contexts rather than treating equal numeric indices as
equal types. Negative tests reject different function types both numbered zero;
positive tests accept equivalent types at different indices. JIT imported/linked
functions retain their declared type index for later re-export.

The visibility audit's ownership issues are resolved by placing JIT wrapper
construction/guard queries in its existing entity extension module, heap memory
creation in the interpreter, and engine-specific fixtures in their respective
test support modules. Shared current-limit queries remain shared. The old
predecode differential-test flag now belongs to Predecoder; interpreter stream
accessors live in its subtree. No engine cfgs or warning suppressions were added
to shared runtime files to hide these diagnostics.

Current native ARM64 evidence: pure-interpreter and pure-JIT library builds and
unit tests are warning-free; the workspace suite passes with 581 core unit
tests; release spec runs pass 260 JIT and 175 interpreter files using the existing
support exclusions. Five pinned x64 API profiles retain memprof parity. The
default candidate currently contains 2,936 API listing lines; it is evidence for
review, not an accepted baseline. The separate GC integration-target selection
issue in interpreter-only `cargo test --tests` remains to be handled.

An explicit core `--features memprof --lib` run also passes all 581 unit tests
without warnings, including the isolated assertion that freeing a world leaves
no tracked heap allocations. This checks actual allocator behavior in addition
to the feature-parity snapshots.

### Remove obsolete module construction and harness synchronization

The runtime no longer uses ModuleBuilder; only three JIT test fixtures retained
it. Those fixtures now parse ordinary Wasm, and the unused public builder and
its unchecked Module construction path are removed. Module input validation is
still incomplete: removing this construction path does not make Module::new a
full validator.

The WAST runner no longer reparses stored module bytes to copy globals before
and after calls. Registered Extern objects already preserve shared state during
start functions, nested cross-module calls and traps. Its named-but-unregistered
module fallback also no longer fabricates successful no-op host imports. New
tests cover registration being required, start-function writes, immediate
cross-module visibility and writes preserved after traps, for both engines.
The JIT native-code diagnostic resolves an export name directly, replacing its
public raw-index argument and the fixture's module-internal scan.

Native ARM64 checks for this pass: 31 spectest unit tests and 580 default-feature
core unit tests pass without warnings. The difference from the 581-test workspace
count above is the validator feature enabled by the spectest workspace member.
These removals serve the public API boundary, independently of memprof; they
do not change the compiler, predecoder or execution loops.

The final release spectest build is warning-free and passes 260/260 JIT and
175/175 interpreter files with unchanged support exclusions. Separate pure-engine
library checks are also warning-free. The refreshed five-profile API capture
passes memprof parity; the default candidate is now 2,904 listed lines, 32 fewer
than the preceding capture. These candidates remain unapproved; CI review
enforcement and the remaining module/type surface are still outstanding.

### Validated and prevalidated input

Module::new now parses and validates every module, independent of engine or
feature selection. The optional validator feature and public Validator wrapper
are removed. Module is also exported at the crate root. Its owned, immutable
input representation carries validation forward into Instance::from_module
and RuntimeWorld::instantiate; both avoid another validation pass. The obsolete
JIT-only SIMD recheck is removed because both Module constructors already check
target support.

Module::new_unchecked is explicitly unsafe. Its caller must establish semantic
validity of the exact binary, including function bodies and all type/index
references. Encoding WAT or successfully parsing bytes is not sufficient.
Parsing/target checks still run, and later instantiation still checks imports
and resource limits. External tests cover invalid bodies, local indices,
constant expressions, final supertypes, concrete function-reference equality,
valid GC reference equality and unreachable polymorphic stacks, retained input
ownership, and missing imports through the prevalidated route.

This exposed invalid WAT in existing low-level fixtures. The subtype fixture now
declares a non-final supertype. Function identity fixtures compare returned
handles in Rust or call the caught reference, preserving their original identity
and dispatch assertions with valid Wasm. The validator's context-free eqref
helper wrongly accepted every concrete type, including functions; REF_EQ now
uses the existing context-aware operand check. The type rule is specified in
the [WebAssembly reference instruction validation rules](https://webassembly.github.io/spec/core/bikeshed/#valid-ref.eq).

Evidence: 581 core unit tests and all integration targets passed in the workspace
run; its remaining spectest fixture failure was corrected and all 31 spectest
unit tests then passed. Release spec runs pass 260/260 JIT and 175/175 interpreter
files with existing exclusions. Interpreter-only input-contract tests pass.
Thumb v8-M and RV32IMAC no_std JIT+interpreter library checks pass without warnings.
The Thumb check initially exposed a pre-existing pass-local guard accessor that
was unused without guard-page support. Its existing capability condition now
covers the method, matching the backing field and both callers; the always-false
unused alternative is removed. No lint suppression was added.

The five x64 API candidates retain memprof parity. Validator behavior is now
part of the safe API, not a feature contract. This adds validation cost to safe
startup compared with the old default build. Startup performance and downstream
benchmark timing still need measurement and explicit reporting; no benchmark
has been silently switched to unchecked input and no threshold was relaxed.

### Private module and decoder surface

The core module, opcode and decoder namespaces are now crate-private. Module
exposes owned construction and its diagnostic name at the root, without parsed
entity/type tables or raw decomposition. FunctionType remains a small public
signature object. HostCallback and FunctionInst are implementation details;
FunctionInst now lives in the JIT entity module, which owns that representation.
The interpreter's retained-module queries live in its module_view extension.
Type-context subtyping and binary parsing helpers no longer leak into the
embedding API.

foldsim now depends directly on wasmparser, not on sf-nano-core. It validates
input and reads module signatures and instructions independently; its opcode
byte constants describe the statistical cost model, not a second decoder. Six
real inputs (CoreMark, Fibonacci, c-ray, SHA256, bzip2 and funcref) produce
byte-identical reports before and after migration, including the aggregate.
Dedicated input tests cover recursion groups containing non-function types,
imported/local function index alignment, and rejection of invalid function code.
The tool is explicitly non-publishable. The spectest null-value comparator no
longer constructs an empty internal TypeContext for context-free expectations.

Visibility reduction exposed genuinely unused parsed-module constructors,
layout calculators, code-offset storage, a printer that only adjusted an
indent counter, and 502 numeric opcode aliases. These are removed; the old
layout-calculator unit test covered only the removed unused helper. Passive
data segments no longer carry a fictitious memory index. No execution-loop or
register-allocation change is part of these removals.

Opcode conversion now constructs the declared enum variants in a match instead
of transmuting their numeric discriminants. A standalone optimized comparison
of all four actual opcode tables found identical ARM64 instruction sequences
(three functions were folded to symbol aliases); x64 folds the FB/FC paths to
aliases and uses a shorter table lookup for the plain opcode path. This is
compiler-output evidence, not a throughput measurement; differential startup
measurement remains required.

The initial private-surface workspace run passes all tests without warnings
(580 core unit tests, 31 spectest tests, plus integration/tool tests). Five pinned
x64 captures preserve memprof parity. The default listing shrinks from 2,932 to
880 lines (interpreter 839, JIT 852, both 880, WASI 955). These are candidate
surfaces, not human-reviewed stable snapshots. Public reference representation,
remaining runtime escape hatches, WASI context and API review enforcement are
still unfinished.

The follow-up workspace run remains warning-free, and the final release spec
runs pass 260 JIT and 175 interpreter files. Independent native pure-JIT and
pure-interpreter library checks are warning-free after the ownership moves.
Thumb and RV32 checks are **not passing**: each reports the same two warning
groups for SIMD-only Immediate variants and WasmOpcode::FD, whose constructors
are absent on non-SIMD targets. The diagnostic cluster is recorded before any
new capability-boundary change. Following AGENTS.md, the proposed consolidation
of JIT SIMD decoding and capability-scoped private variants has been presented
to the author for a design decision; no warning suppression or per-arm cfg
patches have been applied. This does not block unrelated release preparation.

### Import-owned WASI context

wasi_imports now consumes an opaque WasiCtx built by WasiContextBuilder. Every
generated callback retains that context through Rc<RefCell<_>>; no thread-local
setter, take operation, or ambient WASI state remains. A separate import set
isolates an instance, while cloned/reused imports intentionally share their
context. Preview1 and legacy namespace aliases in one set share state. The CLI
and embedding example pass the context into import construction explicitly.
PreopenDir, FdEntry, descriptor maps/counters and the raw context constructor
are private. Existing syscall behavior is implemented as context methods;
there is no execution-engine or compiler change in this step.

All 42 syscall imports carry explicit WebAssembly signatures. They were checked
against the raw preview1 bindings in wasi 0.11.1+wasi-snapshot-preview1; malformed
guest import signatures now fail linking rather than reaching callbacks with
incorrect argument/result slices. Callback context borrowing uses try_borrow_mut
and returns a trap on a conflicting borrow. Normal returns and errors release
the borrow. Dropping the last retaining import/instance releases the context.

Tests cover distinct arguments/environment/stdio-close state on one thread,
intentional sharing across namespace aliases, nested calls across both engines,
separate preopened directories containing different files, invalid syscall
signatures, borrow-conflict recovery, and context release via a Weak witness.
The workspace passes without warnings, and the interpreter-only WASI contract
tests pass. Actual CLI/harness runs pass all 72 WASI tests under each engine,
with zero skips. Five x64 API captures retain memprof parity; the WASI candidate
shrinks from 955 to 909 lines, while the default candidate remains 880.

### Guarded host memory access

A safe external test reproduced an unguarded embedding boundary: holding the
slice returned by Instance::memory did not stop a cloned WorldAccess from
executing the same world. The reproducer invokes a no-op, so the failing test
itself does not write through a live shared slice. The API previously permitted
memory writes or growth by the same route.

Instance::memory / memory_mut now return MemoryView / MemoryViewMut in Result.
These own the backing and a world borrow: multiple readers may coexist, a writer
is exclusive, and execution/instantiation in that world fail while a view lives.
The view remains valid if the source instance or world is released. External
views cannot be acquired while the world executes, so a host callback cannot
stash a view that remains borrowed when guest execution resumes. Caller memory
access remains available during callbacks. This deliberately follows the store
borrow scope of the Wasmtime-style API; it does not allow executing an unrelated
peer in the same world while any of its memory views is held. A separate world
has its own borrow state.

The state is maintained at embedding entry points; guest instruction loops and
compiled call conventions do not carry new checks. Public instantiation checks
also reject shared memory already borrowed by a host callback before applying
active data segments. The interpreter's existing callback-borrow preflight now
runs before dispatching imported host functions too. Obsolete raw-slice facade
helpers are removed; only scoped instance-table Miri fixtures retain private
interpreter test helpers.

Eight external tests cover cloned handles, all public invocation routes,
multiple readers, mutable alias rejection, shared-memory growth after release,
instantiation while borrowed, callback view escape, direct imported-host
reentry, linked reentry through a memory-less module, error cleanup, and backing
retention after free. They pass in dual-engine, pure-JIT and pure-interpreter
builds. The full workspace passes without warnings (582 core unit tests).
Both the new view/backing test and the existing instance-table token test pass
under Miri Stacked Borrows and Tree Borrows with strict provenance and no
compiler diagnostics. Miri workflow steps now check their captured stderr for
warnings as well as the required one-test result; a zero-test or warning-only
run cannot pass unnoticed.

Pinned-nightly Miri reported AtomicUsize::fetch_update as deprecated. The world
identity mint uses the equivalent checked compare-exchange loop, compatible
with Rust 1.94, instead of adding a lint exception or raising the supported
compiler floor. The MSRV correctness job now includes the new memory contract
tests and passes all 24 selected cases. CI's 132 unit tests and actionlint pass.
The new readme memory example also runs successfully as a doctest.

All five API captures retain memprof parity and are warning-free. The current
candidate line counts are default/both 912, JIT 884, interpreter 871, WASI 941.
These are still unapproved candidates. General RefValue provenance and raw-value
construction, remaining runtime escape hatches, cross-target warning clusters,
downstream benchmark integration, performance measurement and remote release
review remain unfinished; this memory fix does not establish the whole release
as ready.


### Host value boundaries and unused mutation APIs (2026-09-07)

A safe numeric reproducer showed that `Value::I64(7)` passed to an i32
parameter entered both JIT guest code and a directly exported host callback.
The interpreter adapter also erased the value's variant without checking it
against the signature. The pre-fix test log is
`/tmp/sf-host-types-before.log`; no malformed reference was dereferenced.

Both adapters now check arguments before converting them to slots. Host result
slots start as `Unknown`, and every result must match the declared signature
before execution continues. The common checker inspects non-null reference
objects through the existing world/type context instead of accepting their
caller-supplied `RefType` annotation. Null checks retain both reference family
and nullability. JIT host-throw payloads use the same checker; the interpreter's
existing contextual check was consolidated into it.

Host result checks run after arbitrary host code returns. `Caller` borrows a
scoped checker that captures the engine's existing access capability, and
materializes the instance body only for the actual validation. It retains
neither a store-body borrow across callbacks nor ownership that keeps a freed
instance alive. A first attempt to look up the instance through the world
failed `start.wast`: start runs before slot publication. The final implementation
uses the existing initializing/occupied access paths without changing when
instances are published. A start-callback regression covers direct imported
starts and guest starts receiving both numeric and reference host results. Internal reference
words, table element layout and emitted Wasm-to-Wasm call ABI are unchanged.
Boundary overhead has not yet been measured.

Removed five unused public methods: `Instance::global_at`,
`replace_global_at`, `append_host_function`, `as_jit_mut`, and
`with_interp_mut`. Raw global replacement could violate both declared type
and mutability; the old late-added host-function hook has no remaining user
after the spec runner adopted engine exports. Engine-specific public
introspection is read-only. Removed the now-dead private mutation adapters and
lease mutable-token accessor rather than adding feature gates or suppressions.
The private interpreter storage fixture writes its own existing cells directly.
The JIT runtime-call fixtures now own a real occupied world slot, matching the
lifetime contract exercised by production callbacks.

Six external tests exercise numeric mismatches across named/indexed/resolved/
world calls, side effects before rejection, direct and guest-mediated host
returns, missing results, reference kind/nullability, and malformed exception
payloads. Both-engine, interpreter-only, and JIT-only runs pass. The same tests
are included in the Rust 1.94 CI contract set. Workspace tests and the 30 MSRV
contract cases pass without warnings. The host-call marshalling unit test passes
Miri with strict provenance under both Stacked Borrows and Tree Borrows.
Final release spec runs pass 260/260 JIT and 175/175 interpreter files using
the unchanged exclusion lists. The freshly rebuilt CLI passes all 72 WASI
tests on each engine, with zero failures or skips. The targeted CI Python
test set passes 32 tests; lint policy, diff whitespace and both workflow
actionlint checks pass.
The existing instance-table Miri fixture also checks host results after reentry
under both models; CI reuses that existing test instead of adding another job.
Logs for the final implementation are `/tmp/sf-host-types-*-final.log`
(workspace log: `/tmp/sf-host-types-workspace-init-final.log`); the obsolete
world-lookup attempt is not counted as the final validation.

API candidates `/tmp/sf-api-host-values-final` remove exactly the five methods above
from the previous candidate (as applicable per engine). All five feature
profiles retain memprof parity, including auto traits. These are unapproved
candidates; no accepted snapshot, remote gate, package publication or PR is
claimed by these local checks.

### Existing JIT host-exception catch limitation — unresolved

The positive exception control uncovered a pre-existing JIT limitation. A
valid `Caller::throw(tag, [nullfuncref])` escapes to Rust instead of entering a
matching `try_table` handler. The interpreter returns the expected value.
This reproduces on unchanged main `0983d9e4`, independently built at
`/tmp/sf-host-exn-baseline`; log `/tmp/sf-host-exn-baseline.log`. The original
failing candidate test is retained as evidence in
`/tmp/sf-host-exn-debug.log`. Minimal Wasm:

```wat
(module
  (tag $tag (import "host" "tag") (param funcref))
  (func $throw (import "host" "throw"))
  (func (export "guest") (result i32)
    (block $caught (result funcref)
      (try_table (catch $tag $caught) call $throw)
      ref.null func)
    drop i32.const 7))
```

Bind `host.tag` using `Import::tag_typed_with_handle` with one funcref
parameter, and make `host.throw` return
`Err(Caller::throw(tag, vec![Value::Ref(RefValue::null(), RefType::nullfuncref())]))`.
JIT returns `Exception`, interpreter returns `[I32(7)]`.
JIT's semantic decoder resolves statically known local throws, but the runtime
call's `Thrown` status has no matching dynamic catch continuation in the
current lowering. This is not fixed by value validation. `Caller::throw` docs
now state the engine limitation. The passing payload tests validate a legal
uncaught throw and reject malformed payloads before a catch can run; they do
not claim JIT catch support. Resolve the implementation/support contract before
calling the release ready. No existing spec exclusion was changed.

Reference world provenance remains separate unfinished work: public
`RefValue::new`/raw reconstruction can still forge or transplant identities,
and value-based global imports need the same ownership/type boundary. Contextual
type checks alone do not establish world ownership. Do not claim reference
handle safety or release completion from the checks in this section.

### Opaque world-bound reference values (2026-09-07)

The reference construction gap described above is now closed at the embedding
boundary. Public `Value` and opaque `RefValue` are separate from engine values.
Only the public reference carries a world identity: internal slots, table
elements, GC storage and emitted Wasm call conventions retain their existing
word representation. Conversion is explicit; no public/internal slice or
allocation reinterpretation is used. This change serves the release's handle
contract independently of memprof.

Removed public raw reference constructors/accessors and raw Value conversions.
`Func::to_value` creates a usable opaque function reference. The new
`Instance::get_func_by_index` resolves only functions already declared escapable
(exports, elements or `ref.func`); a private non-referenceable function returns
None. Existing indexed invocation can still call such a function. This keeps
the engine's escapability optimization intact instead of registering every
function just for host introspection. The unused public LinkRegistry and
FuncRefHost interfaces and three registry/hook constructors were removed;
RuntimeWorld is the supported sharing interface.

Guest entry, callback results and value-based global imports reject foreign
worlds before looking up an encoded identity. Function and GC references also
require a live owner with the original generation. GC-to-extern conversion
retains that provenance. Actual referenced types, not user-supplied annotations,
decide compatibility. Start callbacks use the existing initializing-instance
capability. Exception results and fields retain their world identity, including
exceptions forwarded by host callbacks. Nulls and explicit integer host labels
remain portable; labels are bounded to 27 bits on every target and do not own
arbitrary Rust objects. Copies of handles do not keep instances alive.

The spec harness now resolves function-reference identities through the instance
instead of comparing encoded payload indices. The existing reentrancy/Miri
fixture exports its imported function so that reference identity exists by the
normal engine rules; it exercises the public callback conversion after reentry.
Seven external ownership tests cover foreign worlds, freed/reused owners,
forged type annotations, globals, callbacks, start, exceptions, portable labels
and GC-as-extern references (the GC case is JIT-only).

Validation of the final implementation: workspace tests pass without warnings;
JIT/interpreter host and memory contract tests pass; ownership tests pass 7/7
with JIT and 6/6 with interpreter only. Rust 1.94 passes all 37 selected contract
cases. The updated reentrant callback fixture passes Miri Stacked Borrows and
Tree Borrows with strict provenance. Release spec runs pass 260/260 JIT and
175/175 interpreter files with unchanged exclusions; WASI passes 72/72 under
each engine with no skips. Four doctests pass, including forbidden raw-reference
construction; the three pre-existing ignored examples remain ignored. Targeted
CI Python tests pass 33 cases. Logs are `/tmp/sf-world-values-*`; the initial
ownership assertion and Miri fixture failures are superseded by the `ownership2`
and `token-test2` logs, respectively.

Five API candidates at `/tmp/sf-api-world-values` remain warning-free and exactly
equal with and without memprof: default/both 870 lines, JIT 854, interpreter 830,
WASI 899. The API checker also rejects leaked internal engine value paths.
These are still unapproved candidates. Boundary conversions currently allocate
temporary argument/result vectors; their performance cost has not yet been
measured. Downstream/package verification must use these new sources, and the
existing JIT exception limitation, cross-target warnings and remote review
setup remain open. This section does not assert release readiness.

### Packaged wasmi-benchmarks adapter migration (2026-09-07)

Downloaded upstream wasmi-benchmarks at
`b361a36b09340781db6681582804e0e61f35f9af`, whose adapter uses Nano tag `0.5`.
The independent downstream consumer was adapted for the candidate registry
version `0.1.0`. The adaptation replaces the dependency declaration,
`WasmError::Trap` with `WasmError::trap`, and Option memory access with guarded
Result views. Configuration, single-threaded compilation, instantiation and
the named invocation used by the timed benchmarks are unchanged. The final
registry dependency change belongs in wasmi-benchmarks after the package/version
is approved and published; it has not been sent upstream. The unused proposed
upstream patch was removed from this repository's documentation directory.

Fresh Cargo package verification builds the two archives: helper 11 files /
11.6 KiB compressed, core 313 files / approximately 1.2 MiB compressed. An
independent workspace at `/tmp/sf-release-wasmi-adapter` copies the real upstream
adapter and benchmark-utils, and resolves version dependencies via patches to
these extracted archives only. It does not depend on repository source paths.
With both engines enabled it builds without warnings and passes numeric host
callbacks, memory read/write bounds, memory growth, missing exports and traps.
The actual upstream CoreMark module completes with a positive finite score
under each engine. This is a consumer correctness smoke, not a performance
comparison: concurrent build load and the single run do not establish a ratio.
Evidence: `/tmp/sf-world-values-package.log` and
`/tmp/sf-world-values-wasmi-consumer.log`.

The CI-pinned older upstream revision needs the same source-only adapter patch.
`ci/wasmi_adapter.patch` belongs to this runtime revision. The differential
driver applies it to this revision's isolated suite copy, while unchanged main
uses its original adapter. It records the patch SHA-256 and rejects conflicts;
the migration does not add runs, change timed operations, or build competitor
engines. The manual startup-ranking driver uses the same migration. The current
62 targeted CI tests pass, including absent migration, repeated application
and conflict rejection. The patch applies to the real pinned upstream source.

The separate interpreter-only packaged release smoke executes successfully but
is **not passing**: making internal Value private exposes unused `value_type`,
`to_raw` and `from_raw` methods. Their production callers are in JIT
instantiation; a shared unit test also uses `value_type`. The proposed local
fix is to keep these conversions with their existing JIT owner and adjust the
internal fixture, with no loop or call-ABI changes. The AGENTS.md ownership rule
requires a design decision first; approval has been requested and the warning
remains visible. Log: `/tmp/sf-world-values-wasmi-interp.log`. The successful
combined-engine build does not certify interpreter-only warning cleanliness.

The complete CI-pinned Criterion binary subsequently compiled without warnings
with only Nano JIT and interpreter enabled. Its real `--test` runs for
`execute/fibonacci-rec` and `startup/erc20` passed under both engines. Logs:
`/tmp/sf-world-values-pinned-suite-online.log` and
`/tmp/sf-world-values-pinned-smoke.log`. The earlier offline attempt failed
because Cargo needed an uncached optional Git dependency during workspace
resolution; it is not a passing build. Fetching public dependencies resolved
that environment issue without enabling competitor engines.

### Temporary embedding buffers and measured boundary cost (2026-09-07)

A local ARM64 release microbenchmark compared current work against unchanged
main `0983d9e4`: resolved empty calls, one-i32 identity calls and one-i32 host
roundtrips. Before allocation reduction, the identity medians were 62.7 to
127.5 ns for JIT and 89.7 to 204.3 ns for interpreter. These costs concern
embedding transitions, not the guest instruction loop.

The public conversion layer now uses a private, safe ValueBuffer: an empty
variant for zero values, inline storage for up to four values, and a Vec for
larger signatures. Both instance/world argument entry and callback argument/
result conversion use it; resolved calls also use it for temporary results.
No unsafe conversion, new dependency, engine slot change or extra guest-loop
check is involved. The inline capacity bounds the extra host-stack storage;
it does not limit supported signature sizes. Exception transport and the
public Vec-returning invocation APIs retain owned results.

The final local pilot alternated AB/BA order over 11 process pairs per engine,
with 1,000 warmups and 100,000 calls per case. All task-owned compilation and
test jobs had completed before these samples. Median elapsed ns per call:

| Engine / call | Main | Candidate |
| --- | ---: | ---: |
| JIT / empty | 32.2 | 48.3 |
| JIT / one i32 | 61.0 | 91.9 |
| JIT / host roundtrip | 233.4 | 301.7 |
| Interpreter / empty | 37.8 | 57.4 |
| Interpreter / one i32 | 88.8 | 144.6 |
| Interpreter / host roundtrip | 330.9 | 379.0 |

This removes much of the new allocation cost, but does **not** establish
performance parity with main. Remaining absolute overhead is approximately
16–68 ns in this pilot. No ASLR pinning or CI statistical gate was used;
these are local diagnostic medians, not a validated whole-workload regression
verdict. Full wasmi/CoreMark startup and execution comparisons remain required.
Sources/binaries and final raw samples are at `/tmp/sf-release-boundary-bench`
(`final-samples.json`, `final-summary.json`). Intermediate `buffer-samples.json`
overlapped a build and is not used for the table.

The new host-signature test covers 0, 1, 4, 5 and 8 arguments/results through
resolved and named calls, including invalid input rejection before callbacks
and retention of caller results on error. An initial 17-result fixture exposed
the interpreter's pre-existing eight-host-result limit; the final test spans
both buffer paths within that supported range, and the README records the
limit. No engine limit was changed or hidden with a feature exclusion.
Workspace tests pass without warnings; the final empty-buffer variant also
passes all 22 focused host/memory/reference cases. Both strict-provenance Miri
models pass the callback reentrancy fixture. Final release spec runs again
pass 260/260 JIT and 175/175 interpreter files. Logs are
`/tmp/sf-boundary-buffer-*`; `workspace-final` and `empty-tests` supersede the
initial wide-signature fixture failure. The previously packaged consumer
evidence predates this internal allocation change; final release packaging
must be regenerated after all pending edits.

The final Rust 1.94 contract run passes 38 cases with zero warnings/failures
(`/tmp/sf-boundary-buffer-msrv.log`); 62 targeted CI Python tests and lint policy
also pass. Release spec summaries with explicit logging are in
`/tmp/sf-boundary-buffer-spec-{jit,interp}-final.log`.

### Full ARM64 workload pilot and startup attribution (2026-09-07)

Remote main was rechecked and remains `0983d9e4`. The complete pinned upstream
workload set ran as 54 adjacent baseline/candidate pairs (27 workloads, two
engines), alternating process order and disabling macOS ASLR. Both binaries
were built with both Nano features; this is a local pilot, not a substitute for
the CI single-engine statistical gate. Compilation finished before measurement.
No competitor runtime was enabled and no performance thresholds were changed.

The 20 execution cases have elapsed-time geometric means of +0.21% for JIT and
+0.03% for interpreter, with individual changes from -0.84% to +1.33%. The seven
startup cases show +2.78% for JIT and **+94.36% for interpreter**. The latter is
an unresolved regression. Full medians, binary/source hashes, scope limitations
and normalized raw samples are in
`docs/release-evidence/arm64-api-workload-pilot.{md,json}`. Local commands and
Criterion raw artifacts remain at `/tmp/sf-release-workload-pilot/`.

Separate module-construction diagnostics establish semantic validity once,
then compare parsing the exact immutable validated bytes with parsing plus
validation. They do not change the checked upstream benchmark path. Added
validation cost is approximately 0.988 ms for bz2, 2.266 ms for pulldown-cmark,
51.802 ms for SpiderMonkey, 193.088 ms for FFmpeg, 0.098 ms for CoreMark,
0.399 ms for Argon2, and 0.085 ms for ERC20, accounting for most startup growth.
The Argon2 fixture is the upstream Rust-built `res/rust/cases/argon2/out.wasm`;
an initial attempt at a nonexistent `res/wasm/argon2.wasm` failed and was
corrected without repeating the already completed module samples.

A five-second sample of a task-owned FFmpeg validation process locates the
main work in Decoder::decode_one and FunctionValidator::on_op/type checking;
Immediate::clone also appears among the hot routines. Evidence and raw phase
timings: `/tmp/sf-release-validation-profile/`. Sampling ran concurrently with
a separate build, so its process throughput is not a performance result; it
is used only to identify call stacks. No validator optimization has yet been
applied. Skipping semantic validation is not an acceptable fix for safe loading.

### Diagnostic and limit surfaces (2026-09-07)

Removed the remaining public engine bodies/leases and their accessors:
InterpInstance, JitInstanceLease, Instance::with_interp and Instance::as_jit.
Instance::interpreter_stats now returns an owned InterpreterStats snapshot;
Instance::function_has_native_code returns the scalar JIT answer. The CLI keeps
its existing statistics output using that snapshot. Handler names are display
strings, explicitly not stable instruction identities, so no internal Op enum
appears in signatures. Collection is on demand; execution loops are unchanged.

Removed reset_native_runtime_state and the spec runner's global reset call.
CodeBuffer already unregisters its own trap ranges on reset/drop, and native
entry already resets its debug counter. Clearing all registrations could
invalidate another live instance's traps; it is now confined to the existing
trap-table unit-test module. An external test repeatedly traps/drops a peer and
confirms the surviving instance still reads and traps correctly. No signal
handler or code-buffer lifecycle implementation was changed.

Config and error types remain at the root; their duplicate public modules and
the implementation-limit constants were hidden. WASM_PAGE_SIZE is reexported
at the root. Limits::new/new_64 return the existing public WasmError, preserving
the previous error classification/messages. The internal LimitsError type,
conversion implementation and unused Limitable convenience methods were
deleted. Limitable, effective/default-limit machinery and the mutable is64
field are private; Limits::is_64 provides read-only width inspection.

Workspace tests, 41 Rust 1.94 contract cases and 64 targeted CI Python tests
pass without warnings. Rebuilt release specs pass 260/260 JIT and 175/175
interpreter files with no global reset or new exclusion. The CLI
`--interp-stats` smoke prints engine size and a named fallthrough pair through
the new snapshot. Evidence: `/tmp/sf-release-diagnostics-*`. Two initial local
visibility findings were fixed by deleting unused trait defaults and making
the internal opcode enum crate-private; the `workspace-final` log is the clean
run. Separate interpreter-only raw-value warnings remain pending as reported
earlier, so this does not establish all configurations as passing.

The five diagnostic-stage API candidates retain exact memprof parity: default
and both 757 lines, JIT 739, interpreter 727, WASI 786. The checker now rejects
all private vm/utils type paths as well as tracked-allocation paths. No reviewed
baseline has been accepted. Subsequent tag API edits below require a new capture.

### Exception tag identity contracts (2026-09-07)

A safe instantiation-only reproducer showed Import::linked_tag_typed could
rebind an existing i32 tag identity with an i64 signature. The JIT accepted it;
no mismatched payload was executed (`/tmp/sf-release-tag-signature-before.log`).
Removed that constructor. Import::alias now preserves the full original object,
signature, identity, state and world provenance while changing only its lookup
names. The interpreter's existing aliased-tag fixture uses this path and still
checks catching by identity. TagIdentity minting is private; its counter now
fails before exhaustion instead of wrapping and reusing an identity. A local
counter test covers the endpoint without modifying the process-global counter.

Host-created tags have no module type context, so their signatures accept
numeric and abstract-reference parameters. A host tag containing concrete
module-local indices is rejected at instantiation: copying the same index into
another module cannot establish the same type. Concrete tag parameters remain
supported through typed module exports. An external test links equivalent
function-reference types at different indices, rejects a different underlying
function type, and rejects a context-free host tag. The ordinary alias test
checks that a mismatched declared signature is rejected under both engines.
Final tag verification passes without warnings: workspace tests, 11 public
contract tests, 43 MSRV embedding-contract cases on Rust 1.94, 64 targeted CI
script tests, and release spec suites (JIT 260/260, interpreter 175/175 with
unchanged exclusions). The five API captures retain exact memprof parity:
default/both 755 lines, JIT 737, interpreter 725, WASI 784. These are review
candidates, not accepted baselines. Logs: `/tmp/sf-release-tag-*`.

Both fresh 0.1.0 archives pass offline Cargo package verification. A separate
consumer patches registry dependencies to the unpacked archives and builds
the actual upstream wasmi adapter. Numeric host calls, guarded memory, growth,
errors and upstream CoreMark complete with both engines, without warnings
(`/tmp/sf-release-tag-wasmi-consumer.log`). The CoreMark scores are smoke-test
output, not controlled comparative performance evidence. Pending interpreter
and cross-target warning ownership decisions and remote review setup remain
unresolved; these successful configurations do not make the release gate green.

### Registry and remote status recheck (2026-09-07)

Public crates.io metadata queries with a descriptive User-Agent now succeed:
both `sf-nano-core` and `sf-nano-tracked-alloc` return HTTP 404 with an explicit
"crate does not exist" response. Thus no published-version compatibility
baseline applies to these names yet. This is an availability observation, not
a reservation or publication. Responses are saved at
`/tmp/sf-crates-availability-{core,helper}.json`; the earlier 403/tool URL errors
are not treated as registry evidence. Version 0.1.0 remains a proposal.

The ordinary gh authentication check still reports no logged-in host. The
earlier credential-access auto-review rejection has not been bypassed; remote
API-review settings remain unchanged and awaiting the existing authorization
question. All release work remains uncommitted and no PR or package publication
has occurred yet.

### Borrowed validator immediates (2026-09-07)

The first bounded response to the measured validation cost removes the
per-instruction Immediate clone inside FunctionValidator. Helpers borrow the
decoded immediate and its branch-label, catch-clause and typed-select slices.
br_table visits the borrowed labels followed by its default label, preserving
the original checking order. Scalar fields are copied as needed. This changes
only validator data passing, not decoding, validation rules, engine ownership,
runtime layouts, public API or memprof integration.

Six alternating process pairs on each of the seven real startup Wasm inputs
show safe Module::new elapsed time falling 7.46–11.30% on native ARM64. This is
a local checked-construction diagnostic, not full startup or x64 performance
evidence. It reduces part of the previously measured interpreter startup
regression; that regression remains unresolved. Protocol, samples and binary,
input and validator source digests are recorded in
[the validator report](release-evidence/arm64-validator-borrow.md) and its JSON.

The modified source passes the full workspace tests without warnings
(including 583 core unit tests), release spec suites (JIT 260/260 and
interpreter 175/175 with unchanged exclusions), the lint suppression policy
and diff whitespace checks. Logs: `/tmp/sf-release-validator-borrow-*`.
No new suppressions, feature gates, dependencies or benchmark exceptions were
introduced. The package/API captures above describe the preceding tag stage;
final release artifacts still need refreshing when source changes settle.

### Avoid discarded validator operand collections (2026-09-07)

Most Context::pop_vals callers ignored its returned vector of actual operand
types. They now check and pop the same operands without collecting a vector;
br_table alone uses the original type-preserving collection before restoring
the operand stack between labels. Polymorphic Unknown handling and the order
of validation remain unchanged. This is a small validator-local change, not
an engine or memprof accommodation.

Against the borrowed-immediate candidate, safe Module::new elapsed time falls
another 1.92–7.17% across the seven real startup modules in six alternating
local ARM64 process pairs. Details and raw evidence are in
[the operand-collection report](release-evidence/arm64-validator-pop.md).
These diagnostic improvements do not themselves settle full startup regression.
The full workspace tests and release spec suites pass without warnings:
JIT 260/260, interpreter 175/175 with unchanged exclusions. Logs:
`/tmp/sf-release-validator-pop-*`.

### Include the published support package in API review (2026-09-07)

The API capture and protected review digest now include sf-nano-tracked-alloc
with and without memprof, plus its Cargo features, edition and MSRV. Its APIs
serve tools rather than embedding, but publishing the package must not leave
them outside review. Missing support evidence fails closed; changes to either
surface or the feature contract require the same human approval as core changes.
No tool or engine implementation changed for this review coverage.

Fresh pinned captures in `/tmp/sf-api-support-final` pass without warnings.
All five core surfaces exactly match the tag-stage candidates and retain exact
memprof parity (default/both 755 lines, JIT 737, interpreter 725, WASI 784).
The support surfaces contain 176 and 182 lines: its existing memprof-specific
AllocationHandle Debug/Drop and PhaseGuard Drop implementations account for
the difference. Both variants are independently reviewed; they do not appear
in the core embedding surface. The targeted CI unit suites pass 60 tests,
including new support-capture, review and missing-evidence coverage. Logs:
`/tmp/sf-release-support-api-*`. No snapshot has been accepted by an agent.

The workflow's capture-only manual trigger was removed so a standalone run
cannot produce the same successful status name as the required PR review gate.
The repository setup instructions now explicitly require PRs for main and
prohibit branch-rule bypass; post-push captures are audits, not pre-merge review.
Actionlint accepts the updated workflow. A normal browser-session recheck found
the Mac locked and inaccessible to CUA, so that route did not configure any
remote settings either. Ordinary SSH read access still verifies main at
0983d9e4; no credential helper extraction was attempted.

### Full startup follow-up and draft preparation (2026-09-07)

All fourteen startup/engine pairs were repeated against untouched main after
both validator optimizations. The interpreter's geometric-mean elapsed increase
is now 80.56%, compared with 94.36% in the earlier pilot. CoreMark changes from
0.089 to 0.184 ms and ERC20 from 0.087 to 0.162 ms. These are real remaining
costs of the safe input contract, not a passing performance result.

The JIT aggregate is +2.60% using a separate FFmpeg repeat. The first FFmpeg
pair overlapped a failed CUA state/unlock query and showed large outliers; its
raw result is retained as confounded. The reversed-order repeat without UI or
compiler activity gives +2.66%. These small JIT differences remain local pilot
evidence requiring normal CI confirmation. Both source digests and all samples
are recorded in [the startup follow-up](release-evidence/arm64-release-startup.md).
No validation bypass, benchmark boundary change or threshold relaxation was used.

The repository formatter initially found outstanding layout changes in the
release edits. cargo fmt applied those changes, and its full workspace check now
passes. Fresh post-format package verification builds both archives without
warnings (helper 11 files, core 314 files). Rust 1.94's embedding-contract gate
also passes without warnings. These remain provisional preparation archives;
the final approved release must be packaged from its clean reviewed commit.

A fresh independent consumer then builds the actual upstream wasmi adapter
against only the unpacked post-format archives. Numeric host calls, guarded
memory, growth, error paths and upstream CoreMark complete under both engines,
without warnings (`/tmp/sf-release-final-wasmi-consumer.log`). This confirms
packaged downstream use in the tested dual-engine configuration, not the
unresolved single-engine/bare-metal gates or registry publication.

### First draft CI and nested adapter-copy fix (2026-09-07)

Draft PR [43](https://github.com/mbbill/Silverfir-nano/pull/43) starts from
7d1f0c8c. The real x64 Linux API capture completes all core parity and support
profiles; its evidence digest is
`ae547697fc0da341801049d0329b3bbfe7a9038966d2c8a1b6ff1d15a0a224e2`,
identical to the local ARM64-host/x64-target capture. The subsequent environment
check returns HTTP 404, so the workflow remains failed and human review is not
complete. The [uploaded evidence](https://github.com/mbbill/Silverfir-nano/actions/runs/34140029191/artifacts/10025576234)
is still available for the exact head.

Correctness CI confirms the interpreter-only raw Value helpers and the
non-SIMD immediate/opcode warning clusters. It also exposes `Limits.default_max`
and `get_max`: their only readers are the two JIT memory/table grow helpers.
A local pure-interpreter check reproduces these warnings. The proposed
ownership change keeps declared limits/range validation shared and derives the
same effective caps in JIT grow; this additional decision has been presented
for author approval under AGENTS.md. No suppressions or engine cfgs were added.

All eight wasmi primary jobs fail building the unmodified adapter, before any
timing evidence. Their confirmation jobs have no selected measurements and do
not turn the failed primaries into passes. Although git apply returned success,
the copied suite lived below the runtime checkout without its own .git. Git
discovery selected the parent repository and silently skipped the suite-relative
patch paths outside the current-directory prefix. Standalone-copy validation
had not exercised this CI directory arrangement.

The migration helper now sets a per-command Git discovery ceiling at the suite
parent, so patches apply to the standalone copied tree. A regression test first
reproduces the silent no-op, then passes with the fix, also checking parent-file
isolation, repeat application and drift rejection. The actual pinned upstream
adapter copied below this worktree now matches the independently tested adapter
byte-for-byte after migration; the main baseline still receives no migration.
All 140 CI unit tests pass. No runtime source or benchmark timing boundary changes.

The separate x64 Linux CLI JIT performance job completes without a confirmed
regression: CoreMark is 19,237.5 versus 19,232.8, or -0.02% with pilot interval
-0.59%..+0.38%. This covers that configured CLI suite, not the failed wasmi jobs,
the single-engine warning gates or a new V8 comparison. Logs for this audit are
`/tmp/sf-release-ci-job-*.log` and `/tmp/sf-release-ci-adapter-{before,after}.log`.

Both packages also pass verification from clean commit 7d1f0c8c. Compared with
the independently tested unpacked packages, their only changes are clean VCS
metadata and the corresponding support-package checksum in core's lockfile.
Source and dependency selections are identical. The local checksum manifest is
`/tmp/sf-release-clean-package-evidence.json`; these packages are not approved
for publication and will be regenerated for the final reviewed revision.

### Linux startup CI and warning visibility (2026-09-07)

The 4030fa54 run successfully builds the migrated wasmi adapter and measures
both Linux interpreter startup suites. All seven workloads on each architecture
are marked REGRESSION by the primary job. x64 CoreMark rises from 156.558 to
273.042 us, and ERC20 from 145.141 to 241.451 us. ARM64 CoreMark rises from
106.029 to 203.385 us, and ERC20 from 103.613 to 183.988 us. The full primary
tables and exact job links are retained in
[the CI evidence](release-evidence/linux-release-startup-primary.json).
Independent-runner confirmation is still pending; a successful primary job
only forwards these regressions and is not a passing performance result.
These observations agree with the local safe-input validation cost investigation.

The completed x64 Linux, ARM64 Linux, Windows and Darwin correctness jobs all
fail the three-warning interpreter ownership cluster described above. Thumb
and RV32 bare-metal also fail the non-SIMD opcode/immediate cluster. RISC-V64
additionally reports the interpreter test-support memory accessors as unused.
These remain visible failures pending the ownership decisions; no engine
implementation or suppression changes were made to clear them. Policy, Miri
and the Rust 1.94 gate pass. API extraction succeeds but the protected human
review environment is still missing, so the API gate remains failed.

The wasmi identity build used Cargo's json-render-diagnostics mode, whose
diagnostics go to stderr rather than the stdout artifact stream. Its captured
stderr was discarded. The harness now prints it, retains a diagnostic log and
uses the existing CI warning parser to append an ACTION REQUIRED summary.
The change adds no builds and leaves timing and numeric verdicts unchanged.
A regression test exercises the real stdout/stderr protocol and verifies that
Cargo's aggregate warning tally is not double counted. All 141 CI tests pass;
a real interp-only identity build reports all three warnings and preserves
the exact compiled-runtime fingerprint. Lint policy and diff checks also pass.
This logging correction is local while the existing CI measurements finish.

### Inline validator control signatures (2026-09-07)

The validator formerly allocated an Rc<FunctionType> for every empty block and
an additional result vector for every single-result block. Its private control
signature now stores these two forms inline; indexed/multi-value signatures
continue sharing the module's canonical function type. All block/ref-type bounds
checks and control-stack rules remain unchanged. This is a validator-only
representation change, with no engine, public API or profiling changes. Inline
signatures trade some control-frame storage for eliminating those per-block
allocations; the indexed form still uses shared ownership.

Six alternating before/after process pairs on all seven startup input modules
show safe Module::new elapsed improvements of 1.45% to 5.74%. These local
measurements cover validated module construction, not complete instantiation or
the remaining CI startup regression. Samples and source/binary/input hashes are
in [the signature evidence](release-evidence/arm64-validator-signature.md).
Core tests pass 684 cases without compiler warnings. Full spec runs explicitly
report JIT 260/260 and interpreter 175/175 with existing exclusions unchanged.
Formatting, diff and lint-policy checks pass. The single-engine ownership
warnings are separate unresolved gates; this change does not suppress them.

The existing 4030fa54 CI run has now completed all four wasmi execution primary
jobs. Neither JIT execution suite identifies a confirmed regression; ARM64
tiny_keccak is classified NEGLIGIBLE. The interpreter suites still carry their
three compiler warnings. Their numeric tables contain NOISY-FLOOR results
(x64 bulk-ops and ARM64 fibonacci-tail), which must not be described as proof
that those workloads are unchanged. Startup and independent-runner confirmation
are still pending. No V8 or Cranelift comparison was added.

### Complete primary CI evidence (2026-09-07)

All eight wasmi primary jobs at 4030fa54 have completed. Their 108 printed
benchmark rows and job links are retained in
[the complete primary tables](release-evidence/linux-release-wasmi-primary.json).
The previously pending x64 JIT startup job flags bz2 and spidermonkey as
REGRESSION; its CoreMark/ERC20 rows are NOISY-FLOOR, not proof of unchanged
performance. ARM64 JIT flags five startup workloads. Both interpreter startup
suites flag all seven. Four independent startup confirmation jobs are now live
in run 34141541984; all statements about these primary verdicts remain provisional.
The interpreter jobs' three compiler warnings remain real audit failures.

Correctness CI has also finished all jobs. The final RV32 and ARMv7 logs add
no new warning categories or test failures; they fail the already reported
ownership/capability clusters. Main and the remote PR head were rechecked over
normal Git SSH and remain 0983d9e4 and 4030fa54 respectively. The local validator
signature and warning-display commits have not interrupted the active run.

### Interpreter startup confirmation and Fibonacci sample audit (2026-09-07)

Independent x64 interpreter startup job 101811485362 fails the cross-run gate
for all seven workloads. CoreMark is 136.479 to 234.215 us on this runner;
ERC20 is 128.244 to 207.041 us. This reproduces the primary startup regression,
not merely a compiler warning or a single-run observation. ARM64 interpreter
job 101811485428 also fails all seven rows: CoreMark is 105.804 to 203.524 us
and ERC20 is 103.614 to 185.326 us. Both tables and exact job links are in
[the confirmation evidence](release-evidence/linux-release-startup-confirmation.json).
The two JIT startup confirmation jobs are still running at this point.

The ARM64 interpreter fibonacci-tail NOISY-FLOOR row was inspected from the
actual artifact, whose archive digest matches GitHub's SHA-256 metadata.
The pilot's baseline/candidate means are 6.531881/6.547102 ms; the reversed
process order yields 6.099505/6.503400 ms. The baseline gets 6.62% faster between
pairs while the candidate changes by only -0.67%. The current printed -6.21%
performance ratio comes from that second pair, not a stable loss across both.
Main already certifies a 5.44% identical-binary floor for this row; the existing
1.5 multiplier makes its effective elapsed gate 8.16%. Neither the calibration
file nor classification policy changed in this branch. The floor explains why
the row does not enter an independent-runner confirmation, but does not prove
absence of a source regression. The raw metric, process orders and provenance
are retained in [the sample audit](release-evidence/arm64-interp-fibonacci-tail-ci.json).
The artifact also identifies the actual measured candidate as PR merge commit
8427c328, corresponding to PR head 4030fa54 on unchanged base 0983d9e4.

Independent x64 JIT startup job 101811485364 subsequently confirms both
selected regressions: bz2 is 53.014 to 54.932 ms (-3.49% in the performance
ratio), and spidermonkey is 2.761 to 2.885 s (-4.30%). Its cross-run verdict
fails. Only ARM64 JIT startup confirmation remains live; the performance
workflow cannot pass with the three already confirmed failing suites.

### Completed confirmations and private-runtime ownership follow-up

ARM64 JIT confirmation 101811485461 has now completed with a failing cross-run
verdict. Four cases reproduce: pulldown-cmark 118.975 to 122.108 ms, FFmpeg
9.390 to 9.609 s, argon2 16.677 to 17.179 ms, and ERC20 3.785 to 3.900 ms.
Spidermonkey is classified NEGLIGIBLE on that confirmation runner. All 46 jobs
in the performance run are terminal. The complete printed confirmation rows
are preserved in `release-evidence/linux-release-startup-confirmation.json`.
No threshold or measurement floor was changed.

The [benchmark validation audit](release-evidence/benchmark-validation-audit.md)
confirms that V8 and Wasmtime/Cranelift validate their inputs, and the wasmi
execution reference is `eager.checked`. The main startup ranking excludes
wasmi's lazy modes. Nano main's optional validator was not enabled by the
benchmark features; the release candidate enables the existing implementation
in safe loading. Consequently the startup regression includes additional work,
not a competitor configuration that omits verification. No second verifier
was written.

The private raw Value inspection/encoding helpers have moved to their sole
production owner, JIT global instantiation. The scoped instance-table fixtures
now borrow their heap memory backing rather than use two interpreter-only
unsafe slice accessors. No public signature or call ABI changes in this step.

The interpreter now reads `Limits::effective_max`, preserving the existing
explicit/default metadata instead of duplicating growth ceilings. Its u64
Wasm-cap and byte-size checks remain. An external boundary test then exposed
a separate existing JIT bug: a host-imported table32 without an explicit max
accepted growth beyond the table32 index space (returning the old size 2 rather
than -1). JIT memory/table growth now additionally applies address-width and
representation caps; the new regression test covers normal/zero/failed growth,
local and host-imported resources, explicit/unspecified maxima, and both memory
index widths. The failure test passed after that correction with both guarded
and pure-heap JIT configurations. This behavior correction is intentional and
is not described as a pure ownership move.

The final native core run passes 686 test cases in 28 groups with no warnings;
four ignored tests remain. Release spec runs pass 260 JIT and 175 interpreter
files. The existing scoped-materialization Miri test runs once and passes with
strict provenance under both Stacked Borrows and Tree Borrows. Native pure-JIT
and pure-interpreter library checks and pure-interpreter test compilation have
no warnings. These local results do not override hosted failures at 4030fa54.

Thumb/RV32 still fail the warning audit for SIMD-only Immediate variants and
WasmOpcode::FD. An isolated capability-gating experiment removed that decoder
cluster but exposed unused SIMD primitives in the intentionally backend-neutral
semantic IR on RISC-V JIT. It was reverted rather than extending cfgs through
the IR or adding suppressions. The earlier blanket requirement for author
approval on every ownership cleanup was too broad: the policy explicitly
requires individual approval for new suppressions, while these retained fixes
settle ownership without adding any. The remaining SIMD representation boundary
is still unresolved and the release remains a draft.
