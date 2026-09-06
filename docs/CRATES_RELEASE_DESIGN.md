# crates.io release design proposal — 2026-09-06

This is the design for the cleanup **after** the performance campaign. No
public runtime API or package identity has been changed by this audit.

## Recommended first release shape

Keep one runtime implementation crate, `sf-nano-core`, and the CLI
`sf-nano-cli`. Keep JIT/interpreter and ISA backends private modules selected
by their existing features and target meanings. A crate per engine or ISA
would expose currently private contracts, complicate feature coordination and
release ordering, and is not necessary to omit unused code.

Use these package roles initially:

| Package | First-release role |
|---|---|
| `sf-nano-core` | Public embedding API and runtime. Keep the existing package name for consumers and the upstream benchmark adapter. |
| `sf-nano-cli` | Optional installable command-line application. |
| `sf-nano-tracked-alloc` | Small versioned support crate while it is a non-optional dependency of core. Document as implementation support; avoid leaking its wrapper types through the primary API. |
| `sf-nano-memprof-report` | Support crate needed if the published CLI retains its optional `memprof` dependency. Optional dependencies still need a publishable registry identity. |
| spec/WASI harnesses, foldsim, bare/device demos | `publish = false`; remain workspace/development tools. |

A new umbrella crate is unnecessary for the first release. Extract
`sf-nano-wasi` only when the host API can support it cleanly. Today WASI uses
a thread-local context, so moving files alone would publish that design
problem under another package name. Do not make the release contingent on a
large compiler reorganization.

## Public API to stabilize

Build on the existing model: `Config`, `Engine`, opaque `Module`,
`RuntimeWorld`, `InstanceId`, `Instance`, `Func`, `Import`, `Caller`,
`Value`/`RefValue`, public value/function types and structured errors.

The two embedding paths should remain explicit:

- Simple standalone use: create an `Engine`, create an `Instance`, resolve a
  `Func` once, call with caller-provided parameter/result slices.
- Linked/reentrant use: create a `RuntimeWorld`, instantiate modules into it,
  use world-scoped instance/function handles, and reenter through a checked
  world access object from a host callback.

The existing `Instance::call(..., results: &mut [Value])` is a good efficient
base. Keep an allocating convenience call, but make its returned container
an `alloc::vec::Vec<Value>` or a deliberate public result wrapper whose type
does not change when `memprof` is enabled. Currently `invoke` signatures
expose `collections::Vec`, which aliases the tracked allocation facade.

Do not expose JIT leases, interpreter bodies, link registries, instruction
decoders, opcode tables, raw module builders or process-global reset hooks
as the ordinary embedding contract. Inventory actual users before narrowing
visibility. Engine diagnostics can return small owned snapshots through a
documented diagnostics module. A feature-gated public method is still an API
commitment; `doc(hidden)` alone does not make it private.

The root already reexports most user types, but `Module` and `ValueType`
remain reached through public implementation-oriented module trees. Reexport
the intended user types, then reduce those trees only after moving any
legitimate low-level consumer to a named API. Keep opaque world identities;
do not introduce a second ownership system named `Store` just to resemble
another runtime.

Document that a world has one engine tier and the current runtime uses
single-threaded `Rc` ownership. Do not promise `Send`/`Sync`, cross-tier
calls, compiled-module sharing, fuel, interruption, or async execution that
the implementation does not provide.

## Concrete release blockers and design questions

1. **Validation contract.** `Instance::new` is safe and takes arbitrary bytes.
   `vm/instance.rs::validate` calls the full module validator only under
   `sf_module_validator`; `validator` is absent from core's default features
   and the CLI dependency. The parser documents structural validation only,
   and interpreter predecode contains assumptions about validated Wasm.
   This audit does not establish an exploit, but a spec-suite pass with the
   validator enabled does not establish the default embedding contract.
   Stabilize a checked constructor whose guarantee does not disappear under
   an additive Cargo feature. If trusted input requires a separate fast path,
   define and audit its preconditions explicitly; do not silently disable
   semantic validation for a better startup ranking. Measure checked startup
   separately so its cost remains visible.
2. **WASI state and reentry.** `set_wasi_ctx` installs one context for the
   current thread. Multiple worlds/instances with different filesystem/args
   settings and nested calls need a specified context ownership rule. Prefer
   instance-owned host state passed through imports/Caller. This is the
   prerequisite for a useful independent WASI crate.
3. **Registry dependencies.** Core and CLI use path dependencies without
   registry versions. Add matching `version` plus `path`, and publish support
   crates before dependants. `[patch]` in the development workspace is not a
   substitute for a resolvable published dependency graph.
4. **Metadata and contents.** Core's description still says Wasm 2.0
   interpreter although the project has two engines and a Wasm 3.0 surface.
   Add repository/readme metadata and an experimentally verified MSRV.
   Ensure the `.crate` includes `build.rs`, `interp_gen/**` and all shared
   sources used by the generator. Keep benchmark blobs and stale temporary
   design notes out of the library archive. Packaging tests that reach
   `../benchmarks` need separate workspace and package treatment.
5. **License materials.** Manifests declare dual MIT/Apache licensing, but
   the tracked tree has no project-level license files; the only tracked
   LICENSE is the CoreMark benchmark's. Add the project license texts
   corresponding to the existing declaration, check retained third-party
   notices, and use the explicit SPDX expression `MIT OR Apache-2.0`.
6. **Version and name decisions.** Cargo packages all currently say 0.1.0,
   while Git release history uses other labels. Choose one first registry
   version after the API settles, check live name availability/ownership,
   and make the release tag match the published packages. Availability has
   not been assumed or reserved by this audit.

Items 1 and 2 are architecture/API decisions, not warning fixes. Apply the
repository's cfg ownership rules: move engine-owned code into its subtree;
do not add local feature gates or suppression reasons to hide a boundary
problem.

## Release sequence and verification

1. Finish the x64 baseline and selected optimization work with the existing
   correctness and paired performance gates.
2. Agree the validation and WASI ownership contracts; inventory exported API
   and migrate examples, CLI, harnesses and device consumers to the intended
   surface. Keep this a separate review from compiler optimization.
3. Complete metadata, license materials, versioned dependency edges and
   `publish = false` on tools. Define the supported feature matrix and MSRV
   based on builds, not the moving `stable` label.
4. Build an external consumer against extracted package contents: no workspace
   paths or implicit dev features. Verify default, JIT-only, interpreter-only,
   WASI and diagnostics builds, plus the supported bare-metal combinations.
   Compile public examples and doctests, and check documentation links.
5. Inspect `cargo package --list` and the generated archives; run package
   verification and `cargo publish --dry-run`. Support crates must already
   resolve from the registry before a dependant's unpatched dry-run is honest.
   Never use `--no-verify` to call packaging complete.
6. Present exact package names, versions, archive contents and successful
   checks for final release approval. Publish support packages, core, then
   CLI; tag the exact source and record the resulting registry URLs.

The expensive cross-engine ranking remains manual. Package checks can be a
manual/tag release lane; normal daily CI need not rebuild all competitors.

Cargo's official references: [publishing and package verification](https://doc.rust-lang.org/cargo/reference/publishing.html),
[multiple dependency locations](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#multiple-locations),
and [manifest fields](https://doc.rust-lang.org/cargo/reference/manifest.html).
Published versions cannot be overwritten, so package verification precedes
the upload rather than relying on a subsequent source fix.
