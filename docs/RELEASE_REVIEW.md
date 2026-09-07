# First registry release: review status

This is a release candidate, not an approved release. PR [43](https://github.com/mbbill/Silverfir-nano/pull/43)
remains a draft. No crates.io publication or initial public API acceptance has
occurred. The proposed packages are `sf-nano-core` and `sf-nano-tracked-alloc`
at `0.1.0`, with Rust `1.94`; development tools remain non-publishable.

## Implemented contracts

The [embedding README](../sf-nano-core/README.md) describes the supported
Wasmtime-style engine/module/instance/callback model. Safe module construction
validates input; the unsafe constructor explicitly requires prevalidated bytes.
Imports and runtime references are opaque, function/reference ownership is
checked, and memory guards prevent conflicting execution and reentry. WASI
state belongs to captured imports. Memprof retains basic statistics without
changing core embedding types, signatures or traits.

Relevant behavioral coverage lives in `module_validation`, `public_api_contract`,
`reference_ownership`, `host_value_types`, `memory_borrowing`, `export_linking`
and `wasi_context` under [core integration tests](../sf-nano-core/tests).
These tests support the named contracts; they do not certify every engine or
target configuration. Interpreter host functions currently allow at most eight
results. JIT host exceptions propagate to the embedder even when Wasm has a
matching handler; that existing limitation is documented, not fixed here.

## Evidence and remaining gates

| Requirement | Verified evidence | Still required |
| --- | --- | --- |
| Minimal API and memprof transparency | Fresh `82ac86d1` capture passes five core memprof parity profiles, both support profiles and Cargo contracts. Digest `ae547697…224e2` is unchanged from the initial hosted capture. | Obtain explicit acceptance of the entire initial API for the final PR head. No snapshot update counts as approval. |
| Human review enforcement | Workflow and policy implementation fail closed when the review environment is absent; CI demonstrates that failure. | Configure the protected environment and required main-branch gate using authenticated repository administration, then verify them. |
| Hosted correctness | All 12 jobs in [run 34151036775](https://github.com/mbbill/Silverfir-nano/actions/runs/34151036775) pass at `68412439`, including the previously failing x64 and bare/cross targets. The validator follow-up `82ac86d1` passes 689 core cases and JIT 260/interpreter 175 spec files locally. | Verify the validator follow-up in final-head CI. |
| Single-engine and low-memory builds | The complete `68412439` hosted matrix passes without suppressions. Unpacked `82ac86d1` packages also pass JIT-only, interpreter-only and dual-engine adapter execution, Thumb interpreter and RV32 dual-engine no_std compilation. | Retain these configurations in final-head CI; validation evidence does not establish memory use for every possible module. |
| Startup and execution performance | All eight `68412439` Linux wasmi primary tables are recorded in [the evidence](release-evidence/linux-release-wasmi-primary.json). Four execution tables show no confirmed regression; all four startup confirmation jobs fail: seven interpreter cases on each architecture, two JIT cases on x64 and four on ARM64. Bounded validator scratch reuse reduces local parse-plus-validation medians by 0.65–5.38%. | Measure the final revision. Reduce startup overhead where practical, report unavoidable full-validation cost for review, and resolve any confirmed execution regression. No threshold is relaxed. The [validation audit](release-evidence/benchmark-validation-audit.md) separates checked, unchecked and deferred competitors. |
| Packages and downstream integration | Clean `82ac86d1` archives pass Cargo verification and independent unpacked-package checks through the actual migrated wasmi adapter. [Package hashes and checks](release-evidence/package-candidate.json) cover three engine feature configurations and two bare targets. | Produce the final approved/tagged archives with exact VCS provenance before publication; these remain candidate packages. |
| Publication | Package names, metadata, licenses and dependency versions are prepared. | Explicit approval of the final package set/version and actual publication; then verify registry-based integration. |

## Ownership follow-up and remaining design work

The earlier blanket statement that all three ownership cleanups required
separate human approval was too broad. [AGENTS.md](../AGENTS.md) requires
diagnostic grouping and a settled ownership design; it explicitly requires
individual human approval for new warning suppressions. None are added here.

Raw `Value` helpers now belong to JIT instantiation. The interpreter consumes
the shared effective resource limits instead of duplicating defaults. Host
imports without an explicit maximum retain their metadata; JIT growth now
also enforces the resource's address-width ceiling. A new external regression
test exposed that an unbounded host table32 previously accepted growth beyond
its index space. This is an intentional correctness fix, not merely a move.
The scoped Miri fixtures use backing borrows instead of test-only raw slices.

The non-SIMD capability boundary now spans decoded operations, semantic
primitives, and the already capability-gated machine IR. Unsupported targets
still reject SIMD in the shared decoder; impossible non-SIMD lowering stubs
are removed. The primitive semantics and ordering for SIMD-capable builds are
unchanged. Local core/spec and cross-compilation checks pass without warning
suppressions; hosted correctness at `68412439` also passes the full matrix.

The added resource-growth tests exposed old x64 memory64/table64 truncation
and interpreter table64 growth semantics. Separate fixes preserve full deltas,
check limits and overflow, and return the correctly typed -1 on failure. Tests
reproduce the old failures and pass after the fixes on local x64/ARM64. The adjacent bulk-memory/table audit found additional old truncation defects.
Those fixes preserve each source/destination index width and check overflow;
ARM64's memory32 fast path now requires explicit module-type evidence. Local
x64/ARM64 bulk-index tests and hosted correctness at `68412439` pass. Its
Linux wasmi execution primaries show no confirmed regression; the later
validator-only follow-up still needs hosted measurements.

The connector cannot administer repository settings. A fresh attempt to use
the management UI is blocked by the Mac lock screen; manual unlock has been
requested. The protected review environment is still unconfigured. The exact
settings and review procedure are in [the API policy](PUBLIC_API_POLICY.md).
Approval of those settings is separate from acceptance of an API and from
permission to publish crates.

Detailed revision-specific logs, measurements and caveats are indexed in
[the release record](PUBLIC_API_RELEASE.md). It is a chronological record;
earlier successful stages do not override later or unresolved failures.
