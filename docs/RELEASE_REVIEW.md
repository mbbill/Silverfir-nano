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
| Minimal API and memprof transparency | Five core API profiles have exact memprof parity; both support-crate profiles and feature/MSRV contracts are captured. Real CI evidence for `7d1f0c8c` matches the local digest. | Capture the final head and obtain explicit acceptance of its entire initial API. No snapshot update counts as approval. |
| Human review enforcement | Workflow and policy implementation fail closed when the review environment is absent; CI demonstrates that failure. | Configure the protected environment and required main-branch gate using authenticated repository administration, then verify them. |
| Hosted correctness | `587f5752` passes 684 core test cases and JIT 260/interpreter 175 spec files locally; exclusions are unchanged. CI at `4030fa54` passes policy, Miri and Rust 1.94. | All supported configuration gates must pass on the final revision. |
| Single-engine and low-memory builds | The ownership follow-up moves raw conversions into JIT, shares effective growth limits with the interpreter, and removes unsafe test-only memory accessors. Native single-engine checks and local Thumb, RV32, RV64, and ARMv7 checks are warning-free after the decoder/IR SIMD capability cleanup. | Verify the final head on the hosted platform matrix. Earlier hosted warning failures remain failures until replaced by final-head evidence. |
| Startup and execution performance | All eight Linux wasmi primary jobs complete at `4030fa54`; [complete printed tables](release-evidence/linux-release-wasmi-primary.json) retain noisy-floor results. All four [startup confirmations](release-evidence/linux-release-startup-confirmation.json) are complete: seven interpreter regressions on each architecture, two JIT regressions on x64 and four on ARM64. | Resolve regressions under the unchanged gates. Validator optimizations reduce safe loading cost but do not establish a passing full-startup result. The [validation audit](release-evidence/benchmark-validation-audit.md) distinguishes checked, unchecked and deferred competitors. |
| Packages and downstream integration | Clean `7d1f0c8c` archives pass Cargo verification; an independent unpacked-package consumer builds the actual wasmi adapter and exercises both engines. CI adapter migration is fixed at `4030fa54`. | Regenerate and verify exact archives from the final clean reviewed revision. |
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
suppressions; hosted final-head validation is still pending.

The added resource-growth tests exposed old x64 memory64/table64 truncation
and interpreter table64 growth semantics. Separate fixes preserve full deltas,
check limits and overflow, and return the correctly typed -1 on failure. Tests
reproduce the old failures and pass after the fixes on local x64/ARM64. The adjacent bulk-memory/table audit found additional old truncation defects.
Those fixes preserve each source/destination index width and check overflow;
ARM64's memory32 fast path now requires explicit module-type evidence. Local
x64/ARM64 bulk-index tests pass; final-head execution performance must still
be measured.

Remote administration also remains unavailable through the current authorized
tools; the requested authentication choice has not been answered. The exact
settings and review procedure are in [the API policy](PUBLIC_API_POLICY.md).
Approval of those settings is separate from acceptance of an API and from
permission to publish crates.

Detailed revision-specific logs, measurements and caveats are indexed in
[the release record](PUBLIC_API_RELEASE.md). It is a chronological record;
earlier successful stages do not override later or unresolved failures.
