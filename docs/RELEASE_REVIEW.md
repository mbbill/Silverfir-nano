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
| Single-engine and low-memory builds | CI reproduces and groups raw-value helpers, effective limits and non-SIMD decoding warnings. Cross-RISC-V64 also reports unused interpreter test-support accessors. | Resolve ownership decisions below and verify without suppressions. These warning failures are not passes. |
| Startup and execution performance | All eight Linux wasmi primary jobs complete at `4030fa54`; [complete printed tables](release-evidence/linux-release-wasmi-primary.json) retain noisy-floor results. Independent runners [confirm all seven interpreter startup regressions on both architectures](release-evidence/linux-release-startup-confirmation.json); JIT confirmation remains live. | Finish confirmation and resolve regressions under the unchanged gates. Validator optimizations reduce safe loading cost but do not establish a passing full-startup result. |
| Packages and downstream integration | Clean `7d1f0c8c` archives pass Cargo verification; an independent unpacked-package consumer builds the actual wasmi adapter and exercises both engines. CI adapter migration is fixed at `4030fa54`. | Regenerate and verify exact archives from the final clean reviewed revision. |
| Publication | Package names, metadata, licenses and dependency versions are prepared. | Explicit approval of the final package set/version and actual publication; then verify registry-based integration. |

## Pending ownership decisions

These proposals are already awaiting author input under [AGENTS.md](../AGENTS.md).
That policy requires a design decision before changing engine/capability
boundaries exposed by warning clusters. No approval is inferred from elapsed time.

- Move private raw `Value` type/raw-word conversion helpers to their existing
  JIT instantiation owner and adapt the internal fixture that calls one helper.
  Keep public value conversion and engine call ABIs unchanged.
- Keep declared memory/table limits and range validation shared; derive the
  existing effective default caps in the JIT memory/table grow implementation,
  which is their only production reader.
- Consolidate JIT SIMD decoding and capability-specific private immediate/opcode
  variants so non-SIMD targets retain a coherent decoding boundary. Do not add
  engine cfgs or warning suppressions to shared code to clear diagnostics.

Remote administration also remains unavailable through the current authorized
tools; the requested authentication choice has not been answered. The exact
settings and review procedure are in [the API policy](PUBLIC_API_POLICY.md).
Approval of those settings is separate from acceptance of an API and from
permission to publish crates.

Detailed revision-specific logs, measurements and caveats are indexed in
[the release record](PUBLIC_API_RELEASE.md). It is a chronological record;
earlier successful stages do not override later or unresolved failures.
