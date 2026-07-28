# Agent Guidelines for Silverfir-nano

## Dev CI soft-fail is not a pass

Jobs in `.github/workflows/performance-regression.yml` use
`continue-on-error` on `dev/**` branches only to suppress GitHub failure email
during rapid development. A failed step or an `ACTION REQUIRED` / `SOFT-FAIL`
summary on a dev branch is still a real CI failure that must be investigated
and corrected.

Never describe such a run as passing, green, acceptable, or complete. Inspect
the individual job steps and summaries rather than relying on the workflow's
overall conclusion. Pull requests and `main` remain hard-fail gates.

## Warnings and lint suppressions

Correctness has one behavior: every compiler warning fails `ci/correctness.py`;
there is no strict/lenient switch. Performance jobs may continue compiling so
they can finish measuring, but must leave warnings visible and mark the
warning audit as action-required. CI runs `ci/lint_policy.py`
before either workflow; it rejects unreviewed `allow` / `expect` attributes
for `warnings`, `dead_code`, and `unused*`, lint-lowering compiler flags or
Cargo lint settings, and stale exception entries.

Fix the cause. Remove genuinely dead code and unnecessary fields or imports.
Do not add `#[allow(dead_code)]`, `#[allow(unused)]`, broad `cfg` gates,
underscore/`let _` band-aids, or edit `ci/lint_suppressions.toml` merely
to get a green run. A new exception requires an explicit human design decision
and an inline reason.

`reason = "..."` is not a fix. The policy script exists to expose every
suppression so it gets fixed properly — deleted, restructured, or gated
precisely — and annotating a finding with a reason plus a manifest entry
merely launders it through the audit, however accurate the reason reads.
Agents never author `reason=` attributes or manifest entries on their own
judgment; each exception is blessed by the user individually, site by
site. A finding with no visible proper fix is left red and reported as a
question, not made green.

## cfg follows meaning

Use each `sf_*` cfg only for what it means: `sf_backend_*` for per-ISA
surface, `sf_ir_dump`/`sf_jitdump`/`sf_call_trace` for JIT debug tooling,
`sf_has_*` for target capabilities — anywhere that meaning applies.
`sf_jit`/`sf_interp` mean "one engine's own code": the engines split at a
few declared gates (module declarations, engine selector, instance
dispatch, exports) and are on their own past it. Engine-only code found
in a shared file is misplaced — move it into the engine's subtree; never
gate it in place with `sf_jit`/`sf_interp`.

When warnings cluster only in a feature combination and suggest unclear
engine ownership or shared-runtime architecture, keep the CI failure visible
and report the cluster. Do not turn every diagnostic into a local suppression;
the architecture must be decided first.

A red warning audit is not blanket authorization to edit the affected code.
First reproduce and group every diagnostic by configuration and likely owner.
The default result of that audit is a report with the failures left intact.
Only fix an item immediately when it is clearly local, has no ownership or
feature-boundary implications, and does not require new `cfg` structure.
