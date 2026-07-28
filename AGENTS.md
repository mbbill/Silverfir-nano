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

When warnings cluster only in a feature combination and suggest unclear
engine ownership or shared-runtime architecture, keep the CI failure visible
and report the cluster. Do not turn every diagnostic into a local suppression;
the architecture must be decided first.

A red warning audit is not blanket authorization to edit the affected code.
First reproduce and group every diagnostic by configuration and likely owner.
The default result of that audit is a report with the failures left intact.
Only fix an item immediately when it is clearly local, has no ownership or
feature-boundary implications, and does not require new `cfg` structure.
