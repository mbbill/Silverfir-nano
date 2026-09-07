# Public API review

Every change to the supported embedding API requires a human decision, including
additions, trait implementations and Cargo feature/default changes. Generated
snapshots, passing compatibility checks, labels and agent-written explanations
do not grant approval.

The `public API` workflow extracts the PR head with pinned rustdoc and
cargo-public-api versions. It covers default, JIT, interpreter, both engines and
WASI on x64 Linux, including inferred auto traits and derived implementations.
Each profile is also extracted with memprof enabled; any difference or leaked
tracked-allocation type fails the job. Compiler/rustdoc warnings fail extraction.
The published support crate `sf-nano-tracked-alloc` is also captured with and
without memprof, including its Cargo features, edition and minimum Rust version.
Both support surfaces are part of the same review diff and evidence digest;
calling this crate internal does not exempt its published API from review.
Cross-target builds and the correctness suite remain separate checks; this API
listing does not prove behavioral compatibility or reference/memory safety.

For subsequent PRs the same extractor captures the actual base revision. A PR
cannot make its API change disappear by editing a checked-in snapshot. Baseline
captures are cached by immutable base commit and extraction script; the candidate
and its memprof parity are always rebuilt. No benchmark engines or performance
suite run in this workflow.
The workflow has no manual-dispatch trigger: a capture-only manual run must not
produce a successful status with the same name as the required PR review gate.
Use the local capture command for standalone evidence.

The initial PR introducing this policy has no previously reviewed API baseline.
Its diff shows the entire candidate surface and always requires human review.
Historical code is not retroactively called warning-free or API-stable. Once
this policy is on main, a missing or failed base capture is a failure, not an
empty diff. Toolchain/profile changes require review too; older revisions must
still be extractable under the proposed toolchain before merging such a change.

## Reviewing a change

1. Open the run's summary and download `public-api-<head SHA>-<run attempt>`.
   `review.json` binds the evidence to full base/head commits and content hashes;
   `public-api.diff` and the profile listings show the change.
2. Review the source and documentation for behavior, ownership, lifetimes and
   safety requirements that signatures cannot express. Review any changes to
   the workflow or API policy implementation as part of the same decision.
3. Approve the `public-api-review` environment for that run. It executes no
   application code and publishes nothing. This is explicit API acceptance for
   that PR head, not permission to publish a crate.

Unchanged APIs proceed automatically. Changed APIs and review-policy files wait
for environment approval. New PR pushes cancel prior pending runs; the final
`public API gate` also checks the current PR head. Require an up-to-date branch
before merging so an intervening base change is tested again. A rejected,
cancelled or failed capture/review cannot make the final gate pass.

## Repository configuration

Before enabling this as a required check, configure the repository as follows:

- Create the `public-api-review` environment with `mbbill` as its sole required
  reviewer. GitHub accepts any one listed reviewer, so adding a bot would permit
  approval without the designated human.
- Leave **Prevent self-review** off: the owner's account also opens many PRs.
  Deployment review supports this case even when an ordinary PR approval cannot.
- Turn off **Allow administrators to bypass configured protection rules**.
- Allow PR merge refs to use the environment. Do not put secrets in it.
- Require pull requests for main and disallow branch-rule bypass. Direct pushes
  must not avoid the pre-merge human review; post-push capture is only an audit.
- Require the `public API gate` status on main, with the branch up to date.
  Preserve all existing correctness/performance required checks.

The workflow verifies the environment configuration after producing the PR
evidence and before the approval job can run. An
absent/misspelled/unprotected environment fails closed; merely naming an
environment in YAML is insufficient. The environment setup and required status
are repository settings, not settings a source patch can install.

As with the other repository checks, maintainers must review modifications to
the workflow itself. Repository administrators can change workflows and branch
rules; this in-repository check is not an independent security boundary against
an administrator deliberately replacing its implementation.

GitHub documents [environment review and plan availability][environments] and
[reviewing deployments][reviews]. This repository is public, so required
environment reviewers are supported on current GitHub plans.

[environments]: https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments
[reviews]: https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/review-deployments

## Local evidence

Install `nightly-2026-09-06` with the `x86_64-unknown-linux-gnu` target and
cargo-public-api `0.52.0`, then run:

```sh
python3 -m ci.public_api capture --output target/public-api-candidate
```

This command may inspect an uncommitted working tree. It writes candidates only;
it never overwrites accepted snapshots. CI comparison additionally requires a
clean tracked checkout of the declared PR head. For a historical base, use the
same script with `baseline --root <base-checkout> --output <base-evidence>`.
Historical captures exercise the five ordinary profiles; candidate captures
always require all five memprof comparisons.
Both historical and candidate captures also include the two support-crate
profiles and its feature contract.

Accepted release snapshots will be retained with the reviewed release version.
Published-version compatibility checks supplement this review requirement;
compatibility alone never approves an API addition.
