# Claude Code Guidelines for Silverfir-nano

## Dev CI soft-fail is not a pass

Jobs in `.github/workflows/performance-regression.yml` use
`continue-on-error` on `dev/**` branches only to suppress GitHub failure email
during rapid development. A failed step or an `ACTION REQUIRED` / `SOFT-FAIL`
summary on a dev branch is still a real CI failure that must be investigated
and corrected.

Never describe such a run as passing, green, acceptable, or complete. Inspect
the individual job steps and summaries rather than relying on the workflow's
overall conclusion. Pull requests and `main` remain hard-fail gates.

## When implementation hits obstacles, stop and discuss

When implementing an agreed design, if you run into a structural problem that
forces a deviation from the plan — a dependency you didn't account for, a
signature change that cascades too widely, or a workaround that compromises the
design — **stop and discuss** instead of silently working around it.

Do not:
- Silently diverge into workarounds (RefCell hacks, threading extra params, etc.)
- Make increasingly invasive changes trying to force the original plan to work
- Revert back and forth when approaches don't pan out
- Suppress problems with `#[allow(dead_code)]` or `let _ = ...` instead of
  removing dead code properly

Instead, state clearly: "I hit [specific problem]. The original plan assumed X
but actually Y. Here are the options I see." Then wait for direction.

The user has deep context about the design. A short discussion often reveals a
simpler solution that workarounds would never reach.

## Do not suppress warnings or errors with band-aids

Correctness has one warning behavior: `ci/correctness.py` always fails when a
compiler warning appears. There is no `--strict` mode and no warning-ignore
mode. Performance builds may compile through a warning so measurement can
still finish, but the warning must remain visible and its audit is
action-required. CI also runs `ci/lint_policy.py`, which rejects
unreviewed `allow` / `expect` attributes for `warnings`, `dead_code`, and
`unused*`, compiler flags or Cargo lint settings that lower those lints, and
stale exception entries.

When fixing a warning or build error, do not blindly add `_` prefixes,
`#[allow(dead_code)]`, `#[allow(unused)]`, or — worst case — `unsafe` blocks
just to make the compiler quiet. These hide real problems.

If code is unused, remove it. If a parameter is unused, remove it from the
signature and fix the call sites. If you believe a suppression is genuinely
the right call, always ask the user for permission first and explain why.
Do not add or modify `ci/lint_suppressions.toml` merely to make CI green.
That file records human-reviewed exceptional compilation boundaries; it is
not an agent-owned allowlist.

### `reason = "..."` is not a fix — never annotate your way past the audit

`ci/lint_policy.py` exists to EXPOSE every `allow`/`expect` hack so it gets
fixed **properly**: delete the dead code, restructure so the code is not
dead, or gate it precisely. Responding to an audit finding by writing an
inline `reason = "..."` and a matching manifest entry fixes nothing — it
launders the hack through the audit's escape hatch and neuters the tool.
This applies no matter how accurate or well-written the reason is.

Hard rules:

- An agent NEVER writes `reason = "..."` attributes or
  `ci/lint_suppressions.toml` entries on its own judgment — not even for
  sites it believes are legitimate boundaries, and not even when the user
  has approved a cleanup task in general terms. Each exception is blessed
  by the user individually, site by site.
- The manifest's only legitimate contents are irreducible compilation
  boundaries the user has personally reviewed (e.g. the `build.rs`
  dual-compile modules). Small and exact means a handful, not dozens.
- A finding with no proper fix you can see is left RED and reported as a
  question. A red audit is the ratchet working; making it green without
  fixing the cause is the failure mode this file exists to prevent.

If a feature-only build exposes a cluster of dead code that points to unclear
engine ownership, shared runtime state, or another architecture question,
leave the failure visible and report the cluster. Do not scatter `cfg`,
`expect`, or manifest entries across individual items before the ownership
decision is made.

Treat a failing warning audit as a request to reproduce and classify the
diagnostics, not as blanket authorization to edit every reported item. Leave
architecture-sensitive failures red until the user makes the ownership
decision. Immediate fixes are limited to clearly local problems that neither
change a feature boundary nor introduce new `cfg` structure.

## cfg follows meaning; engine cfgs mark the gate, not the code

Every `sf_*` cfg has one meaning, and correct use follows that meaning:
`sf_backend_*` is "this ISA's code surface", `sf_ir_dump` / `sf_jitdump` /
`sf_call_trace` are "JIT debug tooling", `sf_has_*` are target
capabilities. Those may appear wherever their meaning genuinely applies.

`sf_jit` and `sf_interp` mean "this is one engine's own code". The JIT and
the interpreter are two engines running the same code under the same
runtime: they split at a few declared gates — the `vm/mod.rs` module
declarations, the engine selector, instance dispatch, `lib.rs` exports —
and past the gate each engine is on its own inside its subtree. Therefore
these two cfgs must stay rare and gate-shaped. If an item in shared code
turns out to be engine-only, that is a misplaced item: MOVE it into the
engine's subtree. Do not gate it in place — `sf_jit`/`sf_interp` sprinkled
through shared files means the separation failed, however locally
convenient each individual cfg looked.

## Do not keep dead code behind `#[cfg(test)]`

If code is only needed by tests, it does not belong in the production source
gated behind `#[cfg(test)]`. Test helpers should live inside `#[cfg(test)]
mod tests { }` blocks. Production structs should not have `#[cfg(test)]`
fields or methods — with very rare exceptions (e.g., a field that enables
test-only invariant checking on a hot-path struct), which require explicit
discussion before adding.

Rule of thumb: if removing a function, struct, or field from production code
means some tests break, that is a signal to **rewrite or delete those tests**,
not to keep the dead code alive with `#[cfg(test)]`.

## Root-cause investigation: evidence first, theory second

When asked why a specific pattern appears in emitted output (IR, assembly,
logs, traces, etc.), do **not** open source files and build a narrative from
code reading. That produces plausible-sounding stories that are usually
wrong, and it wastes everyone's time defending them.

Work in this order:

1. **Find the existing dump / trace / observability mechanism.** Check
   `DEBUG.md`, `README.md`, `scripts/`, tool READMEs, and env vars before
   synthesizing anything. Mature codebases usually already have a way to
   capture the real IR / MIR / assembly for a given input.
2. **Feed the real input through the real pipeline.** Capture the output
   at every stage the dump supports.
3. **Read the captured output.** The answer is usually obvious from one
   or two stages once you see the actual data.
4. **Only then** look at the source that produced it, to understand the
   mechanism — not to guess at it.

If a dump mechanism doesn't exist, say so and discuss before building a
synthetic reproduction. Synthetic tests are for **pinning** understood
behavior, not for **discovering** it.

### Tests exist to falsify hypotheses, not to confirm them

When a diagnostic test is meant to prove a theory and the test comes back
showing the expected pattern is **absent**, that is the theory failing.
Stop and reconsider. Do not label the test "diagnostic-only" and keep
the theory. Do not add more setup to force the pattern to appear.

### Pass / module documentation describes inputs, not origins

A pass's doc comment describes the pattern the pass **sees** in its input.
It does **not** explain where that pattern **came from** in earlier stages.
Those are different questions. Do not conflate them.

### Do not present speculation as findings

If a claim about "why X happens" is not backed by captured evidence from
the real pipeline, do not present it as a finding. Either get the evidence
first, or label it explicitly as a hypothesis that needs verification.
Users can tell the difference, and presenting speculation as fact destroys
trust in everything else in the writeup.
