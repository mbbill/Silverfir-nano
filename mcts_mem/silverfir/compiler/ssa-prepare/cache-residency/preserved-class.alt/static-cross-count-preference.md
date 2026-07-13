- A cached local earns a preserved-lane preference only after its
  whole-function count of crossed local-JIT calls reaches a backend-configured
  static threshold (default 7); each qualifying crossing counts one regardless
  of loop depth.

- The preference is derived over the final SSA as a pure function of the
  program and module facts (`derive_preferred_preserved`), and broadcast
  identically to every block.

- Fixed-local-only indirect calls count as crossable local-JIT calls in the
  tally, alongside direct calls to local bodies.

- The plan models every call as killing all cached residents; the preference
  steers machine lane placement and the within-block carry only.

## Facts

- 2026-07-13 statement: the threshold value 7 and the static-count form were
  never benchmarked, retuned, or weighed against a trip-weighted alternative —
  both origin commits (9cdd924a, c66b42fd) have empty bodies and the recorded
  rationale ("one isolated call does not pay for a callee-saved register")
  never considered that one static call site inside a hot loop is many dynamic
  crossings (sourced).

## Moves

- 2026-07-13 (7db708a6) replaced by [[preserved-class]]: the static unweighted
  cross-count threshold (7, never benchmarked or retuned) gated a machine-side
  carry that could only rescue same-block re-ensures — the plan's class-blind
  kill-at-call had already scheduled the cross-block reloads — so a hot loop
  crossing one direct call paid publish-drop-reload per iteration for every
  cached local; making the class a solver nomination priced in the residency
  objective and a plan-level survival contract removed those reloads
  corpus-wide, all nine modules improving (net −27,402 native instructions)
  (sourced)
