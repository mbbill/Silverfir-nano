# Claude Code Guidelines for Silverfir-nano

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

When fixing a warning or build error, do not blindly add `_` prefixes,
`#[allow(dead_code)]`, `#[allow(unused)]`, or — worst case — `unsafe` blocks
just to make the compiler quiet. These hide real problems.

If code is unused, remove it. If a parameter is unused, remove it from the
signature and fix the call sites. If you believe a suppression is genuinely
the right call, always ask the user for permission first and explain why.
