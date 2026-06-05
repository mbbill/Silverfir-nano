commit: 57ae0b79

When the control-frame type was first built (commit 4336bd63's successor,
80306a9a), its constructor took `height` and `unreachable` parameters but its
body discarded them and hardcoded `height: 0, unreachable: false`. Both fields
are load-bearing: the entry height bounds underflow detection and the
unreachable flag drives stack-polymorphism. With them pinned to zero/false,
every frame believed it started at stack height 0 and was never unreachable, so
underflow detection and the post-unreachable `unknown` path were effectively
inert for any non-trivial body. The fix (57ae0b79) stored the passed values and,
in the same commit, tightened the underflow comparison from `==` to `>=`
(height equal to current length already means nothing left to pop in this
frame). Lesson: a constructor that takes a field's value and then ignores it
reads as correct at the call site while silently defeating the invariant the
field exists to enforce.
