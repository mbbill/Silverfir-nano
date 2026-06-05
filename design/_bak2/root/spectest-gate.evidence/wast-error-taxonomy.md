---
commit: e4a20f95
---
The error enum in the very first commit — Malformed, Invalid, Unlinkable,
Exhaustion, Trap, Exit — mirrors the wast assertion taxonomy
(assert_malformed / assert_invalid / assert_unlinkable /
assert_exhaustion / assert_trap) before any interpreter or test runner
existed. The spec suite shaped the error model from the first file.
