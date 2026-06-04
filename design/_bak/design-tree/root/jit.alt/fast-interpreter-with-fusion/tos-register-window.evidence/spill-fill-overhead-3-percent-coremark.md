---
commit: 545199e, 4bb1de8
---
A TOS height/count tracing mechanism was added specifically to measure spill/fill
overhead before committing to the window depth. On CoreMark *with fusion* (189M
dispatches), the TOS window's combined spill/fill overhead was only ~3.10% — 2.06%
spills, 1.04% fills. Modest enough to justify committing to a 4-deep TOS window and
confirming that LLVM keeps the hot values in a small register region. This is a
measurement-driven sizing decision: the observability tooling was built first so
the window depth was chosen against a number rather than a guess.
