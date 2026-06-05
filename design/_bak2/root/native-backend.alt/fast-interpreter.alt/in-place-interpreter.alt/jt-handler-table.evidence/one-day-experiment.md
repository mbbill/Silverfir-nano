---
commit: d33f2413
---
Born (171da4b7) and deleted (d33f2413) on 2025-08-08, briefly the default
(SF_INTERP), both commits behind fabricated messages. A 2026-06 rollout of
the preserved commit established why no benchmark of it survives: JT was
never functional on real workloads — its call dispatch used `continue`
where `break` was needed (jt.rs:359), so every function call kept executing
the caller's bytecode after pushing the callee's locals, corrupting the
stack. Call-free code ran correctly; CoreMark (call-heavy) exited silently
with no output. The same-day deletion cut a broken experiment, not a
measured loser. Reference DT score in the same rollout: 793 ± 24 iter/s
(loaded machine).

With the one-line call-dispatch fix applied in the rollout, the A/B that
never happened in 2025 was run in 2026: DT (match dispatch) 838 ± 77
iter/s vs fixed-JT (handler table) 311 ± 9 — match dispatch wins 2.7x.
Plain function-pointer dispatch pays a full call/return per op from a
single dispatch site; the experiment's loss prefigures exactly what the
later tail-call trampoline design fixes (jump not call, per-handler BTB
entries, zero prologue).
