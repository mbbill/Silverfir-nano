---
commit: 7bd4154
---
The FMA hazard returned at a new layer: inside fused C handlers the C
compiler itself contracts float mul+add into FMA, changing rounding — the
same spec violation -rs hit at the pattern level. A global
-ffp-contract=off was tried and reverted (it taxes every float op);
the landed fix is surgical: an FP_MUL_BARRIER macro (empty inline asm with
a +r constraint) emitted after float multiplies only in fused patterns
that also contain float add/sub.
