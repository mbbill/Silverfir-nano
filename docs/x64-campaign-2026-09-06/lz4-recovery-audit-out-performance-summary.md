## Performance: x64-windows / jit

`9a8208289a88` -> `431b2062b926`

> Timing: the requested duration applies to every benchmark. CoreMark uses its explicit non-standard regression mode here; a bare CoreMark invocation retains the official EEMBC 10-second-minimum run.

> Probability gate: start with `4` paired samples. A direction with at least `80.0%` pilot probability enters an independent confirmation. Confirmation starts at `6` pairs and adaptively grows to at most `16` while the frozen direction can still converge. Requested family-wide confidence is `99.990%` for regressions and `99.990%` for improvements across `2` metrics, `3` performance jobs, and at most `6` confirmation looks. Bonferroni-adjusted per-look gates are `P(regression) >= 99.999722%` and `P(improvement) >= 99.999722%`.

| Metric | Baseline | Candidate | Delta (pair range) | Pair volatility | P(reg) | P(imp) | Pairs | Pilot P | Initial target | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lz4-compress | 508.73 | 582.96 | +14.59% (+13.20%..+15.52%) | 0.84% | 0.000009% | 99.999991% | 4+6 | 99.996895% | 6 | **IMPROVEMENT** |
| lz4-decompress | 1,317.6 | 1,329.3 | +0.89% (-1.78%..+3.58%) | 2.41% | 25.581331% | 74.418669% | 4 | 74.418669% | - | PASS |

> Only directions selected from the initial full-suite sample may change the gate. Metrics observed incidentally during a targeted rerun are ignored.
