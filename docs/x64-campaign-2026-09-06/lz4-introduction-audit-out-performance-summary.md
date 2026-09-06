## Performance: x64-windows / jit

`35e315c6372e` -> `9a8208289a88`

> Timing: the requested duration applies to every benchmark. CoreMark uses its explicit non-standard regression mode here; a bare CoreMark invocation retains the official EEMBC 10-second-minimum run.

> Probability gate: start with `4` paired samples. A direction with at least `80.0%` pilot probability enters an independent confirmation. Confirmation starts at `6` pairs and adaptively grows to at most `16` while the frozen direction can still converge. Requested family-wide confidence is `99.990%` for regressions and `99.990%` for improvements across `2` metrics, `3` performance jobs, and at most `6` confirmation looks. Bonferroni-adjusted per-look gates are `P(regression) >= 99.999722%` and `P(improvement) >= 99.999722%`.

| Metric | Baseline | Candidate | Delta (pair range) | Pair volatility | P(reg) | P(imp) | Pairs | Pilot P | Initial target | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lz4-compress | 846.26 | 694.22 | -17.97% (-18.51%..-17.41%) | 0.40% | >99.999999% | <0.000001% | 4+8 | 99.999567% | 6 | **REGRESSION** |
| lz4-decompress | 1,839.3 | 2,028.6 | +10.29% (+8.83%..+14.62%) | 1.71% | 0.000039% | 99.999961% | 4+8 | 99.970332% | 8 | **IMPROVEMENT** |

> Only directions selected from the initial full-suite sample may change the gate. Metrics observed incidentally during a targeted rerun are ignored.
