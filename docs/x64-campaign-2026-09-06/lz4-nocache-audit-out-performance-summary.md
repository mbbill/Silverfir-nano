## Performance: x64-windows / jit

`431b2062b926` -> `431b2062b926`

> Timing: the requested duration applies to every benchmark. CoreMark uses its explicit non-standard regression mode here; a bare CoreMark invocation retains the official EEMBC 10-second-minimum run.

> Probability gate: start with `4` paired samples. A direction with at least `80.0%` pilot probability enters an independent confirmation. Confirmation starts at `6` pairs and adaptively grows to at most `16` while the frozen direction can still converge. Requested family-wide confidence is `99.990%` for regressions and `99.990%` for improvements across `2` metrics, `3` performance jobs, and at most `6` confirmation looks. Bonferroni-adjusted per-look gates are `P(regression) >= 99.999722%` and `P(improvement) >= 99.999722%`.

| Metric | Baseline | Candidate | Delta (pair range) | Pair volatility | P(reg) | P(imp) | Pairs | Pilot P | Initial target | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lz4-compress | 579.69 | 580.37 | +0.12% (-1.20%..+1.82%) | 1.27% | 43.201525% | 56.798475% | 4 | 56.798475% | - | PASS |
| lz4-decompress | 1,238.0 | 1,247.6 | +0.77% (-4.73%..+23.48%) | 10.69% | 42.988529% | 57.011471% | 4+6 | 99.437912% | 12 | RECOVERED |

> Only directions selected from the initial full-suite sample may change the gate. Metrics observed incidentally during a targeted rerun are ignored.
