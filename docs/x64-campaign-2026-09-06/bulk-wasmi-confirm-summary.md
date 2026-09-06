## wasmi-benchmarks: x64-linux / jit / execute

`f73219f56a70` -> `17f0914913b4`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `2` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.000%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `1` platform/engine groups x `2` looks; effective P(reg) `99.981481%`, P(imp) `99.999815%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/fibonacci-iter | 566.789us | 566.584us | +0.04% (-6.00%..+3.12%) | 2.52% | 48.211625% | 51.788375% | 10 | 1 | 51.788375% | PASS |
| execute/regex_redux | 23.811us | 23.841us | -0.13% (-0.86%..+0.25%) | 0.33% | 87.556114% | 12.443886% | 10 | 1 | 82.481462% | PASS |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
