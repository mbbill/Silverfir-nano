## wasmi-benchmarks: arm64-linux / jit / execute

`f73219f56a70` -> `39ba77b31668`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `20` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/counter-local | 294.996us | 295.051us | -0.02% (-0.17%..+0.09%) | 0.07% | 78.119194% | 21.880806% | 10 | 1 | 81.804926% | RECOVERED |
| execute/counter-param | 295.019us | 295.010us | +0.00% (-0.11%..+0.19%) | 0.08% | 45.191521% | 54.808479% | 10 | 1 | 54.808479% | PASS |
| execute/counter-global | 147.561us | 147.578us | -0.01% (-0.16%..+0.10%) | 0.07% | 69.201169% | 30.798831% | 10 | 1 | 69.201169% | PASS |
| execute/fibonacci-rec | 3.027ms | 3.027ms | -0.00% (-0.08%..+0.11%) | 0.05% | 53.549321% | 46.450679% | 10 | 1 | 53.549321% | PASS |
| execute/fibonacci-iter | 884.405us | 884.536us | -0.01% (-0.16%..+0.10%) | 0.08% | 70.913627% | 29.086373% | 10 | 1 | 70.913627% | PASS |
| execute/fibonacci-tail | 471.685us | 471.844us | -0.03% (-0.32%..+0.02%) | 0.10% | 83.538917% | 16.461083% | 10 | 1 | 82.096083% | RECOVERED |
| execute/sort | 18.036ms | 17.995ms | +0.23% (-0.45%..+0.77%) | 0.34% | 0.336347% | 99.663653% | 20 | 2 | 99.178247% | PASS |
| execute/prime_sieve | 18.642ms | 18.723ms | -0.43% (-0.92%..-0.00%) | 0.29% | 99.943014% | 0.056986% | 10 | 1 | 99.962688% | PASS |
| execute/matrix_mul | 38.040ms | 38.039ms | +0.00% (-0.20%..+0.19%) | 0.13% | 48.666940% | 51.333060% | 10 | 1 | 51.333060% | PASS |
| execute/nbody | 16.544ms | 16.545ms | -0.01% (-0.11%..+0.09%) | 0.06% | 70.154146% | 29.845854% | 10 | 1 | 70.154146% | PASS |
| execute/argon2 | 75.165ms | 29.615ms | +153.81% (+143.73%..+157.20%) | 1.58% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/tiny_keccak | 12.050us | 12.066us | -0.14% (-0.29%..+0.05%) | 0.12% | 99.752042% | 0.247958% | 10 | 1 | 99.372237% | RECOVERED |
| execute/mandelbrot | 14.376ms | 14.371ms | +0.04% (-0.03%..+0.14%) | 0.05% | 2.617450% | 97.382550% | 10 | 1 | 84.232289% | PASS |
| execute/spectralnorm | 14.945ms | 14.948ms | -0.02% (-0.45%..+0.11%) | 0.16% | 64.624691% | 35.375309% | 10 | 1 | 64.624691% | PASS |
| execute/compression | 6.886ms | 6.883ms | +0.04% (-0.09%..+0.41%) | 0.15% | 21.464602% | 78.535398% | 10 | 1 | 78.535398% | PASS |
| execute/word_count | 860.288us | 860.391us | -0.01% (-0.35%..+0.44%) | 0.20% | 57.350104% | 42.649896% | 10 | 1 | 57.350104% | PASS |
| execute/json_parse | 5.365ms | 5.380ms | -0.27% (-1.68%..+1.15%) | 0.85% | 91.776438% | 8.223562% | 20 | 2 | 96.294200% | RECOVERED |
| execute/reverse_complement | 9.860us | 9.926us | -0.66% (-1.17%..-0.25%) | 0.29% | >99.999999% | <0.000001% | 20 | 2 | 99.996353% | NEGLIGIBLE |
| execute/regex_redux | 20.140us | 20.116us | +0.12% (-1.12%..+1.19%) | 0.79% | 32.123441% | 67.876559% | 10 | 1 | 67.876559% | PASS |
| execute/bulk-ops | 589.292us | 589.149us | +0.02% (-0.13%..+0.18%) | 0.09% | 20.546528% | 79.453472% | 10 | 1 | 79.453472% | PASS |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
