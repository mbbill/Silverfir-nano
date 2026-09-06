## wasmi-benchmarks: x64-linux / jit / execute

`f73219f56a70` -> `53f6bc69dbaf`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `20` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/counter-local | 312.023us | 311.814us | +0.07% (-0.10%..+0.21%) | 0.12% | 5.163771% | 94.836229% | 10 | 1 | 85.485301% | RECOVERED |
| execute/counter-param | 312.008us | 312.299us | -0.09% (-0.87%..+0.41%) | 0.35% | 79.096786% | 20.903214% | 10 | 1 | 79.096786% | PASS |
| execute/counter-global | 156.293us | 155.910us | +0.25% (-0.03%..+1.61%) | 0.48% | 7.164939% | 92.835061% | 10 | 1 | 85.114618% | PASS |
| execute/fibonacci-rec | 4.320ms | 4.323ms | -0.08% (-0.58%..+0.42%) | 0.29% | 79.923587% | 20.076413% | 10 | 1 | 85.273043% | PASS |
| execute/fibonacci-iter | 632.707us | 624.017us | +1.39% (+0.63%..+2.70%) | 0.36% | <0.000001% | >99.999999% | 20 | 2 | 99.999668% | **IMPROVEMENT** |
| execute/fibonacci-tail | 623.462us | 364.528us | +71.03% (+70.21%..+71.79%) | 0.25% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/sort | 47.839ms | 47.838ms | +0.00% (-0.12%..+0.34%) | 0.14% | 48.999274% | 51.000726% | 10 | 1 | 51.000726% | PASS |
| execute/prime_sieve | 24.764ms | 24.226ms | +2.22% (+1.79%..+3.21%) | 0.40% | 0.000001% | 99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/matrix_mul | 58.999ms | 57.986ms | +1.75% (+1.24%..+3.24%) | 0.47% | <0.000001% | >99.999999% | 20 | 2 | 99.999994% | **IMPROVEMENT** |
| execute/nbody | 15.161ms | 15.219ms | -0.38% (-0.71%..+0.11%) | 0.29% | 99.867062% | 0.132938% | 10 | 1 | 99.999994% | RECOVERED |
| execute/argon2 | 156.080ms | 155.919ms | +0.10% (-3.81%..+1.32%) | 1.62% | 42.201535% | 57.798465% | 10 | 1 | 57.798465% | PASS |
| execute/tiny_keccak | 22.225us | 21.976us | +1.13% (+0.70%..+1.53%) | 0.22% | 0.000003% | 99.999997% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/mandelbrot | 17.379ms | 17.383ms | -0.02% (-0.18%..+0.09%) | 0.09% | 76.980116% | 23.019884% | 10 | 1 | 87.185120% | RECOVERED |
| execute/spectralnorm | 17.329ms | 17.316ms | +0.07% (-0.36%..+1.16%) | 0.41% | 29.669833% | 70.330167% | 10 | 1 | 70.330167% | PASS |
| execute/compression | 11.042ms | 10.936ms | +0.98% (+0.25%..+1.75%) | 0.34% | <0.000001% | >99.999999% | 20 | 2 | 99.613912% | **IMPROVEMENT** |
| execute/word_count | 1.300ms | 1.292ms | +0.64% (+0.04%..+1.18%) | 0.30% | 0.000001% | 99.999999% | 20 | 2 | 99.988069% | **IMPROVEMENT** |
| execute/json_parse | 8.280ms | 8.173ms | +1.31% (-0.71%..+3.15%) | 1.39% | 0.782337% | 99.217663% | 10 | 1 | 99.827590% | PASS |
| execute/reverse_complement | 34.614us | 34.258us | +1.04% (+0.04%..+4.97%) | 1.40% | 2.168430% | 97.831570% | 10 | 1 | 99.999933% | PASS |
| execute/regex_redux | 35.495us | 35.770us | -0.77% (-1.80%..+0.09%) | 0.53% | 99.999860% | 0.000140% | 20 | 2 | 99.765328% | RECOVERED |
| execute/bulk-ops | 775.891us | 777.149us | -0.16% (-3.44%..+1.96%) | 1.41% | 63.864139% | 36.135861% | 10 | 1 | 63.864139% | PASS |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
