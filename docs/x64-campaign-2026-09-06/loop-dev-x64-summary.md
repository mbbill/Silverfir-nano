## wasmi-benchmarks: x64-linux / jit / execute

`f73219f56a70` -> `f6362a74cfa6`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `20` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/counter-local | 233.049us | 228.979us | +1.78% (-2.16%..+5.86%) | 2.32% | 1.915478% | 98.084522% | 10 | 1 | 98.707910% | PASS |
| execute/counter-param | 236.232us | 236.266us | -0.01% (-1.86%..+6.66%) | 2.37% | 50.753852% | 49.246148% | 10 | 1 | 50.753852% | PASS |
| execute/counter-global | 117.673us | 117.667us | +0.01% (-1.62%..+1.06%) | 0.80% | 49.166544% | 50.833456% | 10 | 1 | 50.833456% | PASS |
| execute/fibonacci-rec | 3.227ms | 3.216ms | +0.33% (-2.86%..+2.51%) | 1.41% | 23.589009% | 76.410991% | 10 | 1 | 76.410991% | PASS |
| execute/fibonacci-iter | 476.787us | 471.835us | +1.05% (-0.51%..+10.11%) | 3.12% | 15.558277% | 84.441723% | 10 | 1 | 90.149028% | PASS |
| execute/fibonacci-tail | 238.672us | 238.350us | +0.14% (-1.63%..+1.16%) | 0.88% | 31.785190% | 68.214810% | 10 | 1 | 68.214810% | PASS |
| execute/sort | 26.899ms | 21.347ms | +26.01% (+20.52%..+30.31%) | 2.59% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/prime_sieve | 16.050ms | 15.940ms | +0.69% (-1.59%..+4.24%) | 1.52% | 9.087440% | 90.912560% | 10 | 1 | 93.252641% | PASS |
| execute/matrix_mul | 27.953ms | 27.922ms | +0.11% (-1.27%..+1.01%) | 0.91% | 35.396222% | 64.603778% | 10 | 1 | 94.053782% | PASS |
| execute/nbody | 7.088ms | 7.110ms | -0.31% (-1.72%..+0.83%) | 0.80% | 87.776487% | 12.223513% | 10 | 1 | 81.313115% | PASS |
| execute/argon2 | 74.485ms | 25.310ms | +194.29% (+179.95%..+225.35%) | 4.23% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/tiny_keccak | 12.054us | 12.498us | -3.55% (-7.73%..-0.76%) | 1.61% | >99.999999% | <0.000001% | 20 | 2 | 99.999956% | PLACEMENT |
| execute/mandelbrot | 12.196ms | 12.287ms | -0.74% (-3.34%..+3.74%) | 2.07% | 86.014071% | 13.985929% | 10 | 1 | 85.128114% | RECOVERED |
| execute/spectralnorm | 10.906ms | 10.856ms | +0.46% (-2.31%..+4.77%) | 1.98% | 24.152042% | 75.847958% | 10 | 1 | 99.997497% | RECOVERED |
| execute/compression | 6.850ms | 6.851ms | -0.01% (-1.82%..+1.93%) | 1.08% | 50.790777% | 49.209223% | 10 | 1 | 50.790777% | PASS |
| execute/word_count | 702.183us | 716.735us | -2.03% (-9.21%..-0.13%) | 1.97% | 99.992268% | 0.007732% | 20 | 2 | 99.857645% | RECOVERED |
| execute/json_parse | 4.104ms | 4.081ms | +0.57% (-2.18%..+4.45%) | 2.47% | 23.957339% | 76.042661% | 10 | 1 | 76.042661% | PASS |
| execute/reverse_complement | 17.598us | 9.550us | +84.28% (+81.42%..+88.67%) | 1.07% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/regex_redux | 18.487us | 18.417us | +0.38% (-1.63%..+3.17%) | 1.40% | 20.564352% | 79.435648% | 10 | 1 | 79.435648% | PASS |
| execute/bulk-ops | 510.056us | 499.669us | +2.08% (-0.96%..+10.69%) | 3.42% | 4.238076% | 95.761924% | 10 | 1 | 88.442563% | RECOVERED |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
