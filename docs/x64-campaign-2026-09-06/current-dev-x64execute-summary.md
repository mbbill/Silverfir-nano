## wasmi-benchmarks: x64-linux / jit / execute

`f73219f56a70` -> `39ba77b31668`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `20` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/counter-local | 323.206us | 321.386us | +0.57% (-5.61%..+3.70%) | 3.09% | 28.566721% | 71.433279% | 10 | 1 | 71.433279% | PASS |
| execute/counter-param | 321.164us | 319.222us | +0.61% (-4.31%..+3.48%) | 2.24% | 20.448101% | 79.551899% | 10 | 1 | 79.551899% | PASS |
| execute/counter-global | 159.775us | 161.107us | -0.83% (-6.21%..+3.77%) | 3.41% | 77.304649% | 22.695351% | 10 | 1 | 77.304649% | PASS |
| execute/fibonacci-rec | 4.320ms | 4.318ms | +0.03% (-2.12%..+1.60%) | 1.19% | 46.508352% | 53.491648% | 10 | 1 | 99.961953% | RECOVERED |
| execute/fibonacci-iter | 638.148us | 638.156us | -0.00% (-1.61%..+2.07%) | 1.26% | 50.130903% | 49.869097% | 10 | 1 | 50.130903% | PASS |
| execute/fibonacci-tail | 632.237us | 369.532us | +71.09% (+67.38%..+74.82%) | 1.15% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/sort | 43.059ms | 30.272ms | +42.24% (+38.64%..+46.71%) | 1.63% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/prime_sieve | 25.552ms | 25.328ms | +0.88% (-1.59%..+2.31%) | 1.20% | 2.257496% | 97.742504% | 10 | 1 | 99.563720% | PASS |
| execute/matrix_mul | 38.207ms | 38.458ms | -0.65% (-2.87%..+2.00%) | 1.65% | 88.055042% | 11.944958% | 10 | 1 | 88.352770% | PASS |
| execute/nbody | 9.819ms | 9.514ms | +3.20% (-0.36%..+5.95%) | 1.75% | 0.000007% | 99.999993% | 20 | 2 | 99.998673% | **IMPROVEMENT** |
| execute/argon2 | 110.821ms | 33.229ms | +233.51% (+217.09%..+243.22%) | 2.28% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/tiny_keccak | 16.022us | 17.202us | -6.86% (-9.48%..-4.84%) | 1.61% | 99.999990% | 0.000010% | 10 | 1 | 99.999974% | PLACEMENT |
| execute/mandelbrot | 17.751ms | 17.681ms | +0.39% (-2.17%..+3.44%) | 1.68% | 23.814158% | 76.185842% | 10 | 1 | 76.185842% | PASS |
| execute/spectralnorm | 58.544ms | 18.182ms | +221.98% (+217.03%..+228.14%) | 1.12% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/compression | 10.878ms | 10.684ms | +1.82% (-1.39%..+5.29%) | 1.70% | 0.397014% | 99.602986% | 10 | 1 | 99.993219% | PASS |
| execute/word_count | 1.095ms | 1.066ms | +2.76% (-0.81%..+7.33%) | 2.19% | 0.162486% | 99.837514% | 10 | 1 | 99.885029% | PASS |
| execute/json_parse | 5.315ms | 5.213ms | +1.96% (-1.48%..+6.67%) | 1.98% | 0.014915% | 99.985085% | 20 | 2 | 82.720220% | PASS |
| execute/reverse_complement | 30.016us | 14.890us | +101.58% (+95.27%..+107.45%) | 2.19% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/regex_redux | 26.791us | 26.764us | +0.10% (-4.80%..+4.38%) | 2.70% | 45.486500% | 54.513500% | 10 | 1 | 91.225697% | RECOVERED |
| execute/bulk-ops | 442.049us | 450.411us | -1.86% (-6.42%..-0.03%) | 1.91% | 99.403811% | 0.596189% | 10 | 1 | 94.702958% | PASS |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
