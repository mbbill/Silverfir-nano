## wasmi-benchmarks: arm64-linux / jit / execute

`f73219f56a70` -> `f6362a74cfa6`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `20` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/counter-local | 295.100us | 295.103us | -0.00% (-0.26%..+0.13%) | 0.12% | 51.170610% | 48.829390% | 10 | 1 | 51.170610% | PASS |
| execute/counter-param | 295.048us | 295.070us | -0.01% (-0.17%..+0.13%) | 0.08% | 60.801050% | 39.198950% | 10 | 1 | 60.801050% | PASS |
| execute/counter-global | 147.578us | 147.563us | +0.01% (-0.04%..+0.06%) | 0.04% | 21.496621% | 78.503379% | 10 | 1 | 85.458786% | PASS |
| execute/fibonacci-rec | 3.028ms | 3.028ms | +0.00% (-0.15%..+0.14%) | 0.08% | 45.503410% | 54.496590% | 10 | 1 | 54.496590% | PASS |
| execute/fibonacci-iter | 884.626us | 884.593us | +0.00% (-0.12%..+0.12%) | 0.09% | 44.803176% | 55.196824% | 10 | 1 | 55.196824% | PASS |
| execute/fibonacci-tail | 471.776us | 471.713us | +0.01% (-0.19%..+0.25%) | 0.11% | 35.532361% | 64.467639% | 10 | 1 | 64.467639% | PASS |
| execute/sort | 18.073ms | 18.141ms | -0.38% (-0.71%..+0.22%) | 0.34% | 99.670888% | 0.329112% | 10 | 1 | 87.933289% | RECOVERED |
| execute/prime_sieve | 18.728ms | 18.707ms | +0.11% (-1.17%..+1.35%) | 0.80% | 33.611857% | 66.388143% | 10 | 1 | 66.388143% | PASS |
| execute/matrix_mul | 38.135ms | 38.145ms | -0.03% (-1.10%..+0.26%) | 0.40% | 58.174570% | 41.825430% | 10 | 1 | 58.174570% | PASS |
| execute/nbody | 16.549ms | 16.545ms | +0.02% (-0.08%..+0.18%) | 0.08% | 20.492105% | 79.507895% | 10 | 1 | 84.235228% | PASS |
| execute/argon2 | 75.142ms | 30.472ms | +146.59% (+133.18%..+156.00%) | 3.04% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/tiny_keccak | 12.053us | 12.063us | -0.08% (-0.27%..+0.40%) | 0.18% | 91.178444% | 8.821556% | 10 | 1 | 97.625009% | RECOVERED |
| execute/mandelbrot | 14.378ms | 14.376ms | +0.01% (-0.25%..+0.23%) | 0.13% | 42.301619% | 57.698381% | 10 | 1 | 57.698381% | PASS |
| execute/spectralnorm | 14.945ms | 14.945ms | +0.00% (-0.05%..+0.08%) | 0.04% | 41.233487% | 58.766513% | 10 | 1 | 58.766513% | PASS |
| execute/compression | 6.888ms | 6.889ms | -0.00% (-0.31%..+0.26%) | 0.17% | 53.605475% | 46.394525% | 10 | 1 | 97.171733% | RECOVERED |
| execute/word_count | 860.148us | 860.094us | +0.01% (-0.20%..+0.31%) | 0.19% | 45.899922% | 54.100078% | 10 | 1 | 92.431593% | RECOVERED |
| execute/json_parse | 5.357ms | 5.380ms | -0.43% (-1.16%..+1.03%) | 0.65% | 99.631899% | 0.368101% | 20 | 2 | 99.997511% | RECOVERED |
| execute/reverse_complement | 9.867us | 9.924us | -0.57% (-1.25%..-0.09%) | 0.30% | 99.999997% | 0.000003% | 20 | 2 | 99.990317% | NEGLIGIBLE |
| execute/regex_redux | 20.051us | 20.242us | -0.94% (-1.20%..+0.06%) | 0.37% | >99.999999% | <0.000001% | 20 | 2 | 91.875581% | NEGLIGIBLE |
| execute/bulk-ops | 588.504us | 589.309us | -0.14% (-0.56%..+0.00%) | 0.16% | 98.728443% | 1.271557% | 10 | 1 | 99.974619% | RECOVERED |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
