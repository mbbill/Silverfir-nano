## wasmi-benchmarks: x64-linux / jit / execute

`f73219f56a70` -> `35e315c6372e`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `20` `execute` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| execute/counter-local | 332.530us | 330.964us | +0.47% (-0.57%..+1.90%) | 0.73% | 3.564052% | 96.435948% | 10 | 1 | 81.814508% | RECOVERED |
| execute/counter-param | 332.588us | 333.051us | -0.14% (-1.35%..+1.75%) | 0.85% | 69.123153% | 30.876847% | 10 | 1 | 81.840967% | RECOVERED |
| execute/counter-global | 168.587us | 165.790us | +1.69% (+0.76%..+2.72%) | 0.59% | 0.000449% | 99.999551% | 10 | 1 | 84.075228% | RECOVERED |
| execute/fibonacci-rec | 4.484ms | 4.499ms | -0.33% (-2.06%..+1.57%) | 1.21% | 79.806687% | 20.193313% | 10 | 1 | 79.806687% | PASS |
| execute/fibonacci-iter | 660.779us | 660.277us | +0.08% (-1.30%..+1.70%) | 0.92% | 39.953203% | 60.046797% | 10 | 1 | 90.841180% | RECOVERED |
| execute/fibonacci-tail | 665.430us | 383.454us | +73.54% (+67.57%..+88.21%) | 3.53% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/sort | 44.656ms | 31.851ms | +40.20% (+38.08%..+43.66%) | 1.16% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/prime_sieve | 26.531ms | 26.332ms | +0.75% (-0.56%..+1.59%) | 0.79% | 0.719452% | 99.280548% | 10 | 1 | 85.548450% | PASS |
| execute/matrix_mul | 39.215ms | 38.939ms | +0.71% (+0.26%..+2.31%) | 0.60% | 0.240870% | 99.759130% | 10 | 1 | 93.757193% | PASS |
| execute/nbody | 10.225ms | 10.228ms | -0.02% (-1.02%..+0.56%) | 0.57% | 55.292304% | 44.707696% | 10 | 1 | 80.020988% | PASS |
| execute/argon2 | 116.584ms | 34.616ms | +236.79% (+229.40%..+242.57%) | 1.01% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/tiny_keccak | 16.133us | 17.266us | -6.56% (-8.45%..-5.32%) | 1.04% | >99.999999% | <0.000001% | 10 | 1 | 99.999998% | PLACEMENT |
| execute/mandelbrot | 18.271ms | 18.256ms | +0.08% (-1.89%..+1.24%) | 0.88% | 38.982955% | 61.017045% | 10 | 1 | 84.888984% | RECOVERED |
| execute/spectralnorm | 60.403ms | 60.344ms | +0.10% (-0.60%..+0.62%) | 0.41% | 23.143354% | 76.856646% | 10 | 1 | 93.888303% | PASS |
| execute/compression | 11.376ms | 11.359ms | +0.15% (-0.99%..+1.15%) | 0.73% | 27.101684% | 72.898316% | 10 | 1 | 72.898316% | PASS |
| execute/word_count | 1.138ms | 1.123ms | +1.39% (+0.32%..+5.04%) | 1.37% | 0.531301% | 99.468699% | 10 | 1 | 95.084553% | PASS |
| execute/json_parse | 5.572ms | 5.538ms | +0.61% (-0.60%..+1.65%) | 0.83% | 2.196330% | 97.803670% | 10 | 1 | 99.910679% | PASS |
| execute/reverse_complement | 31.710us | 15.628us | +102.90% (+97.49%..+105.52%) | 1.28% | <0.000001% | >99.999999% | 10 | 1 | >99.999999% | **IMPROVEMENT** |
| execute/regex_redux | 28.007us | 27.997us | +0.04% (-0.45%..+0.89%) | 0.46% | 40.649111% | 59.350889% | 10 | 1 | 59.350889% | PASS |
| execute/bulk-ops | 510.202us | 511.518us | -0.26% (-2.23%..+1.76%) | 1.17% | 74.879364% | 25.120636% | 10 | 1 | 90.874980% | RECOVERED |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
