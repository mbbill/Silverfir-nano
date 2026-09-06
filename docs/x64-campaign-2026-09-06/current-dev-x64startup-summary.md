## wasmi-benchmarks: x64-linux / jit / startup

`f73219f56a70` -> `39ba77b31668`

- wasmi-benchmarks: `16a3d7c8fdb05506c116a9451175732d1ac77099`
- corpus: `7` `startup` Criterion benchmarks; dedicated CoreMark score excluded, `startup/coremark` retained
- schedule: one 10-sample adjacent A/B pilot; selected benchmarks receive up to `2` independent reverse/alternating confirmation pairs
- requested family confidence: regression `99.990%`, improvement `99.990%`; pilot `80.0%`
- family correction: `27` metrics x `4` platform/engine groups x `2` looks; effective P(reg) `99.999954%`, P(imp) `99.999954%`

| Benchmark | Baseline | Candidate | Delta (sample range) | Volatility | P(reg) | P(imp) | Samples | Runs | Pilot P | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| startup/bz2 | 52.248ms | 54.362ms | -3.89% (-4.56%..-3.38%) | 0.38% | >99.999999% | <0.000001% | 10 | 1 | >99.999999% | **REGRESSION** |
| startup/pulldown-cmark | 121.313ms | 125.100ms | -3.03% (-3.79%..-2.11%) | 0.53% | 99.999999% | 0.000001% | 10 | 1 | 99.999972% | **REGRESSION** |
| startup/spidermonkey | 2.713s | 2.787s | -2.63% (-3.03%..-2.29%) | 0.25% | >99.999999% | <0.000001% | 10 | 1 | >99.999999% | **REGRESSION** |
| startup/ffmpeg | 9.420s | 9.761s | -3.50% (-4.13%..-2.60%) | 0.48% | >99.999999% | <0.000001% | 10 | 1 | >99.999999% | **REGRESSION** |
| startup/coremark | 6.308ms | 6.526ms | -3.33% (-7.65%..-0.30%) | 1.83% | 99.999996% | 0.000004% | 20 | 2 | 99.999894% | NOISY-FLOOR |
| startup/argon2 | 19.290ms | 19.669ms | -1.92% (-3.08%..+1.60%) | 1.67% | 99.753661% | 0.246339% | 10 | 1 | 99.248261% | RECOVERED |
| startup/erc20 | 4.701ms | 4.632ms | +1.48% (-3.54%..+5.74%) | 3.98% | 13.196903% | 86.803097% | 10 | 1 | 99.758615% | RECOVERED |

> Only directions selected by the full-suite pilot can affect the gate. Later changes in other benchmarks are ignored.
