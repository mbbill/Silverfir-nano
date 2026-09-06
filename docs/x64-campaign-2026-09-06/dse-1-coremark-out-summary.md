## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| main | 19607.20 |
| float | 20178.26 |
| dse | 20175.87 |
| nocache | 20093.84 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| float/main | +2.91% | 99.997107% | 0.002893% |
| dse/main | +2.90% | 99.999494% | 0.000506% |
| nocache/main | +2.48% | 99.975686% | 0.024314% |
| dse/float | -0.01% | 45.030158% | 54.969842% |
| nocache/dse | -0.41% | 3.609195% | 96.390805% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
