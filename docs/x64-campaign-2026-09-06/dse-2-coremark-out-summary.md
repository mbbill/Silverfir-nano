## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| main | 23779.84 |
| float | 25425.30 |
| dse | 25694.01 |
| nocache | 24368.74 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| float/main | +6.92% | 98.369411% | 1.630589% |
| dse/main | +8.05% | 99.171225% | 0.828775% |
| nocache/main | +2.48% | 94.609117% | 5.390883% |
| dse/float | +1.06% | 77.433465% | 22.566535% |
| nocache/dse | -5.16% | 2.140591% | 97.859409% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
