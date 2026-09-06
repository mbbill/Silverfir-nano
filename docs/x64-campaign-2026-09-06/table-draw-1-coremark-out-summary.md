## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| main | 19606.89 |
| dse | 20203.45 |
| tables | 20544.78 |
| commute | 20506.88 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| dse/main | +3.04% | 99.991804% | 0.008196% |
| tables/main | +4.78% | 99.999969% | 0.000031% |
| commute/main | +4.59% | 99.998394% | 0.001606% |
| tables/dse | +1.69% | 99.935716% | 0.064284% |
| commute/tables | -0.18% | 7.656763% | 92.343237% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
