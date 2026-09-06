## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| main | 19535.71 |
| dse | 20188.43 |
| tables | 20212.57 |
| commute | 20217.01 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| dse/main | +3.34% | 99.999273% | 0.000727% |
| tables/main | +3.46% | 99.999413% | 0.000587% |
| commute/main | +3.49% | 99.996675% | 0.003325% |
| tables/dse | +0.12% | 93.468029% | 6.531971% |
| commute/tables | +0.02% | 61.723901% | 38.276099% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
