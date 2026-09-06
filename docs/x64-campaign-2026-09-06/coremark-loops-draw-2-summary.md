## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19457.89 |
| corrected | 19814.41 |
| alu | 19629.51 |
| loop | 19790.64 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| corrected/base | +1.83% | 99.935597% | 0.064403% |
| alu/base | +0.88% | 99.901481% | 0.098519% |
| loop/base | +1.71% | 99.950826% | 0.049174% |
| alu/corrected | -0.93% | 1.012101% | 98.987899% |
| loop/alu | +0.82% | 99.508785% | 0.491215% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
