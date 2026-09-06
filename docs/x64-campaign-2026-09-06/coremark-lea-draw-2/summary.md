## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19595.13 |
| bulk | 18318.38 |
| reg | 19519.51 |
| offset | 18334.06 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| bulk/base | -6.52% | 0.003599% | 99.996401% |
| reg/base | -0.39% | 7.679848% | 92.320152% |
| offset/base | -6.44% | 0.003690% | 99.996310% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
