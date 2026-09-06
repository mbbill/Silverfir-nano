## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 22262.24 |
| bulk | 22036.16 |
| reg | 22393.81 |
| offset | 21576.54 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| bulk/base | -1.02% | 0.039927% | 99.960073% |
| reg/base | +0.59% | 91.501358% | 8.498642% |
| offset/base | -3.08% | 0.002129% | 99.997871% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
