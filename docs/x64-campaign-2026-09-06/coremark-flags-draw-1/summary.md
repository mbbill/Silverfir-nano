## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19566.60 |
| bulk | 18212.45 |
| flags | 18421.58 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| bulk/base | -6.92% | 0.000009% | 99.999991% |
| flags/base | -5.85% | 0.000008% | 99.999992% |
| flags/bulk | +1.15% | 99.992872% | 0.007128% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
