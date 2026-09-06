## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19580.97 |
| flags | 19728.13 |
| spill | 19978.73 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| flags/base | +0.75% | 99.504818% | 0.495182% |
| spill/base | +2.03% | 99.962569% | 0.037431% |
| spill/flags | +1.27% | 99.903731% | 0.096269% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
