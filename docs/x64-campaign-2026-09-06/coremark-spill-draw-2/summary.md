## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19461.04 |
| flags | 19800.36 |
| spill | 19801.99 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| flags/base | +1.74% | 99.936206% | 0.063794% |
| spill/base | +1.75% | 99.832756% | 0.167244% |
| spill/flags | +0.01% | 54.639876% | 45.360124% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
