## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19477.80 |
| bulk | 17877.63 |
| flags | 18088.10 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| bulk/base | -8.22% | 0.001881% | 99.998119% |
| flags/base | -7.13% | 0.002115% | 99.997885% |
| flags/bulk | +1.18% | 99.965711% | 0.034289% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
