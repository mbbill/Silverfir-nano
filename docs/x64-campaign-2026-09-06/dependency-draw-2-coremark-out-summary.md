## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| loop | 20014.02 |
| flags | 20253.58 |
| noalu | 20152.25 |
| float | 20176.92 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| flags/loop | +1.20% | 99.523617% | 0.476383% |
| noalu/loop | +0.69% | 99.280552% | 0.719448% |
| float/loop | +0.81% | 99.933236% | 0.066764% |
| noalu/flags | -0.50% | 9.911050% | 90.088950% |
| float/noalu | +0.12% | 88.458982% | 11.541018% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
