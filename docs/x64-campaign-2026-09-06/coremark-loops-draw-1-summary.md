## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| base | 19565.70 |
| corrected | 19940.11 |
| alu | 19852.06 |
| loop | 20024.75 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| corrected/base | +1.91% | 99.791388% | 0.208612% |
| alu/base | +1.46% | 98.253468% | 1.746532% |
| loop/base | +2.35% | 99.949902% | 0.050098% |
| alu/corrected | -0.44% | 22.185330% | 77.814670% |
| loop/alu | +0.87% | 97.436106% | 2.563894% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
