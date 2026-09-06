## Nano-only wasmi CoreMark experiment

4 process pairs per comparison, alternating variant order; score ratios.

| Variant | Geomean score |
|---|---:|
| loop | 22608.09 |
| flags | 22636.60 |
| noalu | 22541.27 |
| float | 22603.97 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| flags/loop | +0.13% | 56.784792% | 43.215208% |
| noalu/loop | -0.30% | 40.459307% | 59.540693% |
| float/loop | -0.02% | 48.572254% | 51.427746% |
| noalu/flags | -0.42% | 21.632658% | 78.367342% |
| float/noalu | +0.28% | 57.452680% | 42.547320% |

These are diagnostic estimates. Inspect regressions and retain the full dev CI verdict.
