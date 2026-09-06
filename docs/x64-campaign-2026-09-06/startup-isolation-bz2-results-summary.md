## Nano-only startup: bz2

4 rounds, alternating process order. Instantiation and destruction are timed.

| Variant | Instantiations/s |
|---|---:|
| main | 18.79819 |
| corrected | 18.58699 |
| loop | 18.09490 |
| current | 18.02198 |
| nocache | 18.44338 |
| nodse | 18.25411 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| corrected/main | -1.12% | 1.402428% | 98.597572% |
| loop/main | -3.74% | 0.007835% | 99.992165% |
| current/main | -4.13% | 0.170648% | 99.829352% |
| nocache/main | -1.89% | 0.603526% | 99.396474% |
| nodse/main | -2.89% | 0.149198% | 99.850802% |
| loop/corrected | -2.65% | 0.130254% | 99.869746% |
| current/loop | -0.40% | 19.002113% | 80.997887% |
| nocache/current | +2.34% | 96.563550% | 3.436450% |
| nodse/nocache | -1.03% | 0.156804% | 99.843196% |
| nodse/current | +1.29% | 89.505971% | 10.494029% |

Diagnostic estimates; full dev startup CI remains the gate.
