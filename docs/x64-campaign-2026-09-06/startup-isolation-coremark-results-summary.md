## Nano-only startup: coremark

4 rounds, alternating process order. Instantiation and destruction are timed.

| Variant | Instantiations/s |
|---|---:|
| main | 157.50633 |
| corrected | 156.59350 |
| loop | 152.64615 |
| current | 152.89550 |
| nocache | 156.88481 |
| nodse | 152.12364 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| corrected/main | -0.58% | 23.638574% | 76.361426% |
| loop/main | -3.09% | 2.220646% | 97.779354% |
| current/main | -2.93% | 0.975917% | 99.024083% |
| nocache/main | -0.39% | 18.870356% | 81.129644% |
| nodse/main | -3.42% | 0.669241% | 99.330759% |
| loop/corrected | -2.52% | 6.353282% | 93.646718% |
| current/loop | +0.16% | 67.816493% | 32.183507% |
| nocache/current | +2.61% | 98.331902% | 1.668098% |
| nodse/nocache | -3.03% | 0.145884% | 99.854116% |
| nodse/current | -0.50% | 28.034214% | 71.965786% |

Diagnostic estimates; full dev startup CI remains the gate.
