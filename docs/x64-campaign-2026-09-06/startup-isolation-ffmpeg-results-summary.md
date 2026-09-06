## Nano-only startup: ffmpeg

4 rounds, alternating process order. Instantiation and destruction are timed.

| Variant | Instantiations/s |
|---|---:|
| main | 0.10496 |
| corrected | 0.10429 |
| loop | 0.10229 |
| current | 0.10239 |
| nocache | 0.10380 |
| nodse | 0.10273 |

| Comparison | Throughput change | P(improvement) | P(regression) |
|---|---:|---:|---:|
| corrected/main | -0.64% | 1.057427% | 98.942573% |
| loop/main | -2.54% | 0.033930% | 99.966070% |
| current/main | -2.45% | 0.000820% | 99.999180% |
| nocache/main | -1.11% | 0.058838% | 99.941162% |
| nodse/main | -2.12% | 0.011472% | 99.988528% |
| loop/corrected | -1.92% | 0.222324% | 99.777676% |
| current/loop | +0.09% | 71.607532% | 28.392468% |
| nocache/current | +1.38% | 99.957322% | 0.042678% |
| nodse/nocache | -1.03% | 0.108629% | 99.891371% |
| nodse/current | +0.34% | 98.944121% | 1.055879% |

Diagnostic estimates; full dev startup CI remains the gate.
