# LEA plus generic byte-copy: first native wasmi comparison

Complete 20-item geometric throughput change: +13.8455%. AMD EPYC 9V74.

The independent confirmation is still pending for the two REGRESSION rows.

| Metric | Throughput change | Verdict |
|---|---:|---|
| execute/counter-local | -0.06% | PASS |
| execute/counter-param | -0.03% | PASS |
| execute/counter-global | -0.07% | RECOVERED |
| execute/fibonacci-rec | -0.07% | PASS |
| execute/fibonacci-iter | -4.94% | REGRESSION |
| execute/fibonacci-tail | +71.00% | IMPROVEMENT |
| execute/sort | +36.53% | IMPROVEMENT |
| execute/prime_sieve | +0.57% | PASS |
| execute/matrix_mul | +1.13% | IMPROVEMENT |
| execute/nbody | +0.98% | IMPROVEMENT |
| execute/argon2 | +232.59% | IMPROVEMENT |
| execute/tiny_keccak | -3.54% | PLACEMENT |
| execute/mandelbrot | -0.76% | NOISY-FLOOR |
| execute/spectralnorm | +1.45% | IMPROVEMENT |
| execute/compression | -1.12% | RECOVERED |
| execute/word_count | -0.29% | RECOVERED |
| execute/json_parse | +1.16% | PASS |
| execute/reverse_complement | +91.98% | IMPROVEMENT |
| execute/regex_redux | -4.99% | REGRESSION |
| execute/bulk-ops | +0.09% | PASS |
