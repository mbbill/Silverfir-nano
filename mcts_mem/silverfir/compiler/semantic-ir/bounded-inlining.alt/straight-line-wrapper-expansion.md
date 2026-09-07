- Small straight-line callees, including wrappers containing calls, expand before frame planning; structured callees remain calls.

## Facts

- 2026-09-06 (7fbc510b) measurement: two AMD 9V74 CoreMark draws gained only +0.1725% and +0.1135%, with weak evidence, while CoreMark startup throughput fell 7.12% on 9V74 and 8.21% on N2. Recursive Fibonacci code was unchanged. Runs 34079252500 and 34079252501 are recorded in docs/x64-campaign-2026-09-06/small-call-inline-native-verdict.json (sourced).

## Moves

- 2026-09-06 (3b88ca70) replaced by [[compiler/semantic-ir/bounded-inlining]]: Expanding straight-line nonleaf wrappers grows frames without exposing useful control flow; structured callees expose recursive base cases and early returns instead (code).
