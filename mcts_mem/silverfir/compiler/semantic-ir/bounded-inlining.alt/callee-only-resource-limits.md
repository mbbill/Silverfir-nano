- Structured numeric callees expand with per-callee limits and per-caller additive expansion limits of 256 extra operations and 64 extra locals. The expanded whole caller has no Algorithm4 region-by-local work limit.

## Facts

- 2026-09-06 (3b88ca70) measurement: diagnostic Fibonacci input35 improves 53.26–55.25% on sampled AMD hosts, but Argon2 startup throughput falls 6.01% on AMD 7763 and 8.04% on N2. Compression compile-heap peak grows 10.93%; a flat Argon2 module peak does not imply flat total compiler work. This input35 result must not replace standard input30 in suite standings (sourced).

## Moves

- 2026-09-06 (049874d0) replaced by [[compiler/semantic-ir/bounded-inlining]]: Whole-caller operation, local, frame and region-by-local limits contain compile work and frame growth that callee-size and additive expansion limits failed to control (code).
