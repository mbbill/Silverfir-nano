- Small numeric local callees can be expanded into a caller before frame planning. Expansion visits only original call sites and never recursively expands copied calls.

- Eligibility requires 64-bit GP words and an unlimited compiler-memory budget. Every finite-budget configuration and every 32-bit GP backend keeps the original call boundaries.

- The expanded caller is limited to 128 conservative operations, 8 charged locals, 32 frame slots including call scratch, and 32 regions times max(locals, 1). Original callers above 256 bytecode bytes are excluded; callees are limited to 128 bytes, 32 operations and 8 locals.

- Every expansion receives initialized private locals. Structured returns exit through a result-typed wrapper with a callee-relative stack floor. Straight-line nonleaf wrappers, unsupported control transfers and nonnumeric local/results are excluded.

## Facts

- 2026-09-06 (049874d0) measurement: standard-input recursive Fibonacci throughput improves 50.4992% on AMD EPYC 7763 and 23.0366% on Intel Xeon 8573C; all20 geometric changes are +2.1245% and +1.0104%. N2 all20 is +0.0697%. Exact inputs, negative rows and startup costs are in [[bounded-inlining.fact/native-evidence]] (sourced).

- 2026-09-06 (049874d0) pitfall: unlimited compiler budget alone does not identify a large-memory host; ESP32-C6 uses that sentinel while its code arena and stack are small. GP-word width must also participate in eligibility (code).

- 2026-09-06 (fc26a6a9) statement: the user explicitly answered “是，保留。” to retaining bounded inlining with an ERC20 startup increase of approximately 0.29–0.41 ms. Execution performance takes priority for this measured tradeoff; this does not authorize unbounded expansion or a general regression tolerance (sourced).

- 2026-09-06 (b5be4dad) experiment: disjoint expansions of the same callee can share physical local banks while retaining cumulative expansion charges. Despite smaller frames, the local Fibonacci compile-heap peak rose from 12648 to 13418 requested bytes. This candidate has no native timing evidence and is shelved; do not adopt it from frame-size or correctness results alone (sourced).

## Moves

- 2026-09-06 (049874d0) replaced [[compiler/semantic-ir/bounded-inlining.alt/no-semantic-inlining]]: Resource-bounded structured expansion recovers substantial recursive execution throughput while retaining original call boundaries for finite-memory and 32-bit GP targets; the user accepts the measured ERC20 startup cost (sourced).

- 2026-09-06 (3b88ca70) replaced [[compiler/semantic-ir/bounded-inlining.alt/straight-line-wrapper-expansion]]: Expanding straight-line nonleaf wrappers grows frames without exposing useful control flow; structured callees expose recursive base cases and early returns instead (code).

- 2026-09-06 (049874d0) replaced [[compiler/semantic-ir/bounded-inlining.alt/callee-only-resource-limits]]: Whole-caller operation, local, frame and region-by-local limits contain compile work and frame growth that callee-size and additive expansion limits failed to control (code).
