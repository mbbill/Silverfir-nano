commit: 005fae86

Apple-Silicon dispatch micro-benchmarks (benchmarks/btb): passing 8 virtual
registers as preserve_none call arguments costs ~0.6ns/op versus ~0.5ns for 3
registers (nearly free, since the values stay in CPU registers), while any
dynamic register selection is the real cost — indexing an 8-element register
array reached 1.7ns (pointer array) to 7-17ns (local/value array), and
register-only dynamic selection by nested ternary was 3.4ns due to branch
misprediction.

The lesson: scale to 8 hot registers by baking the register choice into
per-permutation handlers (fixed indices the compiler resolves) rather than
selecting registers dynamically at runtime.
