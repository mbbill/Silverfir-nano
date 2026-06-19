commit: a05de669

Draft paper benchmark table (Apple M4, 5 runs mean): Silverfir-nano is the
fastest interpreter on CoreMark / SHA-256 / bzip2 / LZ4, beating wasm3 by
1.7-2.5x (geomean 2.0x), WAMR by 2.1-3.8x, wasmi by 2.5-5.4x, and reaching
27-62% of the optimizing Cranelift JIT (geomean 38%, CoreMark strongest at 62%).

Core interpreter is ~230 KB stripped (no fusion / WASI / std), ~2.9 MB with the
full 1,500-pattern fusion set plus WASI — a 12.6x growth dominated by 1,500
patterns x 4 depth variants = ~6,000 generated C functions.

Static handler classification: of 1,748 unique handlers (6,671 with depth
variants), 89.1% are always-linear (guard eliminated), 10.5%
potentially-branching (one guard kept), 0.5% always-nonlinear.
