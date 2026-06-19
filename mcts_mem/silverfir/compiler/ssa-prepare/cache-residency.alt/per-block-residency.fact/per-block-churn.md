commit: a50a44d4

Measured across 9 WASI benchmarks on ARM64, the per-block cached-local residency
planner left the cache set completely unstable: isect=0, meaning no local was
cached in every block of any large function. It produced 20-31% boundary-only
blocks that exist solely to run ensure/drop ops, and blew code size up 1.3-1.7x.
The entire extra-op delta over the old fixed-bank system was ensure+drop boundary
fixup; spill/fill — what the unified budget was meant to reduce — was unchanged,
making the new system strictly worse on every benchmark.
