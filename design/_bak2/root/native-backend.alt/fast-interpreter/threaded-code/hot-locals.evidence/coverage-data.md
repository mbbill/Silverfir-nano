---
commit: 5b1d2e6
---
Profiled at the moment L0+L1 landed (CoreMark, 308M instructions, fusion
disabled): L0 ops cover 24% of all local accesses, L1 another 18% — 42%
combined, 58% still generic. Per-function counters show CoreMark's top 3
functions carry 96% of all local accesses, and within each, the top-2
locals carry 34–50% of that function's accesses. Weighted optimal coverage
for two cached locals across all functions: 41%. CoreMark with l0+l1+fusion:
7014.
