---
commit: 8b7fab5
---
Native backend (CFG+SSA pipeline) benchmarks on Apple M4: CoreMark 33,770 — 230%
of Cranelift, 89% of V8 — versus the micro-JIT's 14,692. SHA-256 269 MB/s (108% of
Cranelift). Lua/fib38 478% of Cranelift. Most telling: mandelbrot 808 ms now BEATS
Cranelift (855 ms) and is 252% of V8 — the *same* FP loop kernel that was only 30%
of Cranelift under the micro-JIT. c-ray still lags (57% of Cranelift).

This is the validating measurement for the interpreter→native pivot: the new
pipeline closed the gap exactly on the loop-kernel / FP workloads the roadmap had
flagged as structurally unfixable under the micro-JIT, while the micro-JIT's no-SSA
structure could not. Measured on Apple Silicon; the mandelbrot reversal (30% →
beating Cranelift) is the load-bearing data point that a real SSA pipeline removes
the loop-boundary overhead the linear LIR could not.
