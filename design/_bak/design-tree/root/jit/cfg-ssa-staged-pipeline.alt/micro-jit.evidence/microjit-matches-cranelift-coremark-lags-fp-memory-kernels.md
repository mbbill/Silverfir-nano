---
commit: afd1653, 37c40ff
---
Micro-JIT benchmark results (microjit branch, Apple M4): CoreMark 14,692 — 100.2%
of Cranelift, 170% of Winch; beats Winch on SHA-256, bzip2, LZ4-decompress, and
Lua-fib (223%). But the loop-kernel / memory-kernel workloads cluster well below:
mandelbrot 2,849 ms = 76% of Winch / 30% of Cranelift; c-ray = 67% of Winch / 18%
of Cranelift; STREAM = 53–75% of Winch.

Crucially, profiling located *why*: the lagging workloads were ALREADY executing
inside JITed code that still retained interpreter-shaped overhead — repeated
loop-boundary dispatch, repeated memory-metadata / bounds-check setup, and hybrid
JIT↔handler transitions. So the ~30% gap is **structural**, not peephole-sized: it
comes from the micro-JIT's no-SSA / no-regalloc shortcut being unable to carry
register residency across loops or eliminate loop-boundary overhead. This is the
diagnostic fact that undermined the micro-JIT's linear-LIR structure and drove the
fork to a real CFG+SSA pipeline.
