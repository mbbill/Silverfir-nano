---
commit: a3fcdde
---
Cross-runtime WASI benchmark suite (coremark, lua, mandelbrot, c-ray, smallpt,
stream, binary_trees, brotli) on Apple M4, config 600 fused patterns / max-freq
merge / window=8, comparing against wasm3, wasmi, WAMR-fast, wasmtime-winch,
wasmtime-cranelift. CoreMark: Silverfir 8344 vs wasm3 4235, wasmi 2172, WAMR 3195
(~2× the fastest other interpreter), vs Winch 9071 and Cranelift 14964 (~56% of
Cranelift). Lua fib 7.00s vs wasm3 9.94s, Winch 6.14s, Cranelift 4.58s. Later
DESIGN.md reports beating Winch on CoreMark and Lua fib and reaching ~62% of
Cranelift on CoreMark, with wasm3 the next-fastest interpreter at ~46% of
Silverfir's CoreMark score. NOTABLE FAILURE: brotli FAILs on Silverfir with an
out-of-bounds memory access.

These are direct measurements on Apple M4. They confirm the "fastest pure
interpreter" claim *among interpreters* but also pin the ceiling: even fully tuned,
the interpreter sits at ~56–62% of an optimizing JIT on CoreMark and the gap is
much worse on FP/memory kernels — the head-to-head numbers that motivated looking
past the interpreter.
