commit: 4ea03b39

Two recovered investigation documents (WASM_ENGINE_INVESTIGATION.md,
SSA_OPTIMIZATIONS_ANALYSIS.md) justify building an SSA-reconstruction pipeline
rather than directly executing the already-LLVM-optimized wasm bytecode. LLVM
does run a full optimization pipeline (register allocation to wasm locals, phi
elimination, block structuring) before emitting wasm, so reconstruction looks
redundant; but:

(a) WebAssembly's mutable locals break SSA form, forcing re-analysis to recover
def-use;

(b) LLVM optimizes for an unknown runtime environment, whereas a runtime can do
runtime-specific optimizations LLVM cannot — speculative/profile-guided
inlining, bounds-check elimination from a known memory layout, redundant-load
elimination from runtime alias info, cross-module/cross-language inlining, and
physical register allocation (wasm has infinite locals, not registers).

All major engines (V8, SpiderMonkey, Wasmtime) likewise reconstruct SSA, so the
approach is industry-aligned. The investigation also recommends a fast non-SSA
interpreter path for cold code to avoid reconstruction overhead.
