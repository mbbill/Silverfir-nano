# Extraction questions

## Q1 (status: open)
context: The first commit (e4a20f95) starts an entire WebAssembly engine from
scratch in Rust — its own binary parser, module model, and opcode tables —
rather than wrapping an existing Wasm runtime (wasmtime/wasmer) or even reusing
an off-the-shelf decoder such as `wasmparser`. Git shows the chosen path but
never the road not taken.
question: Was building the engine (and its parser) from scratch a deliberate
choice over wrapping/reusing an existing Wasm runtime or parser, and what drove
it — learning, control over the hot path, license, embeddability, something
else?
blocks: silverfir

## Q2 (status: open)
context: Commit 9e801234 drops the external `leb128` crate and vendors a
hand-unrolled decoder; the recorded rationale (avoid the `io::Cursor`
indirection on the parse path) is inferred from the code shape, not stated.
question: Was the LEB128 crate dropped for measured/expected decode speed, or
for another reason (no_std intent, dependency minimization, the
consumed-byte-count API mismatch)?
blocks: silverfir/parser/leb128
