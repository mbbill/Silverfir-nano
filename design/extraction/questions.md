# Extraction questions

## Q1 (status: open)
context: The very first commit (e4a20f95, "BOOM") lands a from-scratch
WebAssembly parser/runtime in Rust — its own binary parser, payload reader,
module model, and a `VM` placeholder — with no dependency on an existing Wasm
engine (wasmtime, wasmi, wasm3) or even an existing Wasm binary-format crate
(wasmparser). Git shows the chosen path but never the rejected one.
question: Why build a Wasm interpreter from scratch rather than wrap or fork an
existing engine/parser? Was an existing crate evaluated and rejected (and on
what grounds — performance ceiling, no_std/embedded target, license, learning,
control over the eventual JIT), or was from-scratch a non-negotiable premise of
the project?
blocks: silverfir.md (the root's from-scratch commitment)

## Q2 (status: open)
context: Commit 4336bd63 introduces the code decoder as a streaming visitor
(`OpcodeHandler` trait, `decode_function` callback walk) with no materialized
instruction list between decode and consumer. Its first consumers (validator,
disassembly printer) are interchangeable handlers over the same walk. The diff
shows the visitor was adopted from the start; it does not show a materialized-IR
alternative being built and rejected.
question: Was a materialized instruction representation (a decoded `Vec<Instr>`
or bytecode array) ever considered for the validate/execute path, or was the
re-decode-on-demand streaming-handler model a deliberate premise from the
outset — e.g. to avoid the per-function allocation and to let validation,
disassembly, and (later) execution share one decode loop?
blocks: silverfir/decoder.md
