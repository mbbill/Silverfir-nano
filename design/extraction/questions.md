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

## Q3 (status: open)
context: Commits ba69c050 ("Added jump table, unfinished yet"), 920cf91b ("link
the jump table"), and 0305ca11 build a per-function jump table during the
validation walk: each branch gets a precomputed target offset, stack_offset, and
arity, plus a linked next-index, explicitly "for the interpreter." No
interpreter exists yet in the repo at this point, and no alternative
branch-resolution scheme (e.g. resolving branch targets lazily at interpretation
time, or threading the code into a separate CFG / block structure) appears in
history — the jump table is introduced as the sole mechanism.
question: Was precomputing branch targets into a jump table during validation a
deliberate up-front commitment to the eventual execution model (fused with
validation specifically to reuse the stack-height/arity bookkeeping the
type-checker already computes), and were lazy/at-runtime branch resolution or a
separate CFG representation considered and rejected — on what grounds (avoiding a
second pass over the code, interpreter dispatch speed, memory)?
blocks: silverfir/validator/jump-table.md
