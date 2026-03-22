# Wasm Constant Folding Plan

## Goals

- Restore a clean baseline by keeping MachineIR constant folding conservative.
- Move semantic constant folding earlier so it can reduce transient pressure before spill/fill planning.
- Restrict early folding to numeric ops only: `i32`, `i64`, `f32`, `f64`.
- Preserve a simple backend contract: optimized semantic IR must never require more than one embedded constant operand.

## Immediate Baseline

- Keep the current conservative MachineIR peephole behavior:
  - only fold a transient constant into the immediately following instruction;
  - never perform long-range whole-block constant folding in MachineIR.
- This is a temporary safety baseline, not the final architecture.

## Target Architecture

Constant folding should happen in `wasm/` after decode and after inlining, before `prepare_function` enters the `middle/` pipeline.

Desired ownership by layer:

- `wasm/`
  - decode canonical semantic IR;
  - inline to fixed point;
  - run semantic numeric constant folding;
  - recompute semantic metadata such as stack effects and max stack height.
- `middle/`
  - plan frame layout from already-optimized semantic IR;
  - plan spill/fill;
  - lower into SSA;
  - run only post-lowering cleanup passes.
- `machine/`
  - lower to MachineIR;
  - keep only machine-local peepholes.

## One Combined Wasm Pass

Phase 1 and Phase 2 should be implemented as one pass, not as two separate pipeline stages.

The pass should enforce these output invariants:

- Unary numeric op with constant input is folded to a plain constant.
- Binary numeric op with two constant inputs is folded to a plain constant.
- Binary numeric op with exactly one constant input may be rewritten to an optimized immediate form.
- Optimized semantic IR never contains a numeric op with two embedded constants.
- Only numeric pure ops may use the optimized immediate form.

## Semantic IR Extension

The canonical decoded Wasm op vocabulary should remain unchanged.

Add one internal optimized semantic form for numeric binary ops with one embedded constant operand.

Requirements:

- represent the numeric op kind;
- represent the numeric constant payload;
- represent which operand side is constant for non-commutative ops;
- carry stack effect as `pop 1, push 1`.

Do not add many opcode-specific immediate variants.

## Stack-Effect Rules

Once optimized semantic ops exist, stack effect must be computed from the full semantic op kind, not only from the decoded primitive opcode.

Examples:

- canonical `i32.add`: pop 2, push 1;
- optimized `i32.add` with one embedded constant: pop 1, push 1;
- folded `i32.const`: pop 0, push 1.

This logic must be centralized and shared by:

- semantic validation;
- max-stack-height recomputation;
- the new Wasm optimizer;
- any middle preparation code that reasons about semantic stack effects.

## Optimization Scope

In scope:

- numeric unary ops for `i32`, `i64`, `f32`, `f64`;
- numeric binary ops for `i32`, `i64`, `f32`, `f64`.

Out of scope for the first implementation:

- memory ops;
- table ops;
- global ops;
- calls and other boundary-like operations;
- any non-numeric optimization that could change backend pressure in unclear ways.

## Backend Contract

The semantic optimizer may reduce transient live-value pressure, but it must not assume that constant operands disappear for every ISA.

Backend contract:

- full constant collapse (`const`, `const`, `binop` -> `const`) is always safe;
- one-constant optimized numeric binops must lower to existing backend machinery using either:
  - an ISA immediate form when available, or
  - backend-local constant materialization into fixed scratch registers when not available.

This means the optimization reduces semantic and spill/fill pressure without requiring every target to encode the constant directly.

## Implementation Order

1. Keep the conservative MachineIR fix in place and validate the Lua workloads.
2. Extend semantic IR with one optimized numeric-binop-with-constant form.
3. Centralize semantic stack-effect computation on full semantic op kinds.
4. Add one Wasm-side post-inline numeric constant-folding pass.
5. Recompute semantic metadata after that pass.
6. Teach middle lowering to consume the new optimized semantic numeric form.
7. Remove nontrivial semantic constant folding from MachineIR permanently.

## Success Criteria

- Lua workloads pass with the conservative MachineIR baseline.
- Wasm-side folding reduces spill/fill planning opportunities before `middle/`.
- Optimized semantic IR never requires two embedded constants.
- Backend lowering remains target-independent at the IR boundary.