# Performance Optimization Guide

This document captures the disciplined process for finding and fixing
codegen inefficiencies in the native backend. The reference point is
wasmer's LLVM backend: for any function where Silverfir emits
significantly more instructions than LLVM, there is optimization work
to do.

Every rule below was learned from a real debugging session.

## Core Principle

**Never touch code until you can show the problem in a concrete example
with exact numbers.**

Previous sessions failed because they:

- jumped from a hypothesis directly to a code change
- skipped the "show me the evidence" step
- made speculative fixes that introduced new bugs or had no measurable effect
- tried to fix multiple things at once

The correct sequence is:
measure → compare against LLVM → pick one function → show the full IR
chain → find the root cause in code → add tests → fix → verify.

## Prerequisites

Silverfir (this repo):

```bash
cargo build --release --bin sf-nano-cli
```

Wasmer with LLVM backend (see `~/Dev/wasmer/LLVM.md`):

```bash
cd ~/Dev/wasmer
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm
cargo build --release -p wasmer-cli --features llvm
```

## The Process

### Step 1: Generate Silverfir and LLVM dumps

```bash
# Silverfir
cargo build --release --bin sf-nano-cli
rm -rf /tmp/coremark-sf-dump
SF_NATIVE_DUMP_DIR=/tmp/coremark-sf-dump \
  ./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

# Wasmer LLVM
rm -rf /tmp/coremark-llvm-debug
mkdir -p /tmp/coremark-llvm-debug
~/Dev/wasmer/target/release/wasmer compile --llvm \
  --compiler-debug-dir /tmp/coremark-llvm-debug \
  -o /tmp/coremark.wasmu \
  benchmarks/wasi/coremark/coremark.wasm
```

### Step 2: Postprocess and compare

```bash
python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-sf-dump \
  --out-dir /tmp/coremark-sf-pp

python3 scripts/compare_llvm.py \
  --sf-dir /tmp/coremark-sf-pp \
  --llvm-dir /tmp/coremark-llvm-debug
```

This prints a summary table of every function with SF and LLVM
instruction counts and their ratio (SF / LLVM), sorted worst first.

### Step 3: Profile for hotness and pick ONE function

```bash
SF_JITDUMP=1 samply-for-ai record --save-only --output /tmp/profile.json.gz -- \
  ./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

python3 scripts/compare_llvm.py \
  --sf-dir /tmp/coremark-sf-pp \
  --llvm-dir /tmp/coremark-llvm-debug \
  --profile /tmp/profile.json.gz
```

Pick the function with the highest `ratio * hotness%`. The ideal
candidate is hot in the profile, has a high ratio, and is representative
of a general pattern (not a one-off edge case).

### Step 4: Drill down into the full IR chain

```bash
python3 scripts/compare_llvm.py \
  --sf-dir /tmp/coremark-sf-pp \
  --llvm-dir /tmp/coremark-llvm-debug \
  --function 6
```

This shows, for the chosen function:

1. **Wasm disassembly** — what the original code does
2. **SSA IR** — Silverfir's middle-layer representation
3. **Machine IR** — Silverfir's machine-level representation
4. **Silverfir assembly** — what we emit (with instruction count)
5. **LLVM assembly** — what LLVM emits (with instruction count)
6. **LLVM optimized IR** — what LLVM's optimizer produced

### Step 5: Identify the pattern

Compare the two assemblies instruction by instruction. Categorize every
extra Silverfir instruction:

- Redundant moves (register-to-register copies LLVM avoids)
- Missing strength reductions (LLVM uses a cheaper instruction)
- Redundant loads/stores (LLVM keeps values in registers)
- Missed constant folding (LLVM evaluates at compile time)
- Extra branch overhead (block layout, unnecessary jumps)
- Calling convention overhead (spill/fill around calls)

Count them. For example:

> "12 of 28 extra instructions are redundant GP moves. 8 are redundant
> loads from the linear memory base. 4 are from suboptimal block layout."

### Step 6: Find the root cause in code

Now — and only now — go into the source code. You know exactly what
pattern to look for because you have the concrete IR.

Trace through the lowering of the specific SSA ops that produce the
extra instructions.

Key questions:

- What does LLVM's optimized IR do that Silverfir's SSA IR does not?
- Is there a conservative copy that could be eliminated?
- Is there a missing analysis pass that would avoid the extra instructions?
- Is the machine lowering choosing a suboptimal instruction sequence?

### Step 7: Verify correctness constraints before fixing

Before writing the fix, verify that the optimization is safe:

- Read every function that touches the affected state
- Confirm that the safety mechanisms are already in place
- Check for invariants that might have motivated the conservative approach

### Step 8: Write tests FIRST

Write tests that cover:

1. **The optimization itself** — verify that the redundant instruction
   is gone
2. **The safety invariant** — verify that the mechanism that makes the
   optimization safe actually fires
3. **The common pattern** — verify the end-to-end pattern that will
   appear in real code

Run the existing test suite to establish the baseline of what currently
passes.

### Step 9: Apply the fix

Make the minimal code change. Do not refactor surrounding code. Do not
add features. Do not clean up.

### Step 10: Verify

Run in this exact order:

```bash
# 1. Unit tests
cargo test -p sf-nano-core --features jit --lib

# 2. Spectests
cargo run --bin sf-nano-spectest -- --backend native

# 3. Rebuild release and re-run comparison
cargo build --release --bin sf-nano-cli
rm -rf /tmp/coremark-fixed-dump
SF_NATIVE_DUMP_DIR=/tmp/coremark-fixed-dump \
  ./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-fixed-dump \
  --out-dir /tmp/coremark-fixed-pp

python3 scripts/compare_llvm.py \
  --sf-dir /tmp/coremark-fixed-pp \
  --llvm-dir /tmp/coremark-llvm-debug \
  --function 6
```

Confirm the ratio improved for the target function.

### Step 11: Record the result

Write down the final numbers:

| Metric | Before | After | LLVM |
|--------|--------|-------|------|
| CoreMark | ... | ... | N/A |
| Insn count (func N) | ... | ... | ... |
| Ratio (func N) | ... | ... | 1.00 |
| Code size | ... | ... | ... |

Note what gap remains and what the next optimization target would be.

## Rules

### One optimization at a time

Never try to fix two things in one pass. Fix one, verify, measure, then
move to the next. Batching changes makes it impossible to attribute
improvements or diagnose regressions.

### Always pick the highest-value target

Use the profile-weighted comparison to pick the optimization that
affects the hottest code with the worst ratio. A 10% improvement in a
block that runs 15% of the time is worth more than a 50% improvement in
a block that runs 0.1% of the time.

### The concrete example is mandatory

If you cannot show a specific function with specific SF-vs-LLVM assembly
side by side, you do not understand the problem well enough to fix it.
Go back to Step 4.

### Trust the data, not the hypothesis

If the SSA IR is clean but machine IR diverges from LLVM, the problem is
in machine lowering. If SSA IR is already bloated compared to LLVM's
optimized IR, the problem is in the middle layer. Let the numbers and
the comparison tell you where to look.

### Read the design docs and code structure before fixing

Before writing any fix, read the relevant design documents and the
surrounding code in the layer where the problem lives:

- **Middle layer** (`sf-nano-core/src/vm/middle/`): region solver
  (`joint_plan/region_solver.rs`), rewriter (`rewrite/function.rs`),
  SSA IR (`ssa_ir/ir.rs`), cache planning (`joint_plan/build.rs`)
- **Machine layer** (`sf-nano-core/src/vm/machine/`): lowering context
  (`lower_context.rs`), instruction lowering (`lower_inst.rs`),
  register allocation (`lower_regalloc.rs`), cache layout
  (`lower_cache_layout.rs`), peephole passes (`peephole/`)
- **Design docs**: `ALGORITHM4.md` (cost-optimal public residency via
  region-tree DP), `LANE_MAPPING.md` (order-aware cache placement)

The fix must be aligned with the existing architecture. A local hack
that ignores the design will break invariants or conflict with planned
work. If the design doc describes a mechanism that should handle your
case but does not, the fix belongs in that mechanism — not in a new
ad-hoc path.

### Do not speculate about correctness

When you find a conservative code path (e.g., always emitting a copy),
do not assume it was conservative for a reason. Read the safety
mechanisms. If the safety infrastructure is already in place, the
conservative path is just a bug.

## Example: Cache↔Linear Move Explosion (April 2025)

This is a concrete example of applying the process above.

### Measure and profile

CoreMark dropped from 33,111 to 22,484. MIR ops went from 10,985 to
18,822. Profiling showed `func6::b7` (a linked-list reversal loop) at
8.6% of execution time.

### Show the full IR chain

The Wasm source is a simple loop over three locals:

```wasm
loop $L9
  local.get $l2 / local.tee $l4 / i32.load / local.set $l2
  local.get $l4 / local.get $l6 / i32.store
  local.get $l4 / local.set $l6
  local.get $l2 / br_if $L9
end
```

The divergence was entirely in MIR. The bloated version had 11 ops
where 7 were cache↔linear register copies. The lean version had 4 ops.

### Identify the pattern

Every `local.get_cache` produced a `move.linear` (cache → linear copy).
Every `local.set_cache` consumed a linear value with a `move.cache`
(linear → cache copy).

### Find the root cause

The old code path source-aliased cache registers (zero instructions).
The new code path always allocated + copied (one instruction per get).
All safety mechanisms (`materialize_cache_aliases` at every mutation
point) were already in place. The conservative copy was not protecting
against anything.

### Test, fix, verify

Three tests were added. The fix was a three-line change in
`lower_local_get_cache`. Result:

| Metric | Before | After |
|--------|--------|-------|
| CoreMark | 22,484 | 29,263 |
| MIR ops | 18,822 | 14,232 |
| Hot loop ops | 11 | 5 |
| move.linear (func6) | 267 | 29 |

## Quick Reference

| What | Silverfir | Wasmer LLVM |
|------|-----------|-------------|
| Build | `cargo build --release --bin sf-nano-cli` | `cargo build --release -p wasmer-cli --features llvm` |
| Dump | `SF_NATIVE_DUMP_DIR=/tmp/sf` | `--compiler-debug-dir /tmp/llvm` |
| Assembly | `functions/NNNN/native_disasm.txt` | `llvm/<hash>/function_N.s` |
| IR | `functions/NNNN/ssa_ir.txt` + `machine_ir.txt` | `function_N.postopt.ll` |
| Index | Global wasm (includes imports) | Global wasm (includes imports) |
