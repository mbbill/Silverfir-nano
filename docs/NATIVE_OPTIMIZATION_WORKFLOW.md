# Native Optimization Workflow

This document defines the standard workflow for native/JIT performance work.
The goal is to improve runtime performance without turning the compiler pipeline
into an ad hoc pile of benchmark-specific hacks.

Use this workflow for any optimization that touches:

- LIR preparation in `sf-nano-core/src/vm/plan/prepare/`
- native lowering in `sf-nano-core/src/vm/native/lower/`
- MachineIR and peepholes in `sf-nano-core/src/vm/native/ir/machine/`
- ARM64 codegen in `sf-nano-core/src/vm/native/arch/arm64/`

This workflow complements:

- [DEBUG.md](./DEBUG.md) for tooling and command details
- [NATIVE_DESIGN.md](./NATIVE_DESIGN.md) for backend ownership and design rules

## Goals

- Raise native/JIT runtime performance toward Cranelift and V8.
- Keep the pipeline understandable and mechanically defensible.
- Preserve correctness first.
- Make every optimization explainable in terms of generated code.

## Non-Goals

- Do not improve scores by changing benchmark settings or runtime budgets.
- Do not win by increasing cache-register counts, LIR lane counts, or other
  tuning knobs unless that setting change is itself the explicit subject of
  design discussion.
- Do not land an optimization that cannot be justified by a real codegen issue.
- Do not keep a change just because one noisy benchmark run happened to go up.

## Design Principles

1. Optimize the right layer.

If the bad pattern is visible in LIR, fix preparation. If it appears only after
lowering, fix lowering or MachineIR. If MachineIR is already good and assembly
is still poor, fix the backend.

2. Prove the codegen issue before changing code.

Every optimization must start from a concrete bad pattern in hot code:
redundant moves, repeated reloads, avoidable helper boundaries, poor branch
shape, wasted materialization, unnecessary spills, duplicated checks, or bad
instruction selection.

3. Correctness is a gate, not a follow-up.

No performance claim matters if `cargo test`, spectest, or `run_tests.py` fail.

4. Structural changes are allowed, but not casually.

If an optimization requires a new IR concept, block contract, register class,
calling convention, or helper ABI rule, stop and review the design with a
human before landing it.

5. Generated code is the ground truth.

Before trusting benchmarks, inspect the emitted LIR, MachineIR, and assembly.
If the expected codegen improvement is not visible, assume the optimization is
not working yet.

6. Negative results are useful.

If an experiment regresses performance or fails to improve codegen, record it.
Do not rely on memory.

## Pipeline Map

Before starting a new optimization area, read the relevant pipeline end to end:

1. LIR preparation

- `sf-nano-core/src/vm/plan/prepare/`

2. Native lowering

- `sf-nano-core/src/vm/native/lower/`

3. MachineIR structure, validation, and peepholes

- `sf-nano-core/src/vm/native/ir/machine/`

4. ARM64 backend

- `sf-nano-core/src/vm/native/arch/arm64/`

5. Emulator and debugging backstops

- `sf-nano-core/src/vm/native/arch/emulator/`
- `sf-nano-core/src/vm/native/ir_dump.rs`

Do not start by editing blind. Build a mental model of which layer owns what.

## Standard Optimization Loop

### 0. Read first

For a new optimization topic, read:

- the relevant source area
- the relevant sections of [NATIVE_DESIGN.md](./NATIVE_DESIGN.md)
- the tooling sections in [DEBUG.md](./DEBUG.md)

Do this once per topic, not necessarily before every tiny follow-up patch.

### 1. Pick one target benchmark `X`

Choose one benchmark from `benchmarks/wasi/run_tests.py` as the current target.
Prefer the benchmark with the largest clear gap against Cranelift/V8 or the
most obviously poor native code shape.

Rules:

- Work one primary target at a time.
- Use [benchmarks/wasi/RESULTS.md](../benchmarks/wasi/RESULTS.md) and the most
  recent local runs to decide where the gap is largest.
- Do not bounce between unrelated benchmarks without a reason.

### 2. Measure the baseline for `X`

First build the release CLI:

```bash
cargo build --release -p sf-nano-cli
```

Run `X` three times and record:

- the exact command
- the machine state if relevant
- all three results
- mean, min, and max

For `X`, use the same `cwd`, wasm file, arguments, and stdin shape that
`benchmarks/wasi/run_tests.py` uses. If needed, read `TESTS` in that script and
copy the exact invocation.

Then run the full suite once:

```bash
cd benchmarks/wasi
python3 run_tests.py --exec /abs/path/to/target/release/sf-nano-cli --cli-args "--backend native"
```

The single-benchmark baseline tells you what you are targeting. The full-suite
baseline tells you what you might accidentally break.

### 3. Profile the target

Use `samply-for-ai` on benchmark `X` to identify hot functions and blocks.

Then dump the native output for the same workload:

```bash
SF_NATIVE_DUMP_DIR=/tmp/native-dump \
./target/release/sf-nano-cli --backend native path/to/workload.wasm ...
```

Use:

- `native_index.txt` for LIR and MachineIR
- `native_code.bin` for emitted bytes
- `samply-for-ai query ... hotspots` for runtime hotness
- `samply-for-ai query ... asm "<symbol>"` for disassembly

### 4. Find a real codegen issue

Do not move on until you can point to a specific hot pattern and explain why it
is bad.

Good examples:

- boolean compare result is materialized only to feed a branch
- a frame slot is stored and immediately reloaded in the same hot block
- float values bounce through GP regs around every operation
- the same value is reloaded because the IR lost the dependency
- a cold trap/helper path is inlined into hot code
- a value is copied multiple times because lowering inserted avoidable moves

Bad examples:

- "this benchmark is slower, maybe try more peepholes"
- "Cranelift is faster here, maybe copy what it does"

### 5. Decide the owning layer

Ask where the issue first appears:

- In LIR: fix preparation.
- In MachineIR: fix lowering or a mechanical MachineIR pass.
- Only in assembly: fix ARM64 instruction selection or emission shape.

Prefer the highest layer that can fix the problem cleanly.

### 6. Write a hypothesis and success criteria

Before changing code, write down:

- the hot symbol(s)
- the bad pattern
- why it happens
- the layer that should own the fix
- what IR or assembly difference should appear if the fix works
- what benchmark change would count as meaningful

If you cannot state the expected codegen delta, the plan is not ready.

### 7. Discuss structural changes before landing

If the fix needs a structural change, stop here and review it with a human.

Examples:

- new MachineIR instruction kinds
- new register classes
- new block-param or edge-arg contracts
- helper ABI changes
- new pinned-register policy
- new persistent state across blocks or boundaries

Structural changes are allowed when they simplify ownership and make future
optimizations cleaner. They should not be smuggled in as benchmark patches.

### 8. Implement the smallest clean fix

Implementation rules:

- keep the fix local to the owning layer when possible
- add tests that cover the new invariant or regression
- avoid benchmark-specific special cases
- avoid comments that merely narrate syntax

### 9. Prove correctness immediately

Run, at minimum:

```bash
cargo test -p sf-nano-core -- --nocapture --test-threads=1
cargo run --bin sf-nano-spectest -- --backend native
cd benchmarks/wasi
python3 run_tests.py --exec /abs/path/to/target/release/sf-nano-cli --cli-args "--backend native"
```

If anything fails, fix correctness before returning to performance work.

### 10. Use the emulator when the bug is hard to localize

Use `--emu` when needed.

Rules:

- if the bug reproduces on the emulator, suspect LIR, lowering, or MachineIR
- if the bug does not reproduce on the emulator, suspect backend codegen or
  runtime ABI issues
- use function trace and dump diffing to find the first divergence

### 11. Re-dump IR and assembly before benchmarking

This step is mandatory.

After the code is correct, dump the target again and inspect:

- LIR
- MachineIR
- disassembly
- hot symbols from the new profile if needed

Do not benchmark yet if the expected codegen difference is absent.

If the codegen did not improve:

- stop
- find out why
- either fix the implementation or abandon the hypothesis

### 12. Re-measure `X`

Run benchmark `X` three times again and compare with the original baseline.
Then run the full suite once again.

For small deltas:

- if the apparent change is below about 5%
- or if CPU frequency / machine load may have affected results
- run more samples before making a claim

Recommended rule:

- use at least 5 to 10 samples when the expected gain is small
- treat a stable regression as real even if it is small

### 13. Decide whether the gain is real

A change is ready to land only if all of these are true:

- correctness gates are green
- the expected codegen improvement is visible
- the target benchmark shows a repeatable improvement
- the full suite shows no unexplained regression

If any of these fail, do not call it a win.

### 14. Clean up and record the result

Before reporting or landing:

- remove dead code from abandoned approaches
- remove temporary debug-only prints unless they are intentional tooling
- make sure tests and docs match the final design

Then write down what happened, including failed ideas.

## Comparing Against Other Engines

Using Wasmtime/Cranelift or V8 is allowed when it helps explain a real gap.

Rules:

- compare only for a concrete hotspot
- use the other engine to understand code shape, not to cargo-cult blindly
- copy the underlying idea only if it fits Silverfir's design
- reject ideas that require turning the pipeline into a full traditional
  optimizer unless that is a deliberate architectural decision

## Standard Landing Checklist

Before asking for review or claiming a performance win:

- hotspot identified
- bad pattern proven in dumps or assembly
- owning layer chosen
- structural change discussed with a human if needed
- tests added
- `cargo test` passes
- spectest passes
- `run_tests.py` passes
- pre-change and post-change codegen inspected
- benchmark `X` rerun enough times to rule out noise
- negative result recorded if the idea was dropped

## Experiment Record Template

Copy this template into the active task notes, a task-specific follow-up
document, or an appended dated section in this document after each meaningful
experiment.

```text
Date:
Target benchmark:
Hot symbol(s):

Hypothesis:

Evidence before:
- hotspot:
- LIR issue:
- MachineIR issue:
- assembly issue:

Change:

Correctness:
- cargo test:
- spectest:
- run_tests.py:

Codegen after:
- LIR delta:
- MachineIR delta:
- assembly delta:

Performance:
- target baseline:
- target after:
- full-suite notes:

Decision:
- land / revise / abandon

Notes:
```

## Practical Guidance

- Prefer one clean optimization at a time over stacked experiments.
- Prefer a principled ownership fix over a clever peephole when both solve the
  same problem.
- If a change helps one benchmark but causes a repeatable regression elsewhere,
  treat that as a real design tradeoff, not noise.
- If the code becomes harder to reason about, the burden of proof goes up.
