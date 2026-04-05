# Performance Debugging Guide

This document captures the disciplined process for diagnosing and fixing
performance regressions in the native backend. It exists because rushing
to code changes without concrete evidence consistently fails. Every rule
below was learned from a real debugging session.

## Core Principle

**Never touch code until you can show the problem in a concrete example
with exact numbers.**

Previous sessions failed because they:

- jumped from a hypothesis directly to a code change
- skipped the "show me the evidence" step
- made speculative fixes that introduced new bugs or had no measurable effect
- tried to fix multiple things at once

This session succeeded because it followed a strict sequence:
measure → pick one block → show the full IR chain → find the root cause
in code → add tests → fix → verify.

## The Process

### Step 1: Measure the baseline

Run CoreMark (or the relevant benchmark) on both the old and new code.
Record the exact numbers.

```bash
# Build both
cargo build --release --bin sf-nano-cli
cd /tmp/sf-nano-old-XXXX && cargo build --release --bin sf-nano-cli

# Run both with dumps
rm -rf /tmp/coremark-new-dump
SF_NATIVE_DUMP_DIR=/tmp/coremark-new-dump \
  ./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

rm -rf /tmp/coremark-old-dump
SF_NATIVE_DUMP_DIR=/tmp/coremark-old-dump \
  /tmp/sf-nano-old-XXXX/target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Write down the summary line:

```
[arm64] (func:35, ssa:19893, mir:18822, code:167448)   # new
[arm64] (func:35, ssa:12872, mir:10985, code:83056)    # old
```

If SSA and MIR counts are close, the problem is in codegen or runtime.
If SSA counts diverge, the problem is in middle layer lowering.
If only MIR diverges (SSA similar), the problem is in machine lowering.

### Step 2: Profile and pick ONE hot block

```bash
SF_JITDUMP=1 samply-for-ai record --save-only --output /tmp/profile.json.gz -- \
  ./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

samply-for-ai query --profile /tmp/profile.json.gz hotspots --limit 20
```

Pick the hottest block that is representative of the regression. Do not
pick an edge block or a block that looks unusual. A hot inner loop body
is ideal.

### Step 3: Post-process and show the full IR chain

```bash
python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-new-dump \
  --out-dir /tmp/coremark-new-pp \
  --function 6

python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-old-dump \
  --out-dir /tmp/coremark-old-pp \
  --function 6
```

For the chosen hot block, show the complete chain side by side:

1. **Wasm source** — what the original code does
2. **SSA IR** — old vs new, side by side
3. **Machine IR** — old vs new, side by side
4. **Native assembly** — old vs new, side by side

Write out the comparison in a table or diff format. Be explicit about
every instruction. Do not summarize — list them.

### Step 4: Identify the pattern

From the side-by-side comparison, categorize every extra instruction in
the new code:

- Is it a redundant move? Between what domains?
- Is it a missing optimization that the old code had?
- Is it a new instruction category that did not exist before?

Count the extras. For example:

> "7 of 11 MIR ops are cache↔linear moves. The old code had 4 ops total."

### Step 5: Find the root cause in code

Now — and only now — go into the source code. You know exactly what
pattern to look for because you have the concrete IR.

Trace through the lowering of the specific SSA ops that produce the
extra instructions. Compare the old code path and the new code path
line by line.

Key questions:

- Did the old code have an optimization that the new code dropped?
- Is the new code adding a copy for safety that turns out to be
  unnecessary?
- Is there a missing analysis pass (like `sink_plan`) that the old
  pipeline had?

### Step 6: Verify correctness constraints before fixing

Before writing the fix, verify that the optimization is safe:

- Read every function that touches the affected state
  (e.g., `release_dead_values`, `materialize_cache_aliases`,
  `emit_drop_cached_local`)
- Confirm that the safety mechanisms are already in place
- Check for new invariants in the refactored code that might have
  motivated the conservative approach

### Step 7: Write tests FIRST

Write tests that cover:

1. **The optimization itself** — verify that the redundant instruction
   is gone
2. **The safety invariant** — verify that the mechanism that makes the
   optimization safe actually fires (e.g., alias materialization before
   overwrite)
3. **The common pattern** — verify the end-to-end pattern that will
   appear in real code (e.g., get→set to different slot produces one move)

Run the existing test suite to establish the baseline of what currently
passes.

### Step 8: Apply the fix

Make the minimal code change. Do not refactor surrounding code. Do not
add features. Do not clean up.

### Step 9: Verify

Run in this exact order:

```bash
# 1. Unit tests
cargo test -p sf-nano-core --features micro-jit --lib

# 2. Spectests
cargo run --bin sf-nano-spectest -- --backend native

# 3. Rebuild release and run CoreMark with dump
cargo build --release --bin sf-nano-cli
rm -rf /tmp/coremark-fixed-dump
SF_NATIVE_DUMP_DIR=/tmp/coremark-fixed-dump \
  ./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

# 4. Post-process and verify the hot block improved
python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-fixed-dump \
  --out-dir /tmp/coremark-fixed-pp \
  --function 6
```

Show the before/after comparison of the same hot block to confirm the
fix did what was expected.

### Step 10: Record the result

Write down the final numbers in a comparison table:

| Metric | Old | Before fix | After fix |
|--------|-----|------------|-----------|
| CoreMark | ... | ... | ... |
| MIR ops | ... | ... | ... |
| Code size | ... | ... | ... |

Note what gap remains and what the next optimization target would be.

## Rules

### One optimization at a time

Never try to fix two things in one pass. Fix one, verify, measure, then
move to the next. Batching changes makes it impossible to attribute
improvements or diagnose regressions.

### Always pick the highest-value target

Use the profile to pick the optimization that affects the hottest code.
A 10% improvement in a block that runs 15% of the time is worth more
than a 50% improvement in a block that runs 0.1% of the time.

### The concrete example is mandatory

If you cannot show a specific block with specific old-vs-new IR and
assembly, you do not understand the problem well enough to fix it. Go
back to Step 3.

### Trust the data, not the hypothesis

If the SSA IR is identical but MIR diverges, the problem is in machine
lowering — do not investigate the middle layer. If SSA diverges, the
problem is in the middle layer — do not investigate machine lowering.
Let the numbers tell you where to look.

### Trace the old code path

When the old code was faster, the question is always "what did the old
code do that the new code does not?" Read the old lowering path for the
specific op that produces extra instructions. The answer is usually a
concrete optimization (like source-aliasing or sink planning) that was
not carried forward.

### Do not speculate about correctness

When you find a conservative code path (e.g., always emitting a copy),
do not assume it was conservative for a reason. Read the safety
mechanisms. Check whether the old code had the same mechanisms. If the
safety infrastructure is already in place, the conservative path is
just a bug.

## Example: Cache↔Linear Move Explosion (April 2025)

This is a concrete example of applying the process above.

### Step 1–2: Measure and profile

CoreMark dropped from 33,111 (old) to 22,484 (new). MIR ops went from
10,985 to 18,822. Profiling showed `func6::b7` (a linked-list reversal
loop) at 8.6% of execution time.

### Step 3: Show the full IR chain

The Wasm source is a simple loop over three locals:

```wasm
loop $L9
  local.get $l2 / local.tee $l4 / i32.load / local.set $l2
  local.get $l4 / local.get $l6 / i32.store
  local.get $l4 / local.set $l6
  local.get $l2 / br_if $L9
end
```

SSA IR was nearly identical between old and new. The divergence was
entirely in MIR:

Old MIR (4 ops):
```
move.gp          r4  <- r9
indexed_load.u32 r9  <- [base + r4]
indexed_store.u32     [base + r4] <- r12
move.gp          r12 <- r4
branch r9 then b10 else b11
```

New MIR (11 ops):
```
move.linear.gp  r7 <- r4
move.cache.gp   r6 <- r7
move.linear.gp  r7 <- r6
indexed_load    r7 <- [base + r7]
move.cache.gp   r4 <- r7
move.linear.gp  r7 <- r6
move.linear.gp  r8 <- r5
indexed_store   [base + r7] <- r8
move.linear.gp  r7 <- r6
move.cache.gp   r5 <- r7
move.linear.gp  r7 <- r4
branch r7 then b7 else b76
```

7 of 11 ops were cache↔linear register copies.

### Step 4: Identify the pattern

Every `local.get_cache` produced a `move.linear` (cache → linear copy).
Every `local.set_cache` consumed a linear value with a `move.cache`
(linear → cache copy). The old code had neither.

### Step 5: Find the root cause

Tracing the old code path for `LocalGet` on a cached local:

```rust
// Old: source-alias, zero instructions
self.push_value_location(*dst, cached.reg, None);
```

The new `lower_local_get_cache` instead:

```rust
// New: allocate + copy, one instruction per get
let dst_reg = self.alloc_slot_load_value(dst)?;
self.emit_machine_inst(Move { dst: dst_reg, src: Reg(cached.reg) });
```

### Step 6: Verify safety

The aliasing approach requires that when a cache register is about to be
overwritten, any live SSA values aliased to it are copied out first.

Checked all mutation points:
- `lower_local_set_cache` calls `materialize_cache_aliases` before
  overwriting
- `emit_drop_cached_local` calls `materialize_cache_aliases` before
  unbinding
- `release_dead_values` knows cache regs are not linear and does not
  free them
- `apply_sink_premap` calls `materialize_cache_aliases` before
  pre-mapping a result to a cache register

All safety mechanisms were already in place. The conservative copy was
not protecting against anything.

### Step 7–9: Test, fix, verify

Three tests were added:
1. `local_get_cache_source_aliases_without_move` — no Move emitted
2. `local_set_cache_materializes_live_alias_before_overwrite` — safety
   mechanism fires
3. `local_get_cache_to_set_cache_different_slot_single_move` — common
   copy pattern produces one move, not two

The fix was a three-line change in `lower_local_get_cache`. Result:

| Metric | Old | Before | After |
|--------|-----|--------|-------|
| CoreMark | 33,111 | 22,484 | 29,263 |
| MIR ops | 10,985 | 18,822 | 14,232 |
| Hot loop ops | 4 | 11 | 5 |
| move.linear (func6) | 0 | 267 | 29 |
