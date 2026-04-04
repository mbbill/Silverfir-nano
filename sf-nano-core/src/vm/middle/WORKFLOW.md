# Middle-End Optimization Workflow

This file defines the workflow for improving planner policy in `src/vm/middle`.

We optimize one planner surface at a time, not the whole pipeline at once.

Current planner surfaces:

1. `block_open(block)`
2. `block_exit(block)`
3. `local_access(query)`
4. `before_op(query)`
5. `edge_repair(query)`
6. `target_entry(semantic_index)` if the transient model itself changes

## Per-Item Workflow

For each planner item:

1. Understand what the item does by reading the code.
2. Pick one hot function and a small stable hot-block set.
3. State a concrete hypothesis before changing code.
4. Dump and inspect the current SSA-IR / MachineIR / native disassembly.
5. Decide whether there is clear evidence that the current policy is wrong.
6. If the policy is fine, skip it and move to the next item.
7. If it is not fine, identify the root cause and make sure the cause is understood.
8. Apply an improvement or fix only after the cause is clear.
9. Rebuild, redump, and compare the same function and blocks.
10. Check for obvious regressions in the dumped IR.
11. Run correctness gates.
12. Run CoreMark again and decide whether to keep the change.

## Profiling and Target Selection

Do not optimize against the whole benchmark blindly.

For one planner item:

1. Use `samply-for-ai` to find one hot CoreMark function.
2. Inside that function, pick a limited number of hot blocks.
3. Keep that block set fixed while working on the item.

The goal is to make before/after comparisons small and readable.

## Hypothesis Discipline

Before editing code, write down:

- the suspected problem
- the expected IR effect

Example:

- `block_open` is carrying the wrong locals into this hot block
- expected effect: fewer `local.ensure_cache`, fewer `local.drop_cache`, less slot traffic on the hot path

Do not change code without a concrete expected effect.

## Static Success Metrics

Before looking at runtime, compare the selected hot function and blocks for:

- `local.ensure_cache`
- `local.drop_cache`
- `local.get_slot`
- `local.set_slot`
- `local.get_cache`
- `local.set_cache`
- `spill`
- `fill`
- edge repair count
- code size

For `block_open`, always inspect:

- predecessor exit cached locals
- successor entry cached locals
- inserted repair blocks

`block_open` must not be judged in isolation from edge repair.

## Correctness Gates

Before rerunning benchmarks:

```bash
cargo check -p sf-nano-core
cargo test -p sf-nano-core
```

If the change touches the CLI or dump path, rebuild the release CLI and redump the target.

## Keep-or-Drop Rule

Keep the change if it is:

- a general policy improvement
- a correctness fix
- a simplification
- a clear hot-path improvement with acceptable complexity

Drop the change if it is:

- a narrow hack
- a rare corner-case special case
- added complexity without clear IR or runtime benefit

Flat CoreMark does not automatically mean revert. General policy improvements and real bug fixes are still worth keeping.

## Artifact Discipline

For each experiment, keep:

- one dump directory
- one postprocessed directory
- one short hypothesis note

Do not overwrite the previous experiment if the before/after comparison matters.
