# AGENTS

## Purpose

This file contains persistent instructions for coding agents working in this
repository.

Read this file at the start of work, especially for native/JIT compiler,
performance, and debugging tasks.

## Native/JIT Performance Work

For any native/JIT compiler optimization task, always follow:

- `docs/NATIVE_OPTIMIZATION_WORKFLOW.md`
- `docs/NATIVE_DESIGN.md`
- `docs/DEBUG.md`

Required rules:

- Do not guess about performance problems. Prove them from hotspots, dumps,
  MachineIR, and disassembly.
- Do not claim a performance win unless the generated code clearly improved and
  the benchmark result is repeatable.
- Correctness comes first: `cargo test`, spectest, and `benchmarks/wasi/run_tests.py`
  must pass before treating an optimization as valid.
- Do not improve benchmark scores by changing settings such as cache-register
  counts, lane counts, or other tuning budgets unless the human explicitly asks
  for a settings/design change.
- Structural compiler-pipeline changes are allowed only after careful reasoning,
  validation, and explicit human discussion.

## General Debugging Rules

- Do not speculate when a dump, trace, profile, or binary inspection can answer
  the question.
- For native/JIT codegen issues, inspect all relevant levels:
  - LIR
  - MachineIR
  - disassembly
  - hot symbols from profiling
- If a correctness bug is difficult to localize, use the native emulator
  (`--emu`) to determine whether the issue is in lowering/MachineIR or in final
  backend codegen.
- Use function trace and dump diffing to find the first behavioral divergence.
- Prefer the highest clean fix in the pipeline. If the issue starts in LIR, do
  not patch around it in ARM64 unless there is a strong reason.
- Remove failed experiments, dead code, and temporary debug scaffolding before
  finalizing changes unless the scaffolding is intentionally retained as a
  debugging tool.

## Benchmarking Rules

- Use release builds for performance measurement unless the task explicitly
  requires debug mode.
- When targeting one benchmark from `benchmarks/wasi/run_tests.py`, use the same
  workload shape, args, stdin, and cwd as that script.
- Measure a target benchmark multiple times before and after a change. Treat
  small gains and small regressions carefully and verify they are repeatable.
- Always rerun `benchmarks/wasi/run_tests.py` after a performance change to
  check for regressions outside the target benchmark.
- Before benchmarking a supposed optimization, re-dump the IR and assembly to
  confirm the expected codegen change actually happened.

## Change Hygiene

- Add regression tests for compiler and lowering invariants whenever practical.
- Keep optimization logic mechanically explainable. Avoid benchmark-specific
  hacks and “magic” transforms without a clear ownership story.
- Record both successful optimizations and failed experiments so future work can
  build on actual evidence rather than memory.
