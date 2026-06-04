# Spectest harness + differential oracle

Define correctness as agreement with the official WebAssembly spec testsuite,
executed through `sf-nano-spectest` (the `wast_test_runner`), and extend it into
a differential oracle that cross-checks backends against each other. This is the
ground truth every execution strategy is measured against, deliberately
independent of how fast any strategy runs.

The harness drives any backend via `set_backend_mode` / `BackendMode`. Beyond the
host backends it gains a portable non-host oracle: the MachineIR emulator
(`backend-emu64` / `backend-emu32`), used to run cross-target spectest and to
diff one backend's results against another's.

## In practice

Must:
- Correctness must be checked against the official Wasm spec suite run through
  `sf-nano-spectest`'s `wast_test_runner`; full WebAssembly 2.0 must be exercised
  (multi-value, reference types, bulk memory, multiple tables, mutable global
  import/export).
- The harness must be able to select any backend through `set_backend_mode` /
  `BackendMode` and run the same suite against each, including the emulator
  backends (`backend-emu64` / `backend-emu32`) as a portable oracle.
- Passing spectest must be the gate that promotes each new strategy and each new
  arch backend from untested to tested.

Must not:
- Must not depend on the host architecture for the authoritative correctness
  result: the emulator oracle must be runnable as a non-host ground truth.
- Must not treat the smoke test (add + Fibonacci) or any hand-rolled cases as the
  correctness authority in place of the spec suite.
