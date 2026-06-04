---
commit: a852850, 2404c7a
---
The project passes the official WebAssembly spec testsuite via `sf-nano-spectest`
(a ~1200-line `wast_test_runner`), shipped from the initial commit alongside a
smoke test (add + Fibonacci). Full WebAssembly 2.0 is exercised: multi-value,
reference types, bulk memory, multiple tables, mutable global import/export. "Passes
spectest" is the recurring promotion fact: it is what moves each new execution
strategy and each new backend from `explored-untested` to `tested`. It later
generalizes — base interpreter vs micro-JIT differential equivalence, then the
MachineIR emulator as a non-host oracle, then cross-target spectest green across
arm64 / x86_64 / armv7a / emu64 / emu32. The observation is a binary correctness
gate (pass/fail against an external authoritative suite), not a performance number.
