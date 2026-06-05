- Correctness is validated against the official WebAssembly spec testsuite.
- The suite stays green; a change that turns it red is blocked or reverted.
- The suite is consumed directly as `.wast` via the wast/wat crates,
  downloaded and version-pinned by a build script — no offline conversion
  step. (Harness-only dependencies; the production frontend stays
  from-scratch.)
