# Silverfir-nano

A `no_std`, zero-runtime-dependency engine that executes WebAssembly 2.0 (later
3.0) as fast as possible while staying small enough to embed — down to a
few-hundred-KB stripped core that fits an MCU. "Fast *and* small *and* portable"
is the standing why; every decision below is a bet about how to get all three at
once on hardware ranging from Apple Silicon to a 520 KB Cortex-M.

This root is the one special-cased node — it does not change. Everything under it
is a search: options are candidate ways to honor the why, facts are the evidence
that re-weights them, and the current design is the `selected: true` traversal
from here down.

The root opens these sub-problems (an AND-branch — all must be answered):

- **execution-strategy/** — how to actually run Wasm. The load-bearing fork; the
  one that pivoted (interpreter → JIT).
- **correctness-validation/** — how we know any execution strategy is *correct*,
  independent of how fast it is. Answered once and reused by every strategy, so
  it is its own AND-sibling rather than buried under one of them.

Only the first two decision-steps and their explored children are recorded here;
this is a bounded reconstruction of the project's earliest reasoning, not its
full history.

## Ground rules — execution-strategy
Must:
- The crate must expose exactly one selectable execution backend feature for the
  hot path; today that is the native/JIT pipeline (`feature = "micro-jit"`).
- The chosen strategy must pass the spectest gate (see
  root.all/correctness-validation.md) before it can be the primary backend.
- Whichever strategy is `selected: true` must be the path `BackendMode::Auto`
  resolves to for normal execution.

Must not:
- Must not ship more than one live hot-path execution backend at once: the
  abandoned strategies' build systems and entry points must be removed, not kept
  compiled-but-dormant.
- Must not let any non-selected strategy remain reachable as a default execution
  mode (a demoted strategy may survive only as a ground-truth oracle, never as
  the primary path).

## Ground rules — correctness-validation
Must:
- Correctness must be defined by the official WebAssembly spec testsuite,
  executed through `sf-nano-spectest` (`wast_test_runner`), not by ad-hoc tests.
- Every execution strategy and every new arch backend must pass the spec suite
  before it is treated as `tested`/promotable.
- The validation harness must be drivable across backends via `set_backend_mode`
  / `BackendMode`, so the same suite runs against each backend and against the
  portable emulator oracle (`backend-emu64` / `backend-emu32`).

Must not:
- Must not couple the correctness definition to a specific execution strategy:
  the gate must survive a backend pivot unchanged.
- Must not substitute hand-rolled correctness cases for the spec suite as the
  authority (smoke tests may exist as a fast pre-check, not as the gate).
