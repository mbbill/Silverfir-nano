# PR 42 CI investigation

Original head: df70f1c9a1b72e46b29ab77620cc76458873382f.
Correctness run 34092213833: complete, 6 successful and 5 failed actual jobs.
Performance run 34092213857: still running; no complete performance verdict.

## Reproduced and locally fixed

1. Workspace feature unification enables tracked allocation; memory_length,
   integer_shifts and narrow_loads compared returned tracked vectors directly
   with standard vectors/arrays. Explicit slice comparisons preserve all value
   checks. ARM local workspace debug: 683 passed, 4 existing ignored, zero
   compiler warnings. The five focused memory tests also pass with memprof.
2. Nightly Cargo reports unused paste for core JIT-only and interp-only builds.
   The untouched main f73219f reproduces it; no Rust source uses the dependency.
   Remove the core dependency and update the four consumer lockfiles. Other
   device dependencies that genuinely need paste retain it. Both nightly core
   configurations now build without warnings.
3. Zig receives rustc's nonexistent tier-3 prebuilt-std search directory and
   unsupported generic ELF linker optimization option. An argument adapter
   removes only that absent target-specific search path and ignored -Wl,-O1.
   Linker stderr is untouched, unknown paths/options remain diagnosable.
   Native host cross-link probes reproduce warnings before and none after;
   controlled debug and release ELF outputs are byte-identical before/after.
   Three adapter tests plus the complete 121 CI unit tests pass. The actual
   RV32 JIT spectest also cross-builds without warnings. All eight affected
   x64/Rosetta integration tests pass with memprof enabled.

## Feature ownership decision pending

Nightly spectest interp-only declares four unused dependencies: sf-nano-core,
wat, env_logger and structopt. Both untouched main and the PR reproduce them.
The binary only prints a missing-JIT-driver message and exits 2. The source
feature comment describing an interp-only runner contradicts that behavior.

Proposed design: explicitly require jit for the WAST binary; interpreter spec
execution remains jit,interp and pure-interpreter compile coverage remains in
core and CLI. Alternative: preserve the diagnostic placeholder and separate
its dependencies from the WAST runner. AGENTS.md requires this feature-boundary
cluster be reported before changing ownership; the user decision is pending.
No suppressions, new engine cfg structure or warning exemptions were added.

## Performance policy

The already accepted ERC20 startup cost remains recorded. A failed performance
gate cannot be called green merely because that tradeoff is accepted. Keep all
remaining primary and confirmation results before deciding next action.
