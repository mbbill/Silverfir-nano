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

## Follow-up on head 8d506555 (2026-09-07)

Correctness run 34094837354 finishes with 8 successful and 3 failed jobs.
The Windows and Linux x64 jobs expose two more tracked-vector/array comparisons
in the x64 encoder library tests. Reproduced both errors locally with
`cargo test -p sf-nano-core --lib --features memprof --target x86_64-apple-darwin`.
Compare slices in both assertions, preserving every expected instruction byte.

The complete x64 library test run then exposed an existing parallel-test race:
`empty_world_after_free_has_no_live_tracked_bytes` enables process-wide tracking
and sees live allocations from other tests (the captured allocation stacks name
`runtime_world_invokes_and_frees_by_generation_checked_id`, among others).
All 564 tests pass serially. Isolate only this allocation assertion in a child
test process; retain the zero-live-bytes check and normal parallel test execution.
After isolation, normal parallel runs pass all 564 x64/Rosetta library tests
and all 577 ARM64 library tests with memprof, with zero compiler warnings.
Formatting, lint policy and whitespace checks also pass.

The RV32 job still fails the four-dependency pure-interpreter spectest warning
audit described above; all its actual spec/WASI execution checks pass. PR #41
contains standalone commit a1da5a7b enabling the shared WAST harness for either
engine independently. That PR's correctness run 34093520841 passes all platforms.
Reusing that specific fix, instead of requiring jit for spectest, has been
presented to the user as the preferred resolution; no feature-boundary change
has been applied here pending the decision.

### Why this run takes longer

Daily correctness.yml and performance-regression.yml are unchanged by this PR.
The only new workflow, x64-standings.yml, is workflow_dispatch-only.
Measured queue delay is the main increase: dev run 34084103483 started its
primary matrix within 3–10 seconds; run 34094837354's armv7, RV32 and x64 Linux
jobs waited 21–23 minutes. Its RV32 bare job waited 20m25s then ran in 48s.
PR #41's workflows overlap this run. An exact account concurrency limit has
not been established from these observations.

The existing PR matrix also adds six emulated cross-target correctness rows
inside the performance workflow compared with dev, plus the correctness
workflow itself. The previous head's performance run 34092213857 was cancelled
after 36m09s by the corrective push, extending the overall wait.

Execution time also increased in some cells: Windows JIT benchmark measurement
692s → 813s, Windows interpreter 473s → 608s, ARM interpreter WASI 412s → 424s.
Windows JIT baseline/candidate build stayed similar (125s → 122s). Both x64
correctness jobs report no cache found. These are workflow timings, not evidence
that the JIT runtime slowed by those amounts. Confirmation matrices currently
allocate runners even for cells that ultimately need no confirmation; reducing
that scheduling overhead is a separate CI improvement opportunity.
