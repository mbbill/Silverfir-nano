# Recent work audit and x64 performance campaign — 2026-09-06

Baseline: `main@f73219f5`. Remote refs were refreshed on September 6.
Scope: branches with tip commits dated August 1 or later. No old branches were
deleted, rebased, or revived. The complete machine-readable inventory is
[`recent-branch-audit-2026-09-06.tsv`](recent-branch-audit-2026-09-06.tsv).

## What remains worth recovering

61 recent branches were inspected. 24 contain no patches absent from main,
using `git cherry main origin/<branch>` rather than ancestry alone. Several
others are overlapping experiments or contain implementation/revert pairs;
their non-equivalent patch count is not a count of useful changes.

| Branch family | Evidence and disposition |
|---|---|
| `dev/x64-tuning`, `dev/alias-preserve-move`, `codex/lz4-lua-performance` | Production changes already integrated, including register returns, near calls, preserved lanes, index proofs, and the Lua timer correction. Nothing to cherry-pick. |
| `codex/execution-performance`, `codex/arm64-lz4-lua-hotpaths` | Integrated via PRs #33 and #34. Global-loop promotion and ARM64 LZ4/Lua optimizations are already in main. |
| `dev/interp-coremark-10pct` | All patches integrated through PR #40. No unfinished +10% gain remains on this branch. |
| `dev/interp-eager-postpass-fast-callgate`, `dev/interp-callgate-ranking`, `dev/manual-startup-ranking` | Clean eager startup and manual field ranking already integrated, PRs #38/#39. The many earlier `v2`/`v3`/in-place variants overlap this line. |
| `dev/interp-cold-startup`, `dev/interp-cold-startup-v2` | Reject. First layout was reverted; v2 PR #37 was closed without merging after independently confirmed x64 startup regression. |
| `dev/interp-selective-predecode` | Reject the integrated Raw/Folded routing candidate. Six independent confirmation jobs fail despite the workflow's top-level success. Its extra 12,802 added lines relative to the clean eager branch do not buy a demonstrated benefit. |
| `dev/interp-baseline-foundations`, `dev/interp-indirect-resolver-fast` | Components of the Raw/Folded experiment. A separate raw cursor and its differential tests may inform future work, but are not a validated standalone optimization of current main. Do not copy the alternate executor into the release tree. |
| `dev/interp-final-profile` | Small candidate: `4233f730` compresses the common decode handoff to 16 bytes and adds decoder equivalence/boundary tests. Exact-tip Callgrind completed; the performance run was cancelled. Recover only as an isolated A/B experiment. |
| `dev/interp-eager-postpass-fast-reserve`, `dev/interp-eager-parser-reserve` | Potentially useful capacity-planning idea, but commits also modify instance/linking storage. Extract only parser reservation if profiling still finds reallocations material; do not cherry-pick the mixed patch wholesale. The extra scan must repay its own startup cost. |
| sparse-head / compact-safe / stream-link variants | Alternate representations with hundreds of changed lines. Revisit only with current memory/latency evidence; not release cleanup candidates. |
| `dev/interp-eager-tier-census`, stage/direct-cell profile branches | Diagnostic instruments, not production speedups. Retain their provenance; recover a tool only for a specific experiment. |
| `dev/probe-counts` | ARM64 hardware-counter probe, outside the present x64 target. |

The most useful immediate recovery is the **deleted x64 comparison tooling**,
from `a6925d9b^`. It makes the performance questions measurable again without
resurrecting already integrated compiler work.

## Failure evidence that must not be lost

- [PR #37](https://github.com/mbbill/Silverfir-nano/pull/37): x64
  `startup/coremark` primary throughput -2.95%, independent confirmation -4.34%.
- [Selective execution run 33279797828](https://github.com/mbbill/Silverfir-nano/actions/runs/33279797828):
  34 jobs, six failed confirmation jobs. On the independent x64 execution
  runner, Sunfish 333.81 → 135.82 (-59.31%), Lua JSON 764.30 → 431.16
  (-43.59%), SQLite 2815.0 → 1994.8 (-29.14%).
- The same candidate's independent x64 startup confirmation fails all seven
  workloads: CoreMark 153.671 → 525.269 µs; FFmpeg 328.447 → 930.449 ms.
  These are throughput regressions of 70.74% and 64.70%, respectively, not
  percentages of elapsed-time increase.
- `4233f730`: [Callgrind run](https://github.com/mbbill/Silverfir-nano/actions/runs/33266482360)
  succeeded; [performance run](https://github.com/mbbill/Silverfir-nano/actions/runs/33266482394)
  was cancelled. Instruction-count data alone does not establish latency or
  steady-state execution benefit.

## Where x64 work should start

Historical [PR #28](https://github.com/mbbill/Silverfir-nano/pull/28) closes at
roughly 1.04× Cranelift / 1.15× V8 elapsed time on the wasmi corpus and 1.14× /
1.28× on the WASI suite. Its prose mixes rate and time ratio notation; consult
the underlying measurement before reusing a number. These are historical
snapshots, not the current ranking. CPU SKUs changed between measurements.

[PR #29](https://github.com/mbbill/Silverfir-nano/pull/29) subsequently removed
LZ4 index extensions and fixed a Lua timer artifact. In its same-host Lua fib
measurement, Nano was already ahead of both competitors. The old Lua gap and
the old proposal to introduce register returns must not be treated as current
unfinished work.

Working target: improve the complete execution corpus's geometric mean while
tracking every individual loss and preserving startup, memory, binary size,
correctness, and other supported targets. Report wasmi and WASI separately;
do not give the four STREAM rates four times the weight of a workload in a
combined headline. A separate workload-balanced analysis can accompany the
existing metric-weighted WASI table. No universal every-program performance
guarantee is implied by a benchmark win.

Priority after fresh same-host rankings and native profiles:

1. **Register pressure and moves.** x64 currently exposes five volatile and
   two preserved GP lanes, one internal scratch, and reserves RAX/RCX/RDX for
   constrained lowering. Look for spill/reload and loop-parameter rotations
   in Argon2, Lua, and tail loops. First improve coalescing and local lowering
   within that contract. Lending fixed registers requires explicit constrained
   operand/liveness design covering div/rem, variable shifts, scalar returns,
   helpers, and both host ABIs; merely adding a register to the bank is unsafe.
2. **Residual call costs.** Profile direct versus exported-table calls and
   Lua/SQLite. Near calls and RDX/XMM0 scalar returns already exist. Separate
   frame/argument movement from open-world funcref resolution before changing
   the world boundary. Preserve callback reentry, table mutation and traps.
3. **FP lowering and scheduling.** Inspect c-ray/nbody for destructive SSE
   copies and long dependency chains. Existing MOVAPS copies and the FP
   literal pool are already integrated. AVX/BMI variants need target feature
   detection and a baseline path; do not conflate Intel versus AMD with ISA
   capabilities or make unchecked feature assumptions.
4. **Large residual algorithms.** Regex, reverse-complement and recursive
   Fibonacci need current profiles. Inlining is not a free catch-all: it was
   deliberately retired, and per-function output size is an MCU constraint
   (`mcts_mem/silverfir/compiler.md`). Reopening it needs an explicit bounded
   design, rather than copying V8's whole-module strategy.

SHA-256's historical register-spill floor and STREAM's bandwidth sensitivity
make them poor first targets without new evidence. No percentage gain is
promised for these hypotheses.

## CI instrument and acceptance

The restored `x64-standings.yml` has **no PR or main trigger**. It is manual-only. The initial baseline was launched from the isolated
`codex/x64-release-audit` branch; subsequent pushes do not repeat competitor
builds. Normal daily correctness and performance jobs retain their existing
scope. Optimization iterations use the fast dev revision-difference jobs;
rerun competitors at campaign close or when a comparison version changes.

- Two independent runner draws for each corpus, all three engines on the same
  machine within a draw; record CPU, source revision and tool versions.
- Pinned wasmi suite `16a3d7c8fdb05506c116a9451175732d1ac77099`; local Nano
  dependency resolution verified before compiling and measuring.
- Clear cached Criterion results, compile before measurement, pin one core
  and disable ASLR. Store estimates and the resolved suite lockfile.
- WASI: reverse engine order in the second draw. This remains a diagnostic
  snapshot, not interleaved statistical proof. CPU and ordering both differ
  between draws, so a difference does not isolate either cause.
- Require all 20 execution cases and all 17 WASI metrics. Missing engines,
  invalid timings, process failures and incomplete fields fail the instrument.
- Each compiler candidate still needs the existing paired revision gate and
  independent confirmation. Inspect steps and summaries for warning audits
  and `ACTION REQUIRED`, not only the workflow conclusion.

Initial run: [34026638361](https://github.com/mbbill/Silverfir-nano/actions/runs/34026638361).
Its repository lint check inadvertently included the separate upstream suite;
the repair runs that check immediately after checking out Nano, before adding
the comparison repository. No lint suppression or upstream source was edited.

Current result: [the complete baseline](performance-baseline-2026-09-06/README.md)
finished in run 34026700835: all four jobs and all non-skipped steps successful.
The 20-case corpus is 1.035–1.038× Cranelift time and 1.149–1.167× V8 time;
WASI is 1.108–1.169× CL and 1.258–1.306× V8, on the SKUs named in the
report. No new x64 speedup has been measured or claimed. Local validation of
the instrument: 118 Python CI
tests passed, including four new incomplete/invalid-result regressions; lint
policy and whitespace checks passed. Rust is unavailable on the local host,
so Rust builds and actual native measurements belong to CI.
