# Retained JIT performance work — 2026-09-06

The selected runtime source is `fc26a6a9d12d8fcc3964c3a52045ad72f5e9ae71`:
the previously validated `42396c9d` improvements plus resource-bounded
structured inlining and its compact candidate cache. The final PR extracts
that exact source/workflow delta from main `f73219f5` and adds curated design
memory. Rejected and unmeasured experiments are excluded.

The user explicitly chose “是，保留。” when asked to retain bounded inlining
and accept ERC20 startup increasing approximately **0.29–0.41 ms**. This is an
execution-first tradeoff for this change, not a new global performance
threshold. Existing warning and regression gates remain unchanged.

## Measured benefit and remaining gap

| Native CPU | Bounded inlining all20 vs 42396 | Standard recursive Fibonacci |
|---|---:|---:|
| AMD EPYC 7763 | +2.1245% | +50.4992% |
| Intel Xeon 8573C | +1.0104% | +23.0366% |
| Neoverse N2 | +0.0697% | +1.5145% |

These are same-host Nano throughput differentials from
[34082432990](https://github.com/mbbill/Silverfir-nano/actions/runs/34082432990).
All 60 rows, including negative observations, are preserved in
[the complete result table](inline-resource-limits-full20-results.json).
Fibonacci uses the suite's original input30. Earlier 53–55% results used
input35 and are not substituted into these standings.

The selected compact cache preserves every generated function from 049874
across all20 plus CoreMark (1285 functions on each architecture), plus all
38 ERC20 functions. CoreMark generated code is unchanged by inlining.

Combining the frozen AMD 7763 competitor anchor with measured Nano
differentials projects all20 at approximately **+0.99% versus V8** and
**+12.10% versus Cranelift**. This is a historical same-CPU-model projection,
not a fresh simultaneous three-engine comparison. CoreMark is approximately
level/slightly ahead of V8 and still **8–9% behind Cranelift**. The No.1
objective has not been achieved; no universal x64 ranking is claimed.

## Accepted startup cost and memory limits

[Native startup run 34086840052](https://github.com/mbbill/Silverfir-nano/actions/runs/34086840052)
compares main, 42396, bounded inlining and the final compact cache over all
seven startup cases in alternating rounds.

| ERC20 CPU | main | Selected candidate | Added latency | Throughput change |
|---|---:|---:|---:|---:|
| N2 | 3.4418 ms | 3.7282 ms | +0.2864 ms | -7.6814% |
| AMD 7763 | 4.6995 ms | 5.1074 ms | +0.4079 ms | -7.9859% |

[All startup results](inline-cache-startup-results.json) retain every measured
comparison. Compact caching improves N2 ERC20 throughput 2.2896% against the
original bounded inliner, but does not eliminate the regression against main.
The original inliner's full dev
[34084103483](https://github.com/mbbill/Silverfir-nano/actions/runs/34084103483)
has a **real confirmed ARM ERC20 startup failure**, documented in
[the complete actual-job audit](dev049-gate-audit.json). User acceptance does
not retroactively make it pass. Final PR CI must likewise be read by actual
job/step, including any confirmed regressions.

Inlining applies only to unlimited compiler-budget configurations with
64-bit GP words. It caps the expanded whole caller at 128 conservative ops,
8 charged locals, 32 frame slots and 32 region-by-local entries; it visits
only original call sites. Finite budgets and all 32-bit GP backends retain
original call boundaries. This protects small-memory policy without claiming
that all hosted functions have unchanged footprint: the local Fibonacci
compile-heap peak grows from 6733 to 12648 requested bytes. The counter
excludes code/guard mmap, allocator bookkeeping and parsing/setup.

## What is retained

- Generic guarded byte-copy recovery and conservative native-frame loop caches.
- x64 arithmetic, width-correct flag reuse, BMI selection, CPU-specific scalar
  conversion preferences, duplicate-table lowering and scalar early returns.
- Removal of redundant frame stores and unnecessary compiler analysis work,
  including the empty-cache-bank fix that restored the previous full gate.
- Resource-bounded structured inlining with the compact eight-candidate cache.
- Correctness fixes for memory64/template frames, ARM scratch pressure and
  concurrent signal-trap lookup/code registration.

## Rejected and shelved work

Loop-cache publication plus GP reuse gained roughly 0.54% CoreMark in two
12-pair draws, then regressed sort 33.10% and 25.96% in two full20 draws.
[All three native jobs' 60 rows](loop-cache-register-full20-results.json)
remain recorded. This candidate is rejected. Cross-edge forwarding, broader
cache policies and numerous encoding/layout candidates are also excluded;
[the chronological experiment log](EXPERIMENT_LOG.md) preserves their final
verdicts and earlier hypotheses with source/CI identities.

Local-bank reuse `b5be4dad` has correctness and heap evidence but no native
performance result; its smaller Fibonacci frame increased compile-heap peak.
Interpreter `memory.size` `6a382ad6` has only uncertain all20 gains, and native-
aware load fusion `4d144428` has no native timing. Both remain independent and
unmerged. Generic interpreter work can resume later; no benchmark-specific
fusion instruction is introduced here.

## Design memory and validation

Start with [the campaign decision record](../../mcts_mem/silverfir/compiler.fact/x64-execution-2026-09-06.md),
[bounded inlining](../../mcts_mem/silverfir/compiler/semantic-ir/bounded-inlining.md)
and [loop-cache rejection evidence](../../mcts_mem/silverfir/compiler/machine-peephole/loop-frame-cache.md).
Facts and paired replacement moves preserve the old inlining decision instead
of rewriting its history. MCTS-Mem tools are pinned locally to upstream
`6159557c361e4b2ac87152a143b3069db5428ce1` (0.2.2); the Codex build/use skills
are installed. Run `npx mcts-mem@0.2.2 lint mcts_mem` to validate the tree.

The selected source already passed ARM and x64/Rosetta core tests,
unguarded call/memory tests, and 260 JIT spec files, with zero audited compiler
warnings. Native diagnostics also exercised both architectures. Final branch
local validation and CI are recorded separately; historical passing revisions
are not substitutes for any newly failing job.

Routine PR/main/dev CI remains the existing Nano differential gate. The
V8/Cranelift standings workflow is manual-only. Temporary profiling workflows
are not included in this PR. Repository/API/crates.io publication remains a
separate later stage; this change does not publish a release.

## PR CI follow-up

The initial PR correctness run failed. [Investigation and repairs](PR_CI_AUDIT.md)
record the reproduced workspace-container assertions, baseline nightly
dependency warnings and RV32 linker inputs. The later local workspace test
result is 683 passed / 4 ignored, with no compiler warnings. Historical
selected-source equality and test results above describe the initial PR;
subsequent test and CI/toolchain repairs do not change the JIT optimization
policy. The user-approved standalone spectest fix from PR #41 also enables
the shared WAST runner with either engine independently and uses pure interp
for interpreter spec execution in CI. Remote final validation remains pending.
