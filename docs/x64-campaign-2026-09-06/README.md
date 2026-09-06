# JIT performance campaign

## Goal and order

1. Lead V8 and Cranelift on the complete pinned wasmi execution corpus and
   its own CoreMark module on native x64; preserve ARM64 performance and
   correctness. Keep all per-workload results visible.
2. After the JIT goal is met, optimize interpreter startup.
3. Then optimize interpreter execution. No benchmark-specific fusion,
   names, input values, or result shortcuts; general semantic optimizations
   are allowed. WASI specialization and crates.io publication are deferred.

Normal PR/main CI stays unchanged. Competitor engines are built only for
anchors and the closing comparison. Intermediate measurements compare Nano
revisions. A dev soft-fail or failed warning audit remains a real failure.

## Measured candidate: LEA (`53f6bc69`)

[Run 34028103466](https://github.com/mbbill/Silverfir-nano/actions/runs/34028103466)
compared against `main@f73219f5`. All 34 jobs completed with no failed steps;
no confirmation job reported a regression. The x64 wasmi execute runner
was an AMD EPYC 7763. The full 20-row result is preserved in
`lea-wasmi-execute.md` and `lea-wasmi-execute.json`.

The complete-corpus geometric throughput change is +3.2018%. Tail Fibonacci
is +71.03%; prime sieve +2.22%; matrix multiply +1.75%. The native loop dump
shows the expected replacement of two copy/arithmetic pairs by two LEAs.
This is a revision improvement, not a claim that the JIT goal is complete.

The separate WASI CoreMark differential was -1.29% on Linux and -2.08% on
Windows (both classified NEGLIGIBLE by that harness). Those negative point
estimates are retained as a reason to measure the exact wasmi CoreMark
module independently, rather than assuming a uniform benefit.

## Generic byte-copy recovery (`17f09149`)

[Native profile 34028220277](https://github.com/mbbill/Silverfir-nano/actions/runs/34028220277),
on AMD EPYC 9V74, found 70.14% of Argon2 samples and 27.31% of sort samples
in the same direction-selected byte-copy loop. The existing memmove pass
missed a frame-spilled step and a directly expressed inequality.

The candidate extends the existing semantic proof to these forms. It proves
the two directions, index update and exact load/store pair, rejects hidden
side effects and destructive aliases, and adds widened endpoint guards.
Only ranges inside both the 32-bit address domain and current memory use
`MemoryCopy`; other inputs retain the original loop and its trap/partial-write
behaviour. It adds no engine instruction and uses no module name or input
constant to identify a workload.

Local validation: 520 x64 unit tests, the core integration suite, 554 ARM64
unit tests, overlap/wrapping/partial-trap tests on both hosts, and all 260
x64 specification files. The actual integration fixture was also confirmed
to contain the recovered `memory.copy` in its native dump.

[Dev run 34029646555](https://github.com/mbbill/Silverfir-nano/actions/runs/34029646555)
is the required native differential measurement; no performance result is
claimed until that run and its confirmations have been inspected.

## Pending candidate: select flags (`79809a89`)

Reuse an immediately preceding i32 result's ZF for a select when no operand
materialization has invalidated it. Clear the position-based proof at every
basic-block entry, because other predecessors need not supply those flags.
All 260 x64 spec files passed, including new register, zero-immediate, float,
and join-path cases. Published only to `codex/x64-select-flags` for an isolated
Nano-only CoreMark experiment while the preceding dev run completes.

### Native findings still requiring correction

The first `17f09149` wasmi x64 execute comparison (AMD EPYC 9V74) shows a
complete-20 geometric throughput improvement of +13.8455% against main:
Argon2 +232.59%, sort +36.53%, reverse complement +91.98%. Fibonacci-iter
(-4.94%) and regex-redux (-4.99%) are REGRESSION rows with independent CI
confirmation pending. Full rows and samples are in `bulk-wasmi-execute.*`.
The unchanged ARM64 target also benefits in Argon2 (+151.22%).

[Nano-only CoreMark run 34030092675](https://github.com/mbbill/Silverfir-nano/actions/runs/34030092675)
used four alternating process rounds per host. Against main, the LEA+bulk
revision regressed -6.92% on EPYC 7763 and -8.22% on EPYC 9V74. Select-flag
reuse improves this candidate by +1.15% and +1.18%, respectively, but remains
below main. These are real negative observations; the primary wasmi corpus
excludes the dedicated CoreMark score and cannot clear them. Full logs,
revision IDs, CPU details, and samples are retained in `coremark-flags-draw-*`.

CoreMark native dumps have the same block structure before and after bulk
recovery; that pass does not rewrite this module. The regression is being
isolated between register-sum and immediate-offset LEA forms in
[run 34030690378](https://github.com/mbbill/Silverfir-nano/actions/runs/34030690378),
along with same-host profiles of main and the combined candidate. The goal
is not complete, and none of these pending candidates has been merged.

The independent confirmation runner was an Intel Xeon Platinum 8573C:
Fibonacci-iter +0.04%, regex-redux -0.13%, both PASS. All 34 workflow jobs
have no failed steps, but this different CPU does not explain or erase the
AMD 9V74 negative measurements, and the separate CoreMark regression remains
unresolved. Retain these as hardware-sensitive regressions to investigate.

## Pending frame-spill copy candidate (`0ebcc52b`)

Preserve an exact native-width GP frame word in the reload destination
before the immediately intervening destructive arithmetic, when that
register is neither read nor written by the arithmetic. The frame store
remains for traps and subsequent blocks. The real CoreMark native dump
confirms six frame reloads in the CRC function became register copies.
This is ordinary local store/load scheduling and has no workload matcher.

Validation: 390 JIT-only x64 unit tests, 556 default ARM64 unit tests, all
260 x64 spec files, native-width and alias rejection tests, and the CoreMark
module's result validation in the local x64 probe. Local timings under
Rosetta are not performance evidence; native CI measurement is pending.

### LEA regression isolated by operand form

The four-revision experiment completed on AMD EPYC 7763 and Intel Xeon
Platinum 8573C. Compared with main, register-sum-only LEA is -0.39% / +0.59%;
immediate-offset-only LEA is -6.44% / -3.08%; both forms are -6.52% / -1.02%
(AMD / Intel). All four process pairs, patches and native profiles are in
`coremark-lea-draw-*`. The AMD profile concentrates the added samples in
an unchanged state-machine block, so layout sensitivity remains a plausible
mechanism; the instruction-form experiment establishes the revision cause,
not a definitive microarchitectural cause.

The next candidate restores two-operand immediate i32 ADD/SUB globally,
while retaining i64 LEA and register-sum LEA. CoreMark uses the affected i32
form; the full corpus is needed to validate retaining the separately useful
i64 form. No function name, guest input or benchmark name selects the path.

### Corrected candidate: CoreMark and full-corpus measurements

[Nano CoreMark run 34031452780](https://github.com/mbbill/Silverfir-nano/actions/runs/34031452780)
compares main with the i64-only immediate-LEA/flags combination and with the
additional spill-copy pass. Four alternating process pairs per host:

| Comparison | EPYC 7763 | EPYC 9V74 |
|---|---:|---:|
| corrected LEA + flags / main | +0.75% | +1.74% |
| plus spill copies / main | +2.03% | +1.75% |
| spill copies / corrected LEA + flags | +1.27% | +0.01% |

The large CoreMark regression is eliminated on both AMD generations. The
spill-copy change has a measurable benefit on 7763 and is neutral on 9V74.
These revision gains do not yet close the competitor gap. Full measurements
are in `coremark-spill-draw-*`.

[Dev run 34031425369](https://github.com/mbbill/Silverfir-nano/actions/runs/34031425369)
compares `35e315c6` (runtime `1f904642`) with main. The complete 20-item
wasmi x64 execute comparison is +15.0038% on Intel Xeon Platinum 8573C,
with no REGRESSION rows. All rows and samples are in
`corrected-wasmi-execute.*`; tiny_keccak's -6.56% PLACEMENT result is retained
as a negative layout-sensitive observation. Other platform checks and the
independent confirmation job must still be inspected individually.

The full `35e315c6` dev run has 34 completed jobs with no failed steps.
However, the secondary WASI suite flagged lz4-compress at -7.46% on Intel
8573C; its independent confirmation landed on AMD 7763 and measured -2.48%
(NEGLIGIBLE under that gate). This does not explain the Intel regression.
Keep the negative finding open even though the workflow conclusion is success;
WASI optimization remains deferred behind the primary wasmi/CoreMark target.

## Pending frame ALU and loop-frame candidates

`a9ed0167` extends the existing x64 load/ALU encoding to consume the low
32 bits of an aligned native frame word directly. Guest-memory widths remain
exact; both frame widths touch the same page. Native CoreMark code confirms
one loop's frame-load/add pair became a single memory-operand add. Validation:
391 JIT-only x64 unit tests and all 260 x64 specification files.

`9a820828` carries a repeatedly read native frame word in an allocatable GP
lane unused throughout a natural loop. It retains every frame store and
updates the carried copy immediately afterward. It rejects calls, opaque
operations, overlapping partial writes, writes to the frame base, live lanes,
ambiguous entry edges, and function-entry loops. It runs after dead-parameter
elimination so obsolete cached-local bindings cannot hide a free lane.
CoreMark's state loop now carries its cursor through a previously unused
register, replacing two repeated frame reads. Argon2 still contains the
recovered `memory.copy` and passes its output oracle.

Validation: 525 default x64 unit tests and the complete core integration
suite, 558 ARM64 unit tests and core integration suite, all 260 x64 spec
files, and CoreMark's guest result validation. Native performance is pending;
local Rosetta timings are not evidence of performance gains.
