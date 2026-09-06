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

### Frame ALU and loop cache measurements

[Run 34033234473](https://github.com/mbbill/Silverfir-nano/actions/runs/34033234473)
finished both native jobs without failed steps or compiler warnings. Four
alternating process pairs; all source hashes and samples are retained in
`coremark-loops-draw-*`.

| Comparison | EPYC 7763 | EPYC 9V74 |
|---|---:|---:|
| corrected candidate / main | +1.91% | +1.83% |
| frame ALU / corrected candidate | -0.44% | -0.93% |
| loop cache / frame ALU | +0.87% | +0.82% |
| loop cache + frame ALU / main | +2.35% | +1.71% |

The memory-operand candidate loses on both hosts; `431b2062` removes it.
Loop caching helps relative to its immediate parent. These main-relative
numbers replace, rather than add to, earlier main-relative measurements.
The full dev run for `f6362a74` is still being audited.

### Next isolated candidates

`b54ff280` preserves i32 zero-flag proofs across plain register stores and
uses them for equality/inequality branches against zero. Proof capture is
after operand materialization. Ordering, i64 comparisons and unknown CFG
entries retain explicit comparisons. Actual CoreMark matrix code drops one
TEST after its decrement/store pair. Arithmetic/store integration tests cover
wrapping, zero literals, signed/unsigned ordering, and both integer widths.

`886d1d57` clears the XMM destination before every scalar integer-to-float
conversion. The source is a GP value; no upper scalar carrier lane is
observable. This breaks the otherwise unnecessary previous-destination
dependency without changing conversion rounding. The spectralnorm native
loop reused the same XMM register for conversion and the previous division.
This is a performance hypothesis pending native paired measurement. A
corresponding independent compiler example is in the
[Dart compiler source](https://dart.googlesource.com/sdk/+/dbe5496ade6003efaebf22a660582c1bbaf05b59%5E1..dbe5496ade6003efaebf22a660582c1bbaf05b59/).

Validation includes 524 default x64 unit tests, the complete core integration
suite, all 260 x64 specification files, and the lint policy. The isolated
Nano-only workflow compares loop/flags/ALU-removal/float revisions and
spectralnorm at the suite's original 500-element input with its exact
`1.2742241159529095` result oracle. No competitor engine is rebuilt.

### Native scalar-conversion result

[Run 34034022693](https://github.com/mbbill/Silverfir-nano/actions/runs/34034022693)
completed on Intel 8573C (draw 1) and AMD 7763 (draw 2). Spectralnorm compares
`886d1d57` directly with its parent `431b2062`, six alternating process pairs:
**+225.95%** throughput on Intel (18.08 to 58.92 runs/s), **+0.36%** on AMD
(57.39 to 57.59 runs/s, inconclusive). Every process checked the suite's exact
output. This supports the false-dependency hypothesis for the Intel gap.
It is an isolated Nano revision comparison, not a new competitor ranking.

CoreMark in the same experiment: flags/loop is +0.13% Intel, +1.20% AMD;
removing frame ALU on top of flags is -0.42% / -0.50%, both inconclusive;
float conversion on top of that is +0.28% / +0.12%, also inconclusive.
Instruction interactions mean the ALU-removal result does not simply invert
its earlier isolated measurement. All samples and CPU identities are in
`dependency-draw-*`.

### Full loop-candidate dev findings remain action-required

[Dev run 34033253457](https://github.com/mbbill/Silverfir-nano/actions/runs/34033253457)
measures `f6362a74` (runtime `9a820828`) against main. Full wasmi execute
geomeans are +10.1297% on AMD 9V45 and +4.4904% on ARM Neoverse N2. All 20
rows are retained in `loop-dev-{x64,arm64}-comparison.json`; tiny_keccak's
x64 -3.55% PLACEMENT observation remains visible. This new CPU model has no
competitor anchor, so it cannot establish a ranking against V8/Cranelift.

Windows LZ4 compression is a confirmed regression: primary -12.25%,
independent confirmation -11.89%. The cross-run verdict step failed.
The x64 JIT startup primary also flags bz2 -3.24%, spidermonkey -1.60%,
ffmpeg -2.61%; ARM64 flags bz2 -1.91% and CoreMark -1.67%. Independent confirmation reproduces x64 ffmpeg at -3.63% and ARM64
CoreMark at -1.66%; both cross-run verdict steps failed. All 34 jobs are
finished, with these two failures plus Windows LZ4. This run is not a pass.

### Pending dead-frame-store elimination (`dc549134`)

Delete an exact native frame-word store only when another store overwrites
it in the same block before any read, potentially trapping operation, call,
frame-base definition or opaque operation. The final store remains. This
simple linear pass tracks one candidate and runs after store forwarding and
instruction selection. Native CoreMark CRC code has one rather than six
stores to its scratch slot. No benchmark or guest function name participates.

Validation: 527 x64 units, 561 ARM64 units, both complete core integration
suites, all 260 x64 spec files, and CoreMark's own result validation. Native
measurement is pending, including a separate variant with loop caching off
to quantify its benefit and its possible role in the CI regressions.


### Frame stores and loop caching measured independently

[Run 34035026753](https://github.com/mbbill/Silverfir-nano/actions/runs/34035026753)
compares main, float conversion (`886d1d57`), dead stores (`dc549134`), and
that revision with loop caching removed. Both jobs finish without failed
steps or compiler warnings. Four alternating process pairs:

| Comparison | EPYC 7763 | Xeon 6973P-C |
|---|---:|---:|
| float / main | +2.91% | +6.92% |
| dead stores / main | +2.90% | +8.05% |
| dead stores / float | -0.01% | +1.06% |
| no loop cache / dead stores | -0.41% | -5.16% |

Dead-store removal is neutral on AMD and inconclusive on this Intel draw
(P(improvement) 77.43%). Disabling loop caching loses performance in both
samples, with P(regression) 96.39% / 97.86%; this remains diagnostic evidence,
not the stronger full dev gate. Keep the cache pending its faster analysis
and full-corpus results. Samples, CPU identities and the exact no-cache patch
are in `dse-{1,2}-*`.

### Windows LZ4 isolation

[Run 34034965927](https://github.com/mbbill/Silverfir-nano/actions/runs/34034965927)
retains a failed regression step in its introduction job. Measurements use
the existing paired performance harness and unchanged guest validation:

| Baseline → candidate | CPU | Compression | Decompression |
|---|---|---:|---:|
| corrected `35e315c6` → loop `9a820828` | EPYC 9V74 | -17.97% REGRESSION | +10.29% IMPROVEMENT |
| loop → flags + ALU removal `431b2062` | EPYC 7763 | +14.59% IMPROVEMENT | +0.89% PASS |
| `431b2062` → same, loop cache removed | EPYC 7763 | +0.12% PASS | +0.77% RECOVERED |

The recovery measures flags and ALU removal together; it does not attribute
the recovery solely to ALU removal. Different CPU draws cannot be chained
into a precise recovered percentage. Loop-cache removal alone does not help
compression here. All metrics, source identities and patches are in
`lz4-{introduction,recovery,nocache}-*`.

The preceding audit run 34034632719 failed before measurement because the
root lint scan saw deliberately invalid lint-policy test fixtures inside
nested source checkouts. The workflow now audits its root before those
checkouts and audits each source from its own root. No lint exception or
suppression was added.

### Current unmeasured candidates

`3267e4de` computes register-mention masks once per block during loop-cache
analysis; high register IDs retain the original exact scan. Masks are updated
after rewrites. The optimization targets JIT compile-time overhead without
changing which registers a loop can cache. `39ba77b3` lowers sufficiently
small, duplicate-heavy jump tables to exact unsigned range/bit-membership
branches, preserving complete edges and arguments. Both generic forms have
unit coverage; jump-table integration tests also exercise wrapped 64-bit
indices and out-of-range values. Full dev run
[34035799241](https://github.com/mbbill/Silverfir-nano/actions/runs/34035799241)
is measuring their combined revision with the earlier fixes.

`fca23555` reuses an existing right-hand result lane for scalar float ADD/MUL,
avoiding two MOVAPS in affected nbody loops. SUB/DIV retain ordered operands.
The new integration test covers both float widths, either live input, signed
zeros, subnormals, infinities and quiet/signaling NaNs. Default x64 validation:
531 unit tests and the full integration suite, without compiler warnings;
ARM64 passes the new semantic test. The x64 spec runner completes with exit 0.
The private CoreMark/nbody experiment compares this change to its exact
parent and captures native samples after timing. No performance win is yet
claimed for these candidates.

### Existing JIT-only unit-test warnings (left unresolved)

`cargo +1.98.1 test -p sf-nano-core --no-default-features --features
jit,guard-pages --target x86_64-apple-darwin --lib --no-run` emits two
dead-code warnings in `op_decoder.rs`: `Decoder::predecode_fast_disabled`
and `disable_predecode_fast_for_test`. The field and helper serve the
interpreter predecoder's differential tests. The same warnings reproduce on
main `f73219f5` in the isolated diagnostic checkout; default features do not
warn. This configuration is an action-required warning audit despite the
unit assertions succeeding. It implicates shared-decoder/interpreter test
ownership; it has not been patched with cfg gates or lint suppressions.


### Current full wasmi primary results (`39ba77b3`)

Run 34035799241 finishes the full 20-item execution primary at +22.0269%
throughput geomean on Intel 8573C and +4.7056% on ARM Neoverse N2, including
all negative metrics. Intel spectralnorm is +221.98%, sort +42.24%, Argon2
+233.51%, reverse_complement +101.58%, fibonacci-tail +71.09%. Intel
tiny_keccak remains -6.86% PLACEMENT and bulk-ops -1.86% PASS. These are
Nano-versus-main results; the older competitor anchor gives a useful
projection but cannot establish a new same-host ranking.

The x64 JIT startup primary on AMD 7763 flags bz2 -3.89%, pulldown-cmark
-3.03%, spidermonkey -2.63%, ffmpeg -3.50%. The register-mask change has not
removed the startup regression. Windows LZ4 primary still flags -5.83%.
Independent confirmation has now completed (see below); these failures are
not waived by the execution gains. All execution rows and x64 startup samples are in
`current-dev-*-comparison.json` with their original summaries.


### Completed current dev audit and startup attribution

All 34 jobs of run 34035799241 completed. Both startup confirmation verdicts
failed: AMD 7763 retains bz2 -4.20%, spidermonkey -2.22%, ffmpeg -2.57%
REGRESSION; pulldown-cmark -1.94% is NEGLIGIBLE. ARM Neoverse N2 retains
bz2 -1.82%, ffmpeg -1.74%, CoreMark -1.51%, Argon2 -1.73% REGRESSION.
The original startup and confirmation logs and all final step conclusions
are preserved in `current-dev-*-startup*.log` and `current-dev-final-jobs.json`.
Windows LZ4 confirmation reports compression -2.89% NEGLIGIBLE, which does
not erase the primary -5.83% regression. This dev revision is not a pass.

The separate Nano-only startup isolation
[run 34037420472](https://github.com/mbbill/Silverfir-nano/actions/runs/34037420472)
uses identical single-thread, non-debug JIT probes, alternating four rounds
on AMD 7763. Removing loop caching from `39ba77b3` improves startup throughput
by 2.61% on CoreMark, 2.34% on bz2, and 1.38% on ffmpeg. Removing only frame
DSE instead changes them by -0.50%, +1.29%, +0.34%. These diagnostic estimates
identify loop-cache analysis as the larger contributor; they do not replace
the failing full startup gate. The exact ablation patches, source revisions,
all samples and compiler profiles are in `startup-isolation-*`.

### Native duplicate-table and scalar float results

[Run 34036671158](https://github.com/mbbill/Silverfir-nano/actions/runs/34036671158)
completed both draws without failed steps or compiler warnings. CoreMark
uses four alternating process rounds and the unchanged official module;
nbody uses six rounds at the suite's 400-body input. `table-draw-{1,2}-*`
preserves every sample, source identity and the post-measurement profiles.

| Change | AMD 7763 | AMD 9V74 |
|---|---:|---:|
| Direct tables / DSE parent, CoreMark | +1.69% | +0.12% |
| Float RHS reuse / tables, CoreMark | -0.18% | +0.02% |
| Float RHS reuse / tables, nbody | +1.70% | +1.62% |
| Float RHS reuse / main, CoreMark | +4.59% | +3.49% |

The table improvement has P(improvement) 99.94% on 7763 and 93.47% on 9V74;
the latter is inconclusive. The two nbody improvements have probabilities
above 99.98%; retain the commutative float change on that evidence.
CoreMark's -0.18% float result is retained as inconclusive, not hidden.
These are Nano-only gains; the last competitor anchor still leaves a larger
CoreMark gap to Cranelift, and no new No. 1 ranking is claimed.

### Next bounded candidates

`167bc37c` extends frame caching to single-read mutable loop recurrences,
while retaining stores and rejecting partial-slot aliases. `1401981a`
retains valid x64 zero-flag proofs across plain register MOVs when the
producing value remains intact. `ef0c12ef` makes loop discovery compact and
register masks lazy, and rejects register-saturated loops before frame
analysis. Unit coverage verifies natural-loop membership, duplicate latches,
word boundaries and stable ascending node order. CoreMark's matrix loop
now carries its counter in a register and branches directly after its
store/copy without reloading or repeating TEST. The native experiment
34038942517 and startup isolation 34038942504 are pending.

`523712c4` forwards or reconstructs a native frame word across a unique
predecessor using unchanged explicit edge bindings. Only cheap integer
operations are reconstructed; published stores stay, and calls, unknown
writes, frame-base changes and ambiguous edges prevent the rewrite. This
removes a reload dependency in CoreMark's state scanner without any guest
or function-name check. Default x64 (536 unit tests plus integrations) and
ARM64 (567 plus integrations), then 260 x64 spec files, pass without compiler
warnings. A subsequent additional proof test passes as well. Native speed
and startup cost remain unmeasured, so this is still an experimental candidate.
