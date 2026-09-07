> Historical experiment log. Entries describe the source and evidence available at that time; later entries and README.md supersede preliminary/pending decisions. Paths to omitted raw local artifacts are historical provenance, not packaged files. Published CI run links and the curated JSON result tables accompany this record.

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


### Withdrawn recurrence and cross-edge candidates

The recurrence experiment
[34038942517](https://github.com/mbbill/Silverfir-nano/actions/runs/34038942517)
completed both draws. On AMD 9V45, single-read caching versus the float parent
was -1.03%; retaining MOV flag proofs recovered +1.20% against that candidate,
leaving the combined revision only +0.22% versus main. On AMD 9V74 the two
increments were -0.23% and +0.04%, with combined +3.80% versus main.
The single-read extension has no repeatable net execution benefit and adds
startup work. It was withdrawn in `68357c76`; repeated-read loop caching and
the general MOV flag proof remain. `recurrence-draw-{1,2}-*` includes every
sample and native profiles.

The cross-edge rematerialization experiment
[34039368075](https://github.com/mbbill/Silverfir-nano/actions/runs/34039368075)
also completed both draws. Incremental CoreMark changes were +0.09% on Intel
6973P-C and -0.37% on AMD 7763, both inconclusive. Despite removing the intended
reload, this candidate has no demonstrated runtime win; `0878b63e` removes it.
The implementation remains in git history for inspection, not in production.
`remat-draw-{1,2}-*` preserves all samples, identities and native profiles.

### Startup costs after analysis cleanup

All following rows use same-host Nano pairs; each row includes all three
measured workloads, with six alternating process rounds. They are diagnostic
estimates, not a replacement for the failed full dev startup gate.

| Run and exact pair | CPU | CoreMark | bz2 | ffmpeg |
|---|---|---:|---:|---:|
| 34038942504, `ef0c12ef` / `1401981a` | Intel 8370C | +1.05% | +2.35% | +0.73% |
| 34039368128, remat / `ef0c12ef` | AMD 7763 | +0.75% | +1.00% | +0.29% |
| 34039990108, lean `0878b63e` / current `39ba77b3` | AMD 7763 | +0.07% | +0.62% | +0.53% |
| 34039990108, lean / main | AMD 7763 | -2.84% | -2.57% | -2.30% |

The latest cleanup remains slower than main on all three startup workloads.
It is not a pass. Do not infer precise cost attribution from profiles across
different runners or from probabilities below a full gate's decision level.
`loop-startup-*`, `remat-startup-*`, and `lean-startup-*` contain the complete
samples and compiler profiles.

`d70a230c` avoids rebuilding or reoptimizing blocks whose only change is a
new loop-carried parameter. `27cd41f5` caches call/opaque/frame-base barriers
alongside register summaries, rejecting impossible loops before entry-edge
and frame-slot analysis. Each analysis rewrite was independently checked on
the complete 20-module execution corpus plus CoreMark: every final Machine IR
function is identical to its immediate runtime baseline. Both default x64 and
ARM64 unit suites pass. Native startup measurement rejected the second change; see the completed
barrier result below. It was withdrawn in `aecdb9f1`.

### General demanded-bit candidate

`ff5c13ee` runs backward bit-demand propagation only in blocks containing an
extraction mask, then uses known-zero bits to simplify redundant extraction
operations. It preserves every escaping register and treats unknown operations
as barriers. Published stores stay. It contains no benchmark, function, CRC
polynomial, or fixed iteration-count checks. The retained CoreMark CRC block
shrinks to 659 native bytes, removing redundant AND operations after shifts.

Randomized instruction-sequence comparisons check all escaping lanes with
both scalar widths, either select arm, shifted logical operands and both
zero- and sign-extending i32 target models. Separate negative proofs cover
published words, runtime barriers and escaping high bits. Default x64 passes
534 units plus integrations, ARM64 565 plus integrations, and all 260 x64
spec files pass after the final scheduling guard, without compiler warnings.
CoreMark also validates locally; Rosetta timing is not performance evidence.
The Nano-only execution run [34040589880](https://github.com/mbbill/Silverfir-nano/actions/runs/34040589880)
completed: bits/lean improves CoreMark by +1.45% on Intel 8573C
(P improvement 99.948469%) and +1.02% on AMD 7763 (99.975924%).
Bits/main is +9.52% and +5.91%, respectively. All samples are in
`bits-draw-{1,2}-comparison.json`; this is an incremental win, not a new
competitor comparison or proof of overall No. 1.

Startup run [34040589858](https://github.com/mbbill/Silverfir-nano/actions/runs/34040589858)
on AMD 7763 reports bits/lean +0.26% CoreMark, -0.38% bz2, -0.56% ffmpeg.
Bits/main remains -2.51%, -3.00%, -2.80%. The execution benefit has a small
measured compilation cost; the existing startup failure remains unresolved.

The early barrier-summary run [34040892017](https://github.com/mbbill/Silverfir-nano/actions/runs/34040892017),
also AMD 7763, reports barrier/bits -0.57% CoreMark, -0.11% bz2, -0.46% ffmpeg.
The ffmpeg regression has P regression 99.924158%. This implementation was
withdrawn in `aecdb9f1`; identical output alone is not enough to retain it.
`bits-startup-*` and `barrier-startup-*` preserve samples and CPU identities.

### Native BEXTR and compilation-buffer experiments

`1e37ec04` tries generic unsigned bit extraction with BMI1 on AMD x64.
Intel and CPUs without BMI1 retain the existing scalar fallback. Width,
register encoding, live inputs, zero-length and out-of-range control cases
are checked. Default x64 passes 536 units plus integrations; ARM64 passes
565 plus integrations with no compiler warnings. Rosetta exposes no BMI1,
so its raw execution test explicitly reports a skip. Native execution and
suite validation are in run 34042092630; full dev comparison is 34042099249.
Neither run is yet a passing conclusion.

`9b6032eb` reuses constant-analysis arrays across blocks and skips dead-store
scans when there are fewer than two stores. The arrays are cleared before
all block analyses. The full fixed corpus plus CoreMark has 1285 identical
final MachineIR functions in 21 modules before and after this change; default
x64 and ARM64 tests pass. Native startup performance is pending.


### BEXTR result and warning-audit repair

Run 34042092630 completed native tests (537 core units plus integrations,
260/260 specification files) on Intel 8573C and AMD 7763. Intel emits zero
BEXTRs and changes -0.02% versus bits; AMD emits seven and improves +1.69%
(P improvement 99.998465%). Versus main: +10.14% Intel, +7.44% AMD.
These remain Nano comparisons, not a new V8/Cranelift ranking.

The runner lacked ripgrep, so the old experimental shell warning checks did
not execute. This was a real audit failure despite GitHub's success result.
Every saved build and correctness log from both BEXTR draws was subsequently
checked locally with `ci.runner.parse_log`: zero compiler errors or warnings.
`ed3ca16a` replaces the experimental checks with that parser and tests missing,
empty and warning/error-containing logs. Its CoreMark rerun 34042822115 then
failed due to Python selecting the candidate checkout's `ci` package. The
workflow now changes to the diagnostics root before invoking the audit;
215cdf3c also starts the next short-branch experiment. No suppression added.

The broader diagnostics-worktree `ci.test_correctness` invocation also exposes
an existing workflow-inventory assertion: the private startup and LZ4 workflow
files are not in its expected inventory. It remains a recorded test failure;
the eight warning-audit/parser tests passed independently. These private
workflow files do not exist on the runtime candidate branch.

Full dev run 34042099249, candidate `1e37ec04`, completed all 20 execution
rows: AMD 7763 throughput geomean +12.837470%, ARM Neoverse N2 +4.747188%.
All rows are in `bextr-dev-{x64,arm}-execute-comparison.json`. AMD spectralnorm
is a real -4.45% regression; tiny_keccak is -4.22% with PLACEMENT status.
Windows LZ4 compression is -7.06% in the primary run. Startup and cross-run
confirmation are not final; this full run must not be described as passing.

`96019ff4` emits rel8 for nearby, already-bound branch targets without a
relaxation pass. Short/far boundaries are tested; default x64 passes 537 units
plus integrations and 260 spec files, with no compiler warnings. Native speed
is pending. The apparently unused RAX frame load in CoreMark's entry is the
guard-page stack probe and must be preserved.


### Retained short branches, scratch reuse and AMD conversion fix

Short-branch run 34043142316 used AMD 7763 in both independent draws.
Short/scratch is +0.29% (P improvement 99.295898%) and +0.24% (99.719216%);
short/main is +7.56% and +7.69%. Both native core/spec checks and the repaired
warning audits completed with no compiler diagnostics. Intel performance of
this small change remains unmeasured. `short-draw-*` contains every score.

The first scratch startup run 34042408383 (AMD 9V74) gives scratch/bextr
+0.38% CoreMark, +1.08% bz2, +0.45% ffmpeg. Its old ripgrep audit did not run;
its saved build logs were checked locally without diagnostics. The repaired
run 34042822129, also AMD 9V74, gives +0.78%, +1.62%, +0.61%, respectively.
The repaired audit reports zero errors/warnings for all three source builds.
Scratch/main remains -2.10%, -1.47%, -2.04%. Keep `9b6032eb` for repeated
compile-cost improvement, without claiming the measured slowdown is gone.

Spectralnorm run 34043512656 isolates exactly the CVT destination clear in
`98065afd`, a separate experimental branch based on `1e37ec04`. AMD 9V74:
noclear/current +4.34%, noclear/main -0.02%. AMD 7763: +5.76%, +3.18%.
Both use suite setup(500), verify output 1.2742241159529095, and have clean
compiler audits. The complete results are in `spectral-draw-*`.
`55a4de0c` keeps the Intel dependency break and omits it on AMD, sharing the
cached CPU cost preferences with BEXTR. Local x64 core tests pass.

Full dev 34042099249 is complete and **failed**: the AMD 9V74 confirmation
also reports spectralnorm -2.57%, after the AMD 7763 primary -4.45%.
The startup primary rows classify the remaining slowdowns as NEGLIGIBLE,
NOISY-FLOOR or RECOVERED, so their confirmation jobs performed no timing.
In particular x64 bz2 -3.09%, ffmpeg -2.79%, CoreMark -5.13% (NOISY-FLOOR)
and ARM bz2 -0.92%, ffmpeg -1.29%, CoreMark -1.80% remain visible data.
Windows LZ4 compression was -7.06% in the primary and +2.91% in confirmation;
the second host does not erase the primary result.
The combined CPU/scratch/short candidate `55a4de0c` is now in full dev run
34044124553. It does not contain the memory-update experiment.

`8088dce5` is a separate, still unmeasured x64 memory-update candidate. It
recognizes adjacent matching loads, immediate integer ALU operations and
stores whose transient is dead; it rejects modified address registers,
width/extension mismatches and inexact sign-extended i64 immediates. No lock
prefix or atomic operation is introduced. Local default x64 and ARM64 tests
and all 260 x64 spec files pass. Added tests cover all five operations, both
widths, escaping results, boundary immediates, neighbor-word preservation,
unaligned accesses and out-of-bounds traps. Native execution and startup
measurements are pending. No benchmark identifiers occur in the optimizer.

### CPU fix verified; memory updates withdrawn; entry guards under test

Full dev 34044124553 (`55a4de0c`) is complete and **failed**. AMD 7763
spectralnorm is now -0.05% (PASS), resolving the preceding confirmed AMD
regression. All 20 execution rows give +13.102322% throughput geomean on
AMD 7763 and +4.590098% on Neoverse N2. Raw complete rows are saved in
`tuned-dev-{x64exec,armexec}-comparison.json`. AMD tiny_keccak remains
-4.50% PLACEMENT and regex_redux -1.19% RECOVERED. Windows LZ4 compression
is -2.26% RECOVERED in this run; its confirmation job performed no timing.
Earlier primary failures remain evidence, rather than being overwritten.

The failing gate is x64 JIT startup/bz2: AMD 9V74 primary -3.09%, AMD 7763
independent confirmation -5.36%, both REGRESSION. All seven primary startup
rows and the confirmation are saved locally. Other negative startup point
estimates remain visible, including ffmpeg -2.68% in the x64 primary.

Memory-update run 34044608446 has native 541 unit tests, integrations and
260 spec files passing on both draws, with the repaired warning audits.
Intel 8573C updates/tuned is -0.59% (P regression 93.647088%); Intel 6973P-C
is -2.08% (97.132100%). The retained tuned/main results are +10.07% and
+16.12% on these two hosts. No AMD runtime conclusion follows from this run.
Startup 34044608424 on AMD 9V74 gives updates/tuned -0.68% CoreMark,
-0.24% bz2, -0.38% ffmpeg. This candidate has no demonstrated benefit and
is withdrawn by `97fe9a6b`; its source remains on its experimental branch.
All diagnostic score pairs and machine identities are in `updates-*`.

`19e82afe` limits hoisted-address edge updates to the already-proven loop
members and external entry predecessors. All 21 modules / 1,285 final
MachineIR functions are identical to the pre-change corpus. Local x64
537 unit tests and integrations pass without warnings. Native startup
34046329636 (AMD 7763, six rounds) gives edges/tuned +0.69% CoreMark,
+1.85% bz2, -0.08% ffmpeg; P improvement 83.535052%, 99.902563%,
11.528374%, respectively. Edges/main remains -2.19%, -1.14%, -2.78%.
The compile-cost improvement is retained provisionally for full dev checks;
`edge-startup-*` retains the entire diagnostic result and profiles.

`0fe6b680` adds a generic scalar early-return guard before the x64 body
frame. It requires an empty integer-comparison entry, identity edges,
a register or literal scalar return, an ordinary arm matching the layout's
fallthrough, and no CFG edge back to the entry. The ordinary arm retains
its frame/probe; the fast arm touches neither stack nor preserved lanes.
On the fixed corpus this changes only fibonacci-rec; it does not inspect
function names, test inputs, or recognize recursive arithmetic. Final
MachineIR remains identical for all 1,285 functions. Local x64 541 unit
tests and all integrations pass; the four new integration tests exercise
signed/unsigned widths and boundaries, recursive frame restoration,
constant result bits and guest-memory traps. All 260 x64 spec files pass,
without compiler warnings. ARM64 565 units and integrations passed with
the common pipeline change; the final added constant-return test has only
been rerun on x64. Native runtime and full dev measurements are pending.

Entry-guard run 34047095698 is complete with clean native correctness and
warning-audit steps. AMD 9V74 fibonacci-rec guard/tuned is +2.32%
(P improvement 99.998844%), guard/main +2.44%. Intel 6973P-C guard/tuned
is +8.09% (99.999998%), guard/main +5.84%. Every measured process uses
the fixed suite input 30 and validates the exact result 832040. CoreMark
guard/tuned is +0.03% on AMD and -0.50% on Intel, both inconclusive.
All score pairs, CPU identities and source refs are in `guard-draw-*`.
The final constant-return integration test also passed on ARM64.
Full dev 34047100663 is still running; no full-pass verdict yet.

`4a54090a` is the independent inline-leaf alignment candidate. Its explicit
instruction set excludes unknown operations, helpers, calls, tail calls,
division/remainder and explicit traps. It retains every preserved-register
save/pop and the Wasm frame probe, but omits native RSP alignment work when
no generated body instruction can call. Templates retain the existing shim.
The complete corpus still has identical final MIR (1,285 functions); body
preludes shrink across 19 modules, including CoreMark functions 8, 10 and 13.
Local x64 544 units, integrations and 260 spec files pass with no warnings.
Native execution 34047573958 and startup 34047573966 are pending. They compare
main / guard (`0fe6b680`) / leaf (`4a54090a`) on the diagnostics branch;
normal PR/main workflows remain unchanged.

Leaf-frame run 34047573958 is complete on two AMD 7763 hosts. CoreMark
leaf/guard is +0.07% (P improvement 75.233861%) and +0.21% (93.768152%);
the recursive function's code is unchanged and its estimates are +0.01%
and +0.25%, respectively. Both native correctness and warning audits pass.
Startup 34047573966, also AMD 7763, gives leaf/guard +0.00% CoreMark,
+0.20% bz2, -0.24% ffmpeg (P regression 99.721168%). These data do not
establish enough runtime benefit to retain the added classification work.
`daf55bda` withdraws the leaf-frame candidate; `leaf-*` retains all score
pairs and CPU identities. The experimental source branch remains available.

The diagnostics-only workflow-inventory test failure is now fixed by
`4b41ad14`: its explicit expected inventory includes the three added audit
workflows, and a new test restricts their triggers to manual dispatch or
the exact private diagnostics branch with path filters. All 22 tests in
`ci.test_correctness` pass. This change does not touch runtime candidates
or normal PR/main workflows; the prior broken inventory is recorded above.

### Guard full dev 34047100663; real startup failures remain

Candidate 0fe6b680 versus fixed main f73219f5. Full 20-case execution
geomean throughput improved 17.0870% on Intel Xeon 6973P-C and 4.6876%
on ARM Neoverse N2. All 20 rows are preserved in guard-dev-*-comparison.json;
the Intel host differs from the frozen Xeon 8573C competitor anchor, so this
is not a direct competitor standing. Fibonacci-rec +4.55% IMPROVEMENT on
Intel supports the separate native early-return experiment. Tiny-keccak
-5.51% is classified PLACEMENT and remains explicitly visible.

The workflow FAILED. x64 primary startup pulldown-cmark -3.04% and ffmpeg
-4.04% were confirmed on AMD 9V74 at -2.09% and -2.67%. ARM argon2
startup -2.29% was confirmed at -1.72%; CoreMark/erc20 confirmation was
NEGLIGIBLE (-1.32%/-0.91%). No soft-fail is treated as a pass.

### Context predecessor scan 5911a907, private 34048754672

Intel Xeon 6973P-C, six paired rounds, main/guard/context-scan. Reordering
the non-self-loop rejection before predecessor search changes no final MIR:
all 1,285 functions in 21 modules compare equal. Default-feature core tests
pass on x64 (541 units plus integrations) and ARM (565 plus integrations),
with zero compiler diagnostics. Native startup build warning audits execute
and pass. Scan/guard: CoreMark -0.29% (P improvement 34.172411%), bz2 +0.61%
(P 90.007153%), ffmpeg +1.19% (P 97.108033%). Evidence is suggestive but
not a strong confirmation; retain provisionally while testing the separate
empty-facts invalidation change. Scan/main remains -2.78%, +0.91%, -1.30%.

### Empty facts dbc9431b, private 34049719748

AMD EPYC 7763, six paired rounds, main/context-scan/empty-facts. All three
probe builds have zero errors and warnings, audited by the functioning
warning checker. Empty/parent CoreMark +0.89% (P improvement 93.956308%),
bz2 +0.04% (P 52.776741%), ffmpeg +0.45% (P 99.982072%).
Empty/main remains -1.86%, -0.63%, +0.24%. Retain provisionally; this
three-case isolation does not close the earlier full dev startup failures.
Final MIR is identical across all 1,285 functions in 21 modules.

### Final return stores 57ad635c: bounded semantics, limited measured gain

Native runtime 34049985651: Intel 8573C draw 1 deadstore/parent +0.38%
(P improvement 97.856792%), deadstore/main +10.85%. AMD 7763 draw 2
+0.03% (P 55.224506%), deadstore/main +7.55%. Both native correctness
audits execute: 545 Linux units plus integrations, 260 spec cases, zero
compiler diagnostics. This is not a demonstrated AMD execution win.
Startup 34049985668 AMD 7763: +0.21% CoreMark (P 66.104196%), -0.27%
bz2 (P 28.623349%), +0.41% ffmpeg (P 99.969474%). Final/main -1.90%,
+0.41%, +1.10%. Keep provisional pending the combined full dev gate.

### Concurrent trap-table race fixed separately at 4f21ed57

A full x64 integration run of the coalescing candidate failed in
entry_returns: trap_signal.rs lookup indexed 2 in a vector of length 2.
The signal reader borrowed the global vector without the writers' lock.
The same implementation is present in main f73219f5. A separate worktree
at the pre-coalescing parent 57ad635c plus only the new concurrency test
reproduces the issue: index 128 in a vector of length 1. The lookup now
locks while borrowing the table, copies out the error address, then unlocks
before returning. Guard faults originate in generated code, outside the
registration/teardown critical section; concurrent other-thread writers
are synchronized. Existing table tests now exercise the locked resolver.

Regression evidence: /tmp/sf-parent-trap-race-repro.log (fails),
/tmp/sf-fixed-trap-race-test.log (passes). Combined coalescing+fix full core
tests pass x64 550 units and ARM 574 plus all integrations, zero compiler
diagnostics. A retry alone was not used to dismiss the original failure.

### Destructive input coalescing e0e14479, measurement pending

Generic late physical-register lifetime coalescing, enabled by x64's
two-operand ALU cost preference. Only linear volatile GP inputs; a result
may occupy a volatile or preserved GP lane. A bounded 16-instruction proof
rejects live inputs, escaping temporaries, cached owners, fixed writes,
helpers/traps/memory, and CFG argument changes. Integer widths and opcode
semantics remain intact. Five proof tests include independent evaluation
with arbitrary high bits, aliased RHS, lifetime endpoints and preserved lanes.

Final full corpus changes seven functions in four modules: compression 1,
regex_redux 3, word_count 2, CoreMark 1. CoreMark function 10 native body
shrinks 655 to 601 bytes, MOV count 80 to 67; arithmetic counts unchanged.
No benchmark names, inputs, polynomial patterns or special fused opcodes
are recognized. Local x64 spec 260 cases pass before the trap-lock fix;
new native CI validates the complete final source before timing.

### Coalescing withdrawn at 82a7475a

Private runtime 34050697105 measures AMD 9V74 draw 1 -0.47%
(P regression 96.538272%) and AMD 7763 draw 2 -0.23%
(P regression 96.862437%) versus synchronized parent 4f21ed57.
Native tests/spec/audits execute and pass, but fewer MOVs did not improve
CoreMark. Private startup 34050697132 Intel 8573C: CoreMark +0.01%
(P improvement 51.207976%), bz2 -1.68% (P regression 98.847990%),
ffmpeg -1.51% (P regression 99.996211%). Thus e0e14479 is withdrawn,
including its backend cost flag, pass, scheduling and tests.
The independent trap-table fix remains. Full dev 34050735963 still tests
e0e14479; record its real outcomes when it completes, even though that
source has been superseded. Do not describe that run as current validation.

### Late dynamic reserve cache aa6b2ad7, unmeasured

Existing call-free loop frame caching can now choose a lowering-reserved
dynamic lane after lowering has finished, but only when the complete loop
and entry arguments do not mention that lane. Backend-owned scratch is
a disjoint register set. Every store remains published; calls/opaque ops
still reject the transformation and register-mask proofs include every
lowered scratch use. No SSA allocation budget, argument lane or ABI mapping
changes. This candidate excludes the withdrawn coalescing pass.

Local final x64 546 and ARM 570 units plus integrations pass without
compiler diagnostics. All 21 modules compile and dump successfully.
The combined pre-withdrawal variant also passed 260 x64 spec cases;
private native CI validates the exact final revision before timing.

### Broad reserve policy withdrawn; eligibility ordering retained for measurement

Runtime 34051383696: Intel 6973P-C draw 1 reserve/parent -8.03%
(P regression 99.989867%), AMD 9V74 draw 2 -0.66%
(P regression 95.375207%). Both native core/spec warning audits execute
and pass, but runtime regresses. Startup 34051383701 AMD 9V45:
CoreMark -1.58%, bz2 -2.55%, ffmpeg -2.11% versus parent; regression
probabilities 95.933572%, 99.300529%, 99.840716%.

aa6b2ad7 is withdrawn at 43b0ff93. d98d6e89's independent check reordering
remains: reject uncacheable loops before allocating whole-function
membership/predecessor scratch. Final source 43b0ff93 has exactly the
original 4f21ed57 register range and caching policy; all 1,285 final MIR
functions in 21 modules match that parent. Default-feature core tests
pass x64 545 and ARM 569 plus integrations, no compiler diagnostics.
Its isolated startup measurement is pending, no runtime gain is claimed.

### Superseded coalescing full dev 34050735963, results still incomplete

Full 20-case execution geomean throughput versus fixed main: AMD 7763
+13.291278%, ARM N2 +4.687051%. All rows saved in coalesce-dev-*-comparison.json.
AMD tiny-keccak -4.52% PLACEMENT remains visible; regex -0.90% NEGLIGIBLE.
This result contains withdrawn e0e14479 and is not the current revision's
validation. ARM Darwin JIT job 101533887298 failed artifact creation with
ENOTFOUND, although its timing rows did not classify a regression. That
infrastructure failure is real; the workflow cannot be called passing.
Startup jobs/any confirmations still need to be read when complete.


### Scan-only 43b0ff93: measured startup benefit

Private run 34052103467, job 101537432473, AMD EPYC 9V74. Six paired
rounds, candidate versus synchronized 4f21ed57 parent: CoreMark +1.58%
(P improvement 97.979460%), bz2 +1.23% (99.617525%), ffmpeg +1.03%
(99.999994%). Candidate versus fixed main: -1.72%, +0.45%, +0.45%.
All probe builds pass the actual warning audit. Retain the check reorder;
there are no generated-MIR changes. Evidence in scan-startup-*.

### Superseded coalescing full dev startup completed

34050735963 is complete but failed ARM Darwin JIT artifact creation
(ENOTFOUND); no overall passing claim. x64 startup AMD 7763 has no
REGRESSION rows: bz2 -0.75% RECOVERED, pulldown-cmark -1.83% NEGLIGIBLE,
spidermonkey -1.33% NEGLIGIBLE, ffmpeg -0.71% RECOVERED. ARM N2:
CoreMark -1.62%, argon2 -1.34%, erc20 -1.41%, all NEGLIGIBLE. Startup
confirmation jobs had no REGRESSION metrics to rerun. All startup rows
are preserved in coalesce-dev-{x64,arm}startup-comparison.json. These
results include withdrawn coalescing and do not validate current HEAD.

### Single-block branch recurrence 8b0d6f4c, unmeasured

A general cost rule admits one static native-frame read when the last
full-word store directly supplies the boolean controlling a self backedge.
Only single-block natural loops qualify; passive single reads still require
two uses. The matching recurrence may use a dynamic lowering reserve only
when all loop mentions and entry arguments prove it free. Transparency,
alias, entry and write-through rules remain intact; no guest access moves.
CoreMark function 5 block 18 carries the counter in r10 (physical R15),
replacing an outer passive cache which formerly occupied that lane. The
lowering reserve is not the lane actually selected for this example.

Local x64 547 and ARM 571 units plus integrations pass without compiler
diagnostics, as do 260 x64 spec cases. Ten functions in five corpus modules
change; 21 modules / 1,285 functions compile and dump. Native execution
34052713398 and startup 34052713429 compare main / scan-only 43b0ff93 /
recurrence 8b0d6f4c. No measured benefit is claimed yet.


### Recurrence 8b0d6f4c withdrawn at 91b6dd12

Native 34052713398 draw 1 recurrence/parent +0.21% (P improvement
71.973328%), draw 2 -0.06% (31.245248%). No reliable gain; both native
548-unit core tests, integrations, 260 spec cases and warning audits pass.
Private startup 34052713429 Intel 6973P-C: CoreMark -2.18% (P regression
89.527993%), bz2 +0.37% (P improvement 64.002350%), ffmpeg -0.68%
(P regression 83.925150%). These are noisy, not demonstrated regressions,
but provide no reason to retain the more complex recurrence policy.
Revert includes its reserve exception, branch-word probe and both tests.
The scan-only optimization is retained.

### Dead address copies 2ec7af68, final isolated source 91b6dd12

Native x64 pair lowering substitutes a GP source for a dead linear snapshot
immediately preceding an indexed U8/U16 load. It preserves all index/offset/
extension fields and refuses cached destinations, fixed/FP lanes, or a copy
live after the load (including CFG arguments). Shared MachineIR remains
unchanged. Narrow widths avoid competing with existing U32/U64 load+ALU
fusion. Seven functions in four modules lose eight native MOVs; CoreMark
function 8 block 25 loses its address snapshot MOV, retaining explicit
zero-extension. Two proof tests cover aliasing/high bits/displacements and
liveness; a runtime test covers signed/unsigned narrow reads, OOB after
partial writes and repeated invocation after traps.

Pre-withdrawal combined tests pass x64 549, ARM 571 plus integrations and
260 x64 spec cases without compiler diagnostics. Final source removes
recurrence independently; private CI validates that exact revision before
measuring. Runtime benefit is unmeasured.


### Address-copy 91b6dd12: provisional, no conclusive execution win

Native 34053310774 both draws AMD EPYC 7763: address/parent +0.16%
(P improvement 68.529822%) and +0.27% (83.838087%). Main-relative +7.80%
and +6.96%; never compare absolute rates across runners. Native core/spec
and warning audits execute without failures. Startup 34053310793 AMD 9V74:
CoreMark -0.62% (P regression 95.048079%), bz2 +0.05% (P improvement
61.771583%), ffmpeg +0.09% (71.284169%). Direction is mildly positive in
runtime but evidence is insufficient; keep provisional as the immediate
parent for isolated narrow-equality measurement, not as a confirmed win.
Final source tests x64 547 and ARM 569 plus integrations, no diagnostics.

### Narrow-equality candidate, unmeasured

A closed x64 block suffix can compare a zero-extended U8/U16 loaded value
with a truncated GP source using CMP8/CMP16. The load stays in place. Only
AND of the exact width mask and optionally XOR feeding equality-to-zero
are skipped; the branch must be I32 Eq/Ne, and skipped registers must be
dead on every edge and successor, including read-before-redefinition.
No benchmark names/inputs, guest offsets or polynomial recognition exist.
No new MachineIR instruction, CFG change or data transformation is added.
Fifteen narrow native comparisons in seven functions across JSON, regex,
reverse-complement and CoreMark; all 1,285 final MIR functions are identical
to the scan-only baseline. CoreMark function 2 b3 and b5 retain both guest
loads and replace the masking/comparison sequence with one narrow CMP.

Three proof/encoding tests cover low-byte exhaustion, 16-bit boundaries,
high bits, operand order, deadness, rejected ranges, REX low-byte selection
and operand-size prefix. Runtime search tests cover Eq/Ne, direct/XOR
forms, high-bit needles, bounds, writes before traps and instance reuse.
Local final x64 550 and ARM 569 units plus integrations pass without
compiler diagnostics. x64 spec 260/260 passes; private native CI validates
the committed final source before measuring.


Narrow candidate committed as 000d82165b30515c2c7d371a6361529cbb5f3ea7.
Private runtime 34054031131 and startup 34054031163 compare fixed main /
address-copy parent 91b6dd12 / narrow candidate. Full dev 34054414277 also
tests this exact commit; origin/dev/x64-hotpaths now points to 000d8216.
The old e0e14479 coalescing result is superseded. Current runtime source is
clean; only local evidence documents remain uncommitted.


### Narrow equality 000d8216 measured, retained pending full dev

Native 34054031131 draw 1 Intel 8573C: narrow/parent +0.37%
(P improvement 81.317762%), narrow/main +10.28%. Draw 2 AMD 7763:
+1.11% (99.970766%), narrow/main +8.85%. Actual native core, spec260
and warning audits execute and pass. Private startup 34054031163 AMD7763:
CoreMark +0.18% (P improvement 63.578366%), bz2 +0.65% (89.595946%),
ffmpeg -0.07% (36.627910%). No conclusive private startup regression;
full dev 34054414277 remains in progress, not yet passing. Evidence in
narrow-draw-* and narrow-startup-*. There is still no new competitor ranking.


### Narrow memory equality follow-up, unmeasured

The native narrow-equality plan may now keep its load as CMP's memory
operand when that result is dead on both successors and the compared
source differs from the loaded destination. Observed or source-overwriting
loads retain the register form. Only pure AND/optional XOR are crossed;
read width, address, zero-extension and trap order stay the same. A buffered
preceding pointer load or address snapshot is drained before skipping the
suffix, preserving its definition and trap point.

Ten comparisons in five functions across JSON, regex, reverse-complement
and CoreMark use memory operands; the other narrow comparisons retain
register operands. All 1,285 final MIR functions remain identical. New
proof tests cover register-form fallback; encoder and executed native-leaf
tests cover REX/SIB/displacements, both widths, high bits and address forms.
Final local x64 553 and ARM 569 units plus integrations pass, and x64 spec
260/260 passes, all without compiler diagnostics. No runtime claim yet.


### Memory equality 51497f8e withdrawn at 92c1d465

Private 34055011919 draw 1 AMD 9V74: memory/parent -0.99% (P regression
98.216770%), memory/main +4.49%. Draw 2 AMD7763: +0.07% (P improvement
91.451955%), memory/main +8.77%. Core/spec/audits execute and pass, but
execution does not support retaining memory fusion. Startup 34055011963
AMD7763: +0.62% CoreMark (P improvement 89.140857%), +0.05% bz2
(55.999592%), -0.06% ffmpeg (30.924935%), all inconclusive. Revert removes
memory plans, memory CMP encoder/emission, early buffered-op drain and
its three extra tests. Register-only narrow equality remains.

### Narrow full dev 34054414277, startup still pending

All 20 execution rows are saved. AMD7763 full-corpus geomean throughput
versus fixed main +13.347622%; ARM N2 +4.738082%. x64 tiny-keccak -4.89%
PLACEMENT and regex -0.88% NEGLIGIBLE remain included. ARM reverse-complement
-0.67% NEGLIGIBLE. Windows JIT LZ4 compress -4.42% RECOVERED has a wide
-11.28%..+14.53% range; mandelbrot -2.96% NEGLIGIBLE. Linux JIT SQLite
-1.39% NEGLIGIBLE. ARM Darwin JIT artifact creation now completes on this
revision, unlike the superseded coalescing run. The complete current run
still has startup/confirmation work; no overall passing claim yet.

### BMI2 155c0f9d, isolated final source 92c1d465, unmeasured

CPUID.7.0:EBX[8] enables scalar three-register SHLX/SHRX/SARX and RORX for
immediate rotations whose result differs from the input lane. Unsupported
hosts retain existing lowering. The cached feature probe preserves AMD
BEXTR and Intel FP-conversion preferences. Width/count wrapping and live
inputs are unchanged; no new MachineIR instruction or benchmark pattern.
Encoder formats were checked against Intel XED hsw-bmi-vex-isa.xed.txt;
CPUID enumeration against Intel's official BMI2 documentation.

Local Rosetta does NOT advertise BMI2. Its scalar fallback and encoder
checks pass; the two raw native BMI2 execution tests explicitly skip.
Therefore local tests are not evidence of executing BMI2. The private native
CI rejects a runner whose log contains that skip message, then runs all
core/spec checks before timing all 20 pinned execution benchmarks. It compares
000d8216 against 92c1d465, excluding withdrawn memory fusion from both.
Local combined x64 556 units and ARM569 plus integrations and 260 spec cases
pass without compiler diagnostics; final post-withdrawal checks are separate.


### Narrow full dev 34054414277 completed and checked

All 34 jobs completed, with no failed steps. All 17 primary job summaries
and both startup confirmations were inspected: no warning action-required,
soft-fail or unresolved REGRESSION. This revision passes the current gate;
numeric declines remain included. The seven startup rows are saved below
and as narrow-dev-{x64,arm}-startup-comparison.json.

| Platform | Startup case | Throughput delta | Gate classification |
|---|---|---:|---|
| x64 | startup/bz2 | +0.10% | RECOVERED |
| x64 | startup/pulldown-cmark | -1.61% | NEGLIGIBLE |
| x64 | startup/spidermonkey | -0.60% | RECOVERED |
| x64 | startup/ffmpeg | -1.13% | NEGLIGIBLE |
| x64 | startup/coremark | -5.63% | NOISY-FLOOR |
| x64 | startup/argon2 | -3.13% | NOISY-FLOOR |
| x64 | startup/erc20 | +0.87% | RECOVERED |
| arm | startup/bz2 | +0.85% | IMPROVEMENT |
| arm | startup/pulldown-cmark | -0.44% | RECOVERED |
| arm | startup/spidermonkey | -0.18% | RECOVERED |
| arm | startup/ffmpeg | +0.84% | IMPROVEMENT |
| arm | startup/coremark | -1.48% | NEGLIGIBLE |
| arm | startup/argon2 | -1.09% | NEGLIGIBLE |
| arm | startup/erc20 | -1.07% | NEGLIGIBLE |

### BMI2 native validation and startup, execution still pending

Diagnostic 4b2b41ed launches complete pinned 20-case execution run
34055798689 and startup 34055798684 for parent000d8216/candidate92c1d465.
Both execution jobs (101547349215 draw1, 101547349024 draw2) completed
native BMI2 core/spec validation before timing; the workflow rejects
skipped BMI2 execution tests. Local final source checks separately pass
x64 553 and ARM569 unit tests plus integrations without compiler diagnostics.

Startup job101547348977 AMD7763 executes with clean build audits. BMI2/parent:
CoreMark +0.24% (P improvement75.170262%), bz2 -0.48% (P regression99.658354%),
ffmpeg -0.14% (P regression91.063643%). BMI2/main -2.98%, -0.04%, +0.06%.
The small bz2 decline is recorded; BMI2 remains unmeasured for execution.
Artifact9996013931 is saved locally with all three startup comparisons.

### Next instruction-selection hypothesis

CL uses full-width copies before some 32-bit ALU operations, but Intel's
optimization manual lists both MOV32 and MOV64 as elimination candidates.
There is not sufficient evidence to change Nano's copy width. Instead,
the next small generic candidate lowers exact byte/word AND masks to
MOVZX, combining copy and truncation for a distinct destination and
shortening an in-place mask. It must not claim ALU flags from MOVZX.
No benchmark identity or fixed input is used.


### MOVZX candidate 7d1a5f78, not yet measured

Exact 0xff/0xffff masks now use native MOVZX8/16, for either integer width.
No MIR changes: all 1,285 corpus functions match retained narrow-equality
000d8216 locally. 139 functions in 14 modules change truncation emission;
CoreMark changes 27 operations in seven functions, with 113 fewer body bytes.
This is code shape only, not timing evidence. Local Rosetta does not emit
BMI2; the parent/candidate native experiment includes BMI2 in both.

Two native encoder tests verify low-byte REX, extended registers, full-carrier
zeroing, destination aliases and exhaustive 16-bit inputs with dirty high bits.
A WASM integration test checks both widths, live inputs, wrapping, zero tests
after a different ALU condition, and masks outside the selection range.
Local x64 555 units, ARM569 units and all integrations pass without compiler
diagnostics; x64 spec260/260 passes. Source-only candidate is pushed; local
evidence remains uncommitted.


### BMI2 92c1d465: complete two-draw execution result

Run34055798689 completed. Draw1 job101547349215 and draw2 job101547349024
are both AMD EPYC9V74 hosts. The two direct BMI2 execution tests actually run
on each host, and native core/spec260 plus real warning audits are clean.
The earlier REGRESSION rows in the log are measurement-helper test fixtures
(100ns values), not the final runtime summary. Tool preparation separately
prints rustup's default override and pinned cargo-criterion's yanked
crossbeam-deque0.7.3 dependency notices; these are not source compiler
diagnostics and are not suppressed.

Draw1 full20 geomean throughput versus parent000d8216: +0.490114%.

| Case | Throughput delta | Classification |
|---|---:|---|
| execute/counter-local | +0.13% | PASS |
| execute/counter-param | +0.01% | RECOVERED |
| execute/counter-global | -0.03% | RECOVERED |
| execute/fibonacci-rec | +0.05% | PASS |
| execute/fibonacci-iter | -0.10% | PASS |
| execute/fibonacci-tail | +0.17% | PASS |
| execute/sort | -0.03% | PASS |
| execute/prime_sieve | +2.19% | IMPROVEMENT |
| execute/matrix_mul | +0.39% | PASS |
| execute/nbody | +0.01% | PASS |
| execute/argon2 | -0.04% | PASS |
| execute/tiny_keccak | +0.19% | PASS |
| execute/mandelbrot | +0.57% | PASS |
| execute/spectralnorm | +0.11% | PASS |
| execute/compression | -0.05% | PASS |
| execute/word_count | +0.76% | IMPROVEMENT |
| execute/json_parse | +1.15% | PASS |
| execute/reverse_complement | -0.82% | PASS |
| execute/regex_redux | +4.95% | IMPROVEMENT |
| execute/bulk-ops | +0.34% | PASS |

Draw2 full20 geomean throughput versus parent000d8216: +0.319144%.

| Case | Throughput delta | Classification |
|---|---:|---|
| execute/counter-local | -0.03% | PASS |
| execute/counter-param | -0.05% | PASS |
| execute/counter-global | -0.19% | PASS |
| execute/fibonacci-rec | -0.47% | RECOVERED |
| execute/fibonacci-iter | +0.21% | PASS |
| execute/fibonacci-tail | +0.04% | PASS |
| execute/sort | -0.21% | RECOVERED |
| execute/prime_sieve | +2.07% | IMPROVEMENT |
| execute/matrix_mul | -0.09% | PASS |
| execute/nbody | +0.00% | PASS |
| execute/argon2 | +0.04% | PASS |
| execute/tiny_keccak | -0.15% | RECOVERED |
| execute/mandelbrot | -0.00% | PASS |
| execute/spectralnorm | +0.25% | PASS |
| execute/compression | -0.00% | PASS |
| execute/word_count | +0.70% | IMPROVEMENT |
| execute/json_parse | -0.69% | PASS |
| execute/reverse_complement | -0.53% | RECOVERED |
| execute/regex_redux | +4.75% | IMPROVEMENT |
| execute/bulk-ops | +0.87% | RECOVERED |


Retain BMI2 provisionally on repeated prime-sieve, word-count and regex gains.
All remaining rows, including declines, are preserved. Native Intel execution
and a full dev gate are still required; two AMD9V74 hosts are not an Intel
result. This remains Nano-only differential evidence, not a competitor rank.


MOVZX isolated diagnostics a487b110: CoreMark34056575453 (jobs101549471748
draw1/101549471921 draw2) and startup34056575447 (job101549471674).
Native validation steps have completed; performance is still pending.

### Shifted-operand commutation abae6cdc, unmeasured

For the existing IntBinaryShifted Add/And/Or/Xor, a destination different
from the unshifted left input may hold the shifted right input directly.
The final commutative ALU instruction then reads lhs. Subtraction and
dst==lhs retain backend scratch. No register assignment, lifetime, CFG
or MIR changes; all1285 corpus MIR functions remain identical to7d1a5f78.
CoreMark loses19 MOVs across four functions; CRC function10 loses12 MOVs
(80 to68), with unchanged shifts/ALU operations. Longer extended-register
shift encodings mean its body shrinks only12 bytes. This has not been timed.

The removed e0e14479 coalescer also affected shifted binary operations,
but renamed dead MIR inputs/results. This candidate changes only native
operand order within one operation and preserves all original lifetimes.
The distinction is a different hypothesis, not evidence of a performance win.
New integration coverage spans both widths, Add/Sub/And/Or/Xor, logical and
arithmetic shifts, rotations, modulo-width counts and live-input variants.
Local x64 555 and ARM569 units plus integrations pass, x64spec260/260 passes,
all without compiler diagnostics.


### MOVZX CoreMark/startup complete; execution-suite follow-up required

34056575453 draw1 AMD7763: movzx/parent +0.45% (P improvement98.381313%),
movzx/main +8.94%. Draw2 AMD9V74: -0.24% (P regression85.200993%),
movzx/main +5.36%. All native core/spec260 and source warning audits pass.
Neither draw establishes a conclusive CoreMark improvement.

Startup34056575447 AMD7763: +0.26% CoreMark (P improvement76.761259%),
+0.75% bz2 (97.142685%), +0.31% ffmpeg (99.730797%). Startup/main
-1.70%, +0.31%, +0.21%. Full artifacts9996279431/9996274912/9996246567
are saved locally. MOVZX remains provisional pending full20 execution.

Shifted-operand diagnostics3c9537d0: CoreMark34056956282 and
startup34056956312, both in progress. BMI2 alone92c1d465 is now on
origin/dev/x64-hotpaths, full dev34056978024 in progress.


### Shifted commutation: CoreMark/startup complete, no conclusive execution win

34056956282 draw1 job101550488038 AMD7763: shiftcommute/parent +0.11%
(P improvement91.972472%), shiftcommute/main +8.98%. Draw2 job101550487912
AMD9V74: +0.24% (82.616982%), shiftcommute/main +4.86%. Both native
core/spec260/source warning audits pass. These tiny positive deltas are
inconclusive, not confirmed improvements. Full-suite execution is unmeasured.

Startup34056956312 job101550488014 AMD7763: -0.22% CoreMark (P regression
62.936770%), +0.16% bz2 (P improvement68.504473%), +0.05% ffmpeg
(86.347287%), all inconclusive. Startup/main -2.06%, +0.10%, +0.47%.
Artifacts9996377505/9996384634/9996356250 saved locally.

MOVZX full20 follow-up34057291095 uses diagnostic50f65e58, currently
in progress. RORX for existing shifted rotate forms is being prepared
independently in /tmp/sf-x64-rorx-shift, based on BMI2-only92c1d465.
The source main branch remains abae6cdc, origin/dev remains BMI2-only92c1d465.


### Rotated operands: independent candidate prepared on BMI2-only parent

/tmp/sf-x64-rorx-shift starts from92c1d465 and replaces MOV+ROR scratch
inside the existing IntBinaryShifted operation with RORX on BMI2 hosts.
Final ALU order, register allocation, lifetimes and fallback stay unchanged.
It excludes MOVZX and commutation. The same shifted-binary semantic test
used by abae6cdc is added independently; both widths, all relevant ALU
operations, wrapping counts and live-input variants are exercised.
Local x64 553 and ARM569 units plus integrations pass without diagnostics.
Local Rosetta does not execute BMI2; native CI must validate the new path.

The separate spec build initially failed because its fresh target directory
lacked the pinned testsuite and the build script tried network download
(the runtime TESTSUITE_DIR variable does not configure that build script).
The already validated local testsuite51279a9d was copied into its target
directory. The offline rebuild then passes without compiler diagnostics and
x64 spec260/260 passes. No lint suppression or source workaround was added.


RORX shifted candidate fc373194005f969d4279ac295f1df62a916ee685 is pushed
on codex/x64-rorx-shift-candidate; its worktree is clean. Diagnostics
4ff7577268bf4e2e417831a9a5b4faa0aa4a43c6 launch full20 run34057818814 and
startup34057818795, in progress. This profile workflow currently measures
RORX, not CoreMark. MOVZX full20 run34057291095 (jobs101551391394/101551391518)
and commutation full20 run34057703924 (jobs101552508796/101552508917) are
also in progress. All helper/workflow checks (44 tests, lint policy, diff) pass.

BMI2 dev34056978024: initial 17 primaries are progressing. Resolve warnings
audit has no executed warning-action output (the echoed shell branch and
fixture regressions must not be mistaken for execution). All completed
interpreter jobs, ARM Linux/Darwin JIT, x64 Linux/Windows JIT have summaries
inspected and no unresolved REGRESSION or warning action. Linux JIT
lua-sunfish -1.88% NEGLIGIBLE remains recorded. The Linux/Windows CoreMark
rows +1.63%/+6.25% use the separate WASI corpus and are not the exact
wasmi CoreMark score. Primary wasmi JIT execution and startup are still
pending, so the complete current run is not yet passing.

The mem0-size register investigation is in /tmp/sf-x64-memsize-register-notes.md.
It identifies a global ABI role and module-wide config/cache-identity
constraints; no ad hoc fixed-register reuse was implemented. Local CL
function2 disassembly is now /tmp/sf-coremark-cl-list.txt (system objdump
reads the existing ELF cwasm); no new competitor compilation was needed.


### MOVZX full20 result: insufficient execution benefit, withdraw planned

Draw1: Model name:                              AMD EPYC 7763 64-Core Processor, full20 throughput geomean -0.008173% vs92c1d465.

| Case | Delta | Status |
|---|---:|---|
| execute/counter-local | -0.08% | PASS |
| execute/counter-param | -0.08% | RECOVERED |
| execute/counter-global | +0.54% | PASS |
| execute/fibonacci-rec | -0.14% | RECOVERED |
| execute/fibonacci-iter | +0.00% | PASS |
| execute/fibonacci-tail | -0.55% | PASS |
| execute/sort | -0.04% | PASS |
| execute/prime_sieve | +0.02% | PASS |
| execute/matrix_mul | -0.07% | PASS |
| execute/nbody | -0.64% | PASS |
| execute/argon2 | -0.04% | PASS |
| execute/tiny_keccak | -0.11% | PASS |
| execute/mandelbrot | +0.38% | PASS |
| execute/spectralnorm | -0.21% | RECOVERED |
| execute/compression | +0.01% | RECOVERED |
| execute/word_count | -0.04% | PASS |
| execute/json_parse | +0.84% | PASS |
| execute/reverse_complement | -0.22% | PASS |
| execute/regex_redux | +0.55% | PASS |
| execute/bulk-ops | -0.26% | PASS |

Draw2: Model name:                              AMD EPYC 7763 64-Core Processor, full20 throughput geomean +0.035626% vs92c1d465.

| Case | Delta | Status |
|---|---:|---|
| execute/counter-local | -0.08% | PASS |
| execute/counter-param | -0.02% | PASS |
| execute/counter-global | -0.23% | PASS |
| execute/fibonacci-rec | -0.17% | PASS |
| execute/fibonacci-iter | -0.16% | PASS |
| execute/fibonacci-tail | -0.57% | RECOVERED |
| execute/sort | -0.30% | RECOVERED |
| execute/prime_sieve | +0.01% | PASS |
| execute/matrix_mul | +0.02% | PASS |
| execute/nbody | +0.04% | PASS |
| execute/argon2 | -0.03% | PASS |
| execute/tiny_keccak | -0.00% | PASS |
| execute/mandelbrot | -0.00% | PASS |
| execute/spectralnorm | +0.02% | PASS |
| execute/compression | +0.01% | PASS |
| execute/word_count | -0.05% | PASS |
| execute/json_parse | +1.45% | PASS |
| execute/reverse_complement | +0.48% | PASS |
| execute/regex_redux | +0.08% | RECOVERED |
| execute/bulk-ops | +0.22% | RECOVERED |

Both native core/spec260/warning audits pass; neither complete execution
draw has a confirmed improvement. JSON trends positive but not conclusive.
With inconclusive CoreMark +0.45%/-0.24% and no reliable aggregate execution
benefit, remove MOVZX rather than accumulate this provisional candidate.
Its small one-host startup gains do not establish a repeated retained win.


### Shift commutation, RORX and BMI2 complete results

MOVZX was withdrawn by79814f25; the resulting x64 default test suite passes (553 units plus integrations), without compiler diagnostics.

Shift commutation34057703924 measures7d1a5f78→abae6cdc, so both sides contain the subsequently withdrawn MOVZX. Repeat prime_sieve and spectralnorm gains support retaining commutation provisionally, but its final base without MOVZX still needs verification. Sort -1.16% and word_count -0.48% NEGLIGIBLE in draw2 remain real numeric declines.

RORX shifted operands34057818814/34057818795 have no confirmed execution improvements and add compilation cost; candidatefc373194 remains an unmerged experiment. Both native full20 jobs execute BMI2 tests (554 units plus integrations), spec260 passes and actual source warning audits are clean. Commutation native jobs similarly pass556 units plus integrations/spec260. Tool preparation prints the preexisting rustup override and pinned cargo-criterion yanked crossbeam-deque notices, not source compiler diagnostics.

BMI2 retained92c1d465 full dev34056978024 completed: all34 jobs and individual steps checked, primary summaries and confirmation jobs inspected, no actual SOFT-FAIL/ACTION REQUIRED/compiler-warning or unresolved REGRESSION. The x64 execution host is AMD7763; ARM execute is Neoverse-N2; x64 startup is AMD9V74. Numeric declines and placement/noisy-floor classifications remain included below. This is a Nano-vs-main comparison, not a new competitor closure.


#### shift-commute-full-draw-1

Model name:                              AMD EPYC 9V74 80-Core Processor

Baseline 7d1a5f78657fd5e240b10da4bcddf50dcfa344b6; candidate abae6cdcd621e0ed13c0b4e9df595f867abe26c6. Full 20 metric geomean +0.286076%.


| Case | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.00% | PASS |

| execute/counter-param | +0.15% | PASS |

| execute/counter-global | -0.04% | PASS |

| execute/fibonacci-rec | -0.01% | PASS |

| execute/fibonacci-iter | -0.29% | PASS |

| execute/fibonacci-tail | +0.03% | PASS |

| execute/sort | -0.27% | RECOVERED |

| execute/prime_sieve | +3.76% | IMPROVEMENT |

| execute/matrix_mul | -0.14% | PASS |

| execute/nbody | -1.52% | RECOVERED |

| execute/argon2 | +0.26% | RECOVERED |

| execute/tiny_keccak | -0.05% | PASS |

| execute/mandelbrot | -1.82% | RECOVERED |

| execute/spectralnorm | +1.69% | IMPROVEMENT |

| execute/compression | -0.15% | RECOVERED |

| execute/word_count | +0.96% | RECOVERED |

| execute/json_parse | +1.29% | RECOVERED |

| execute/reverse_complement | +0.78% | RECOVERED |

| execute/regex_redux | +0.13% | PASS |

| execute/bulk-ops | +1.07% | PASS |


#### shift-commute-full-draw-2

Model name:                              AMD EPYC 9V74 80-Core Processor

Baseline 7d1a5f78657fd5e240b10da4bcddf50dcfa344b6; candidate abae6cdcd621e0ed13c0b4e9df595f867abe26c6. Full 20 metric geomean +0.188186%.


| Case | Delta | Status |
|---|---:|---|

| execute/counter-local | -0.08% | RECOVERED |

| execute/counter-param | -0.02% | PASS |

| execute/counter-global | -0.15% | PASS |

| execute/fibonacci-rec | +0.05% | PASS |

| execute/fibonacci-iter | +0.03% | PASS |

| execute/fibonacci-tail | +0.10% | RECOVERED |

| execute/sort | -1.16% | NEGLIGIBLE |

| execute/prime_sieve | +3.75% | IMPROVEMENT |

| execute/matrix_mul | -0.02% | PASS |

| execute/nbody | +0.41% | PASS |

| execute/argon2 | +0.01% | PASS |

| execute/tiny_keccak | +0.05% | RECOVERED |

| execute/mandelbrot | -0.50% | RECOVERED |

| execute/spectralnorm | +1.59% | IMPROVEMENT |

| execute/compression | +0.18% | PASS |

| execute/word_count | -0.48% | NEGLIGIBLE |

| execute/json_parse | +0.36% | RECOVERED |

| execute/reverse_complement | -0.77% | RECOVERED |

| execute/regex_redux | +0.36% | RECOVERED |

| execute/bulk-ops | +0.14% | PASS |


#### rorx-shift-draw-1

Model name:                              AMD EPYC 9V74 80-Core Processor

Baseline 92c1d465818d0f814b8c59b11146b5796af53059; candidate fc373194005f969d4279ac295f1df62a916ee685. Full 20 metric geomean +0.072812%.


| Case | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.17% | RECOVERED |

| execute/counter-param | -0.20% | PASS |

| execute/counter-global | +0.57% | PASS |

| execute/fibonacci-rec | +0.16% | RECOVERED |

| execute/fibonacci-iter | -0.11% | PASS |

| execute/fibonacci-tail | +0.68% | RECOVERED |

| execute/sort | -0.03% | RECOVERED |

| execute/prime_sieve | +0.04% | PASS |

| execute/matrix_mul | +0.01% | PASS |

| execute/nbody | -0.17% | PASS |

| execute/argon2 | +0.10% | PASS |

| execute/tiny_keccak | +0.34% | PASS |

| execute/mandelbrot | +0.01% | PASS |

| execute/spectralnorm | -0.00% | PASS |

| execute/compression | +0.06% | PASS |

| execute/word_count | +0.09% | PASS |

| execute/json_parse | -1.06% | RECOVERED |

| execute/reverse_complement | +0.71% | RECOVERED |

| execute/regex_redux | -0.03% | RECOVERED |

| execute/bulk-ops | +0.13% | PASS |


#### rorx-shift-draw-2

Model name:                              AMD EPYC 9V74 80-Core Processor

Baseline 92c1d465818d0f814b8c59b11146b5796af53059; candidate fc373194005f969d4279ac295f1df62a916ee685. Full 20 metric geomean -0.340938%.


| Case | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.24% | PASS |

| execute/counter-param | -0.04% | PASS |

| execute/counter-global | +0.00% | PASS |

| execute/fibonacci-rec | -0.04% | PASS |

| execute/fibonacci-iter | +0.05% | PASS |

| execute/fibonacci-tail | +0.13% | PASS |

| execute/sort | -0.25% | PASS |

| execute/prime_sieve | +0.18% | PASS |

| execute/matrix_mul | -0.01% | PASS |

| execute/nbody | -3.13% | RECOVERED |

| execute/argon2 | -0.22% | PASS |

| execute/tiny_keccak | -1.80% | PASS |

| execute/mandelbrot | +0.08% | PASS |

| execute/spectralnorm | -0.01% | PASS |

| execute/compression | +0.03% | PASS |

| execute/word_count | +0.18% | PASS |

| execute/json_parse | -0.92% | PASS |

| execute/reverse_complement | -0.64% | PASS |

| execute/regex_redux | -0.26% | RECOVERED |

| execute/bulk-ops | -0.36% | RECOVERED |


#### rorx-shift-startup

Model name:                              AMD EPYC 9V74 80-Core Processor

coremark-results

parent/main: -2.829253%, P(regression) 99.996101%, P(improvement) 0.003899%.

rorxshift/main: -3.305615%, P(regression) 99.987492%, P(improvement) 0.012508%.

rorxshift/parent: -0.490232%, P(regression) 88.755418%, P(improvement) 11.244582%.

ffmpeg-results

parent/main: -0.170267%, P(regression) 96.721920%, P(improvement) 3.278080%.

rorxshift/main: -0.641111%, P(regression) 99.961056%, P(improvement) 0.038944%.

rorxshift/parent: -0.471647%, P(regression) 99.974592%, P(improvement) 0.025408%.

bz2-results

parent/main: +0.462253%, P(regression) 0.722282%, P(improvement) 99.277718%.

rorxshift/main: -0.388584%, P(regression) 95.655357%, P(improvement) 4.344643%.

rorxshift/parent: -0.846922%, P(regression) 99.700668%, P(improvement) 0.299332%.


#### bmi2-dev-arm-startup

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate 92c1d465818d0f814b8c59b11146b5796af53059. Full 7 metric geomean -0.181144%.


| Case | Delta | Status |
|---|---:|---|

| startup/bz2 | +1.51% | IMPROVEMENT |

| startup/pulldown-cmark | -0.62% | PASS |

| startup/spidermonkey | -0.11% | RECOVERED |

| startup/ffmpeg | +1.10% | IMPROVEMENT |

| startup/coremark | -0.99% | NEGLIGIBLE |

| startup/argon2 | -1.31% | NEGLIGIBLE |

| startup/erc20 | -0.81% | NEGLIGIBLE |


#### bmi2-dev-arm-execute

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate 92c1d465818d0f814b8c59b11146b5796af53059. Full 20 metric geomean +4.734384%.


| Case | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.02% | PASS |

| execute/counter-param | -0.00% | PASS |

| execute/counter-global | +0.02% | PASS |

| execute/fibonacci-rec | -0.01% | PASS |

| execute/fibonacci-iter | +0.01% | PASS |

| execute/fibonacci-tail | +0.01% | PASS |

| execute/sort | +0.26% | PASS |

| execute/prime_sieve | +0.49% | PASS |

| execute/matrix_mul | -0.11% | PASS |

| execute/nbody | +0.04% | PASS |

| execute/argon2 | +154.45% | IMPROVEMENT |

| execute/tiny_keccak | -0.12% | RECOVERED |

| execute/mandelbrot | -0.02% | PASS |

| execute/spectralnorm | +0.00% | PASS |

| execute/compression | +0.08% | PASS |

| execute/word_count | -0.04% | PASS |

| execute/json_parse | -0.68% | RECOVERED |

| execute/reverse_complement | -0.64% | RECOVERED |

| execute/regex_redux | -0.08% | RECOVERED |

| execute/bulk-ops | -0.11% | RECOVERED |


#### bmi2-dev-x64-execute

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate 92c1d465818d0f814b8c59b11146b5796af53059. Full 20 metric geomean +13.575760%.


| Case | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.13% | PASS |

| execute/counter-param | -0.03% | PASS |

| execute/counter-global | +0.03% | PASS |

| execute/fibonacci-rec | +1.77% | PASS |

| execute/fibonacci-iter | +1.32% | IMPROVEMENT |

| execute/fibonacci-tail | +69.29% | IMPROVEMENT |

| execute/sort | +42.78% | IMPROVEMENT |

| execute/prime_sieve | +6.54% | IMPROVEMENT |

| execute/matrix_mul | +1.53% | PASS |

| execute/nbody | +1.85% | IMPROVEMENT |

| execute/argon2 | +158.04% | IMPROVEMENT |

| execute/tiny_keccak | -4.72% | PLACEMENT |

| execute/mandelbrot | -0.07% | PASS |

| execute/spectralnorm | +0.08% | PASS |

| execute/compression | +0.52% | IMPROVEMENT |

| execute/word_count | +7.63% | IMPROVEMENT |

| execute/json_parse | +3.93% | IMPROVEMENT |

| execute/reverse_complement | +68.90% | IMPROVEMENT |

| execute/regex_redux | -0.62% | RECOVERED |

| execute/bulk-ops | -0.02% | PASS |


#### bmi2-dev-x64-startup

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate 92c1d465818d0f814b8c59b11146b5796af53059. Full 7 metric geomean -1.659198%.


| Case | Delta | Status |
|---|---:|---|

| startup/bz2 | +0.03% | PASS |

| startup/pulldown-cmark | -3.04% | RECOVERED |

| startup/spidermonkey | -0.89% | RECOVERED |

| startup/ffmpeg | +0.39% | RECOVERED |

| startup/coremark | -2.97% | NOISY-FLOOR |

| startup/argon2 | -2.39% | RECOVERED |

| startup/erc20 | -2.67% | RECOVERED |


### Control-flow loop alignment: rejected

Independent1e899334e154ad50cbdb45c55a45a9b82fe915d3 against92c1d465, CoreMark34059193317 and startup34059193310. Iterative DFS aligns cycle back-edge targets rather than every backward text target; this changes more than acyclic-join padding. Local1285 MIR functions unchanged; CoreMark code17484→17085 bytes, regex1731613→1688290. Native code shrink is not a runtime win.

Local x64 core556 and ARM572 plus integrations/spec260 pass after removing a newly unused import; no warning suppression. Native CoreMark jobs core/spec260/actual warning audits pass. Both execution draws regress, so this candidate is rejected and remains unmerged.


loop-align-draw-1

Model name:                              AMD EPYC 7763 64-Core Processor

coremark-out

parent/main: +8.739525%, P(regression) 0.000017%, P(improvement) 99.999983%.

loopalign/main: +4.954657%, P(regression) 0.000244%, P(improvement) 99.999756%.

loopalign/parent: -3.480673%, P(regression) 99.997067%, P(improvement) 0.002933%.


loop-align-draw-2

Model name:                              Intel(R) Xeon(R) 6973P-C

coremark-out

parent/main: +16.412567%, P(regression) 0.038570%, P(improvement) 99.961430%.

loopalign/main: +13.255957%, P(regression) 0.022815%, P(improvement) 99.977185%.

loopalign/parent: -2.711572%, P(regression) 99.136281%, P(improvement) 0.863719%.


loop-align-startup

Model name:                              AMD EPYC 9V74 80-Core Processor

coremark-results

parent/main: -2.317256%, P(regression) 99.852525%, P(improvement) 0.147475%.

loopalign/main: -2.009155%, P(regression) 99.632780%, P(improvement) 0.367220%.

loopalign/parent: +0.315410%, P(regression) 24.933898%, P(improvement) 75.066102%.

ffmpeg-results

parent/main: -0.221636%, P(regression) 98.185882%, P(improvement) 1.814118%.

loopalign/main: -0.512915%, P(regression) 99.904836%, P(improvement) 0.095164%.

loopalign/parent: -0.291926%, P(regression) 99.659310%, P(improvement) 0.340690%.

bz2-results

parent/main: +0.373824%, P(regression) 5.194772%, P(improvement) 94.805228%.

loopalign/main: +0.044942%, P(regression) 42.506536%, P(improvement) 57.493464%.

loopalign/parent: -0.327656%, P(regression) 95.890493%, P(improvement) 4.109507%.


Next experiment /tmp/sf-x64-uncached-memsize uses a coherent global x64 ABI without pinned mem0 length, with R13 added to preserved dynamics. It is uncommitted. New memory_length tests exposed TWO failures also reproduced on exact parent92c1 in /tmp/sf-uncached-parent: forced-template memory parameter/result handling returns zero, and memory64 address0x100000000 aliases low memory rather than trapping. Both kept failing for investigation. A separate codex/jit-memory-boundary-fix is in progress; do not count the register experiment as validated.


## Native validation of the uncached mem0-length candidate

The prior uncommitted-candidate note above is superseded. Two existing failures were reproduced on parent 92c1d465 and fixed separately in be3db3a50017ae334d7a289129de0140568ad293 (cherry-picked onto the development branch as 3376e54f1f9d043fb640645c9faf65b61c2f543f): full memory64 addresses must not truncate to 32 bits, offset/access-end overflow must trap, and template bodies must use canonical frame parameter/result locations. Four integration tests cover growth with live values, explicit/template bounds, ordinary callers of template callees, and memory64 wraparound in two memories. Local default x64/ARM tests and 260 x64 spec cases passed; the four memory cases also passed without guard pages. The 1,285 fixed-corpus MIR functions are unchanged by this prerequisite. Native development validation of 3376e54f is running in 34060939678; this is not yet a completed gate.

Candidate 8d867e57597dbff1bf24d87ec830cdc2a75b06c0 is based on be3db3a5 and is isolated on codex/x64-uncached-memsize. Its x64 ABI loads memory 0 length from CTX when needed, leaves logical fixed role 3 unmapped, and adds R13 to the preserved dynamic bank. Other architectures retain their cached length. Explicit checks protect the effective-address register when borrowing a length temporary; templates require a third volatile lane. Local x64 (553 unit tests plus integrations), ARM64 (569 plus integrations), x64 spec (260), and unguarded memory checks passed, with no compiler warnings. Rosetta cannot execute BMI2; native validation explicitly rejects skipped BMI2 execution checks.

Native CoreMark comparison 34060913741 and JIT startup audit 34060913749 were launched by private diagnostic commit b2ed2a1050230fb80a99ad136b2b5588e8d16815. Both compare parent be3db3a5 with candidate 8d867e57, also retaining campaign main f73219f5 as an anchor. No measured execution benefit is claimed yet and the register candidate is not integrated. These workflows remain isolated to codex/x64-profiling/manual dispatch.

For the current gap estimate, retained 92c1d465 full-20 execution gain on AMD EPYC 7763 is 13.5757596% relative to campaign main. Applied to the frozen same-model three-engine anchor, this projects approximately 1.1% lower throughput than V8 and 9.7% higher throughput than Cranelift. Parent 92c1d465 in CoreMark draw 34059193317 gained 8.7395251% over campaign main on AMD 7763, projecting +0.7769% versus V8 and -8.4512% versus Cranelift. Closing the latter needs another 9.2314% Nano uplift. These are projections across runs, not a new same-host three-engine measurement; the small V8/CoreMark advantage is not a stable lead. Intel has no equally complete current projection.


### Uncached mem0 length rejected

Draw 1: Model name:                              AMD EPYC 7763 64-Core Processor; parent/main +8.746981%, P(regression) 0.000934%; memsize/main +4.401352%, P(regression) 0.002586%; memsize/parent -3.996091%, P(regression) 99.999426%

Draw 2: Model name:                              AMD EPYC 7763 64-Core Processor; parent/main +8.765035%, P(regression) 0.000136%; memsize/main +4.545839%, P(regression) 0.000346%; memsize/parent -3.879185%, P(regression) 99.998064%

Native CoreMark 34060913741 completed with both candidates slower. Both jobs ran the native BMI2 execution checks (no skip), 554 core unit tests plus integrations, the four unguarded memory integration cases, and 260 specification files; all source builds and correctness logs had zero warnings/errors. The separately added high-register-pressure memory integration test was only local and is not part of this pinned CI revision. It passes on x64 default, x64 without guard pages, and ARM64.

Startup 34060913749 on AMD EPYC 7763: CoreMark -2.853883% (P regression 99.691072%), bz2 -2.251936% (99.996459%), ffmpeg -2.040053% (99.999995%) against be3db3a5. Code-size reduction and fewer static frame references did not translate into runtime improvement. Candidate 8d867e57 remains unmerged; the memory64/template correctness prerequisite remains retained. No full-20 run is warranted for this rejected CoreMark candidate.


### Shift commutation withdrawn after the final combination check

Development run 34060939678 at 3376e54f1f9d043fb640645c9faf65b61c2f543f tested shift commutation without MOVZX plus the memory64/template correctness fix. Its full-20 x64 execution primary on AMD EPYC 7763 reports geometric mean +11.6495561% against campaign main, with spectralnorm -8.0846% classified REGRESSION, tiny_keccak -4.653% PLACEMENT, and all twenty rows preserved in shift-final-x64-execute/comparison.json. Sort is only +11.26% rather than the older 92c1d465 same-model +42.78%; this is a cross-run observation, not an isolated causal measurement. ARM64 N2 full-20 geometric mean is +4.6012386%, with every row retained in shift-final-arm-execute. The x64 WASI primary ran on Intel Xeon 6973P-C and classified LZ4 compression -5.14% REGRESSION; WASI CoreMark +7.88% is a different module from the fixed suite CoreMark. N2 JIT startup primary also flagged erc20 -2.00% REGRESSION. None of these red results is dismissed on the basis of GitHub's individual soft-fail job conclusions.

The unstable commutation combination was withdrawn at 54d2c012. Its reverse diff restores sf-nano-core exactly to be3db3a5 before the independent ARM64 fix and additional tests below. The old workflow subsequently became cancelled through the normal development-branch concurrency rule when the restored revision was pushed; the wasmi x64 execution and ARM64 startup confirmation jobs were cancelled. This run is incomplete, with unresolved primary regressions, and must not be called a pass. Restoration run 34062142640 is now validating 20b01c3dba8b209228bc8ce9a64c92305847ab13. Do not claim recovery before its evidence is inspected.

### ARM64 constant operand scratch exhaustion fixed separately

New generic call-error tests reproduced a code-generation panic on exact parent be3db3a5: signed division of two constants occupied both ARM64 operand temporaries, then requested a third for the overflow check. Failure evidence is /tmp/sf-call-status-parent-arm-repro.log and /tmp/sf-arm64-constant-parent-division-repro.log. Fix 402f5ebf38103cac89d93e624cd14ab13d5301dc stages the left immediate into the distinct result register whenever both operands are immediates, leaving a backend temporary for division/remainder/rotation. It does not change the register bank or suppress diagnostics. The independent fix passed ARM64 569 units plus all integrations and 260 specification files; the new call-status and 160-case integer-division/remainder tests also passed on x64. The development branch retains it as 9503cb33. Test-only cb5b3da3 (main cherry-pick 20b01c3d) separately checks explicit memory bounds with 4, 8, 12, and 16 independent live integer results under 32/64-bit addressing; local x64 default, x64 without guard pages, and ARM64 passed.

### Compiled-call return flags candidate

Candidate 7a8e7f12eed6754030e2df6836d9d76f0e0fc92a on codex/x64-call-flags is isolated from the withdrawn register and shift experiments. Parent cb5b3da3100d3f69f12e9b907ba5b6b30fcdc096 has exactly the same sf-nano-core source as restored development HEAD 20b01c3d. Internal x64 success returns already end in XOR EAX,EAX. Error tails now re-establish ZF from the nonzero status after frame undo. Compiled callers restore their frame with LEA and branch on the returned ZF directly, omitting the repeated TEST; public C entries and foreign-helper status checks retain explicit tests. Tail calls pass through the callee's same internal return contract. No MIR transformations or architecture register-bank changes are involved.

Local x64 default 553 units plus integrations, ARM64 569 plus integrations (after the independent constant-operand fix), unguarded x64 call-status/division/memory tests, and x64 260 spec files pass without source warnings. All 1,285 fixed-corpus MIR functions remain identical to be3db3a5. Generated CoreMark bytes 17,484 -> 17,363, recursive Fibonacci 219 -> 216, regex 1,731,613 -> 1,710,239. These are static sizes, not measured performance improvements. Probe /tmp/sf-call-flags-probe, dump /tmp/sf-call-flags-dump, source /tmp/sf-x64-call-flags.

Diagnostic commit da3e98ef launched CoreMark run 34062227028 and startup run 34062227036. Its subsequent recursive measurement omitted the required --expected-output-i64 argument; that step is expected to fail. CoreMark results may be used only if its earlier measurement and correctness steps completed, and the workflow must not be called passing. The configuration was corrected with expected Fibonacci(30)=832040 and a local harness smoke check; Rosetta timings from that smoke check are not performance evidence. Diagnostic commit e69c2009840006efd146b866648b593cbfe3ecc6 launches recursive-only run 34062350774, avoiding repetition of the long CoreMark measurement. These three experiments are currently pending/running. The call-flags runtime change is unmerged and no gain is claimed yet.

### First completed call-flags measurements (partial; no retained gain)

CoreMark run 34062227028 draw 2 (job 101564659542, AMD EPYC 7763) completed the native correctness and CoreMark measurement steps, then FAILED the subsequent recursive measurement because the workflow omitted the required expected-result argument. The whole job is a failure; only the independently completed earlier measurement is used below. Native correctness ran 554 core unit tests plus integrations, the unguarded call-status/division/five memory tests, and all 260 specification files. All seven source build/correctness warning audits report zero errors and zero warnings. Native BMI2 execution was not skipped. Draw 1 remains in progress.

The corrected recursive-only run 34062350774 draw 1 (job 101564988579, Intel Xeon Platinum 8573C) completed correctness, all build warning audits, recursive measurement with checked result 832040, and profiling. Its paired recursive-call change is neutral. Draw 2 has not yet been inspected as completed. Startup run 34062227036 (job 101564659394, AMD EPYC 7763) completed with three clean source build warning audits; measured startup throughput is lower on all three workloads.

- callflags-coremark-draw-2: candidate/parent +0.140508%; P(improvement) 83.630133%, P(regression) 16.369867%.

- callflags-recursive-draw-1: candidate/parent -0.054111%; P(improvement) 31.578881%, P(regression) 68.421119%.

- callflags-startup-coremark: candidate/parent -0.808147%; P(improvement) 0.251254%, P(regression) 99.748746%.

- callflags-startup-bz2: candidate/parent -0.321861%; P(improvement) 21.645449%, P(regression) 78.354551%.

- callflags-startup-ffmpeg: candidate/parent -0.576831%; P(improvement) 0.728821%, P(regression) 99.271179%.


Candidate 7a8e7f12 remains unmerged. These partial results provide no confirmed execution gain to include in the No.1 projection. Development recovery run 34062142640 is still measuring the JIT execution/startup shards and is not yet a completed gate. The last fully validated 92c1d465 projection remains unchanged; current HEAD recovery is pending.

### Call return flags rejected after completed measurements

The four native jobs have now been inspected. Both CoreMark draws in 34062227028 ran 554 core units plus integrations, unguarded calls/division/memory checks, 260 specification files, and seven source warning audits with zero errors/warnings. Both jobs subsequently FAILED the known recursive-CLI configuration step; these remain failed workflows, with only the earlier completed CoreMark measurements usable. Corrected recursive run 34062350774 completed both jobs and the same native correctness/warning checks. No native BMI2 execution check was skipped.

- CoreMark draw 1 / AMD 7763: +0.136853%; P(improvement) 97.397456%, P(regression) 2.602544%.

- CoreMark draw 2 / AMD 7763: +0.140508%; P(improvement) 83.630133%, P(regression) 16.369867%.

- Recursive draw 1 / Intel 8573C: -0.054111%; P(improvement) 31.578881%, P(regression) 68.421119%.

- Recursive draw 2 / AMD 7763: -2.646572%; P(improvement) 0.003370%, P(regression) 99.996630%.


The tiny CoreMark changes do not offset the AMD recursive regression and lower startup throughput (CoreMark -0.808147%, bz2 -0.321861%, ffmpeg -0.576831%). Candidate 7a8e7f12 is rejected and remains unmerged; there is no retained gain to add to the standings.


### Restored development revision: completed CI with CPU-specific evidence retained

Run 34062142640 at 20b01c3d completed all 34 jobs. Every job step and all primary/confirmation logs have been inspected. There are no failed steps or final soft-fail annotations after independent confirmation. The resolver log contains intentionally failing synthetic measurement tests; these are not runtime regressions. Node dependency deprecation messages remain visible; no source compiler warning was found.

The x64 execution primary ran on AMD EPYC 9V74. Spectralnorm is +2.973595% IMPROVEMENT; however fibonacci-iter is -4.772727% REGRESSION and tiny_keccak -4.127973% PLACEMENT. The independent fibonacci-iter confirmation ran on a DIFFERENT model, AMD EPYC 7763, and measured +1.027424% PASS. This resolves the workflow confirmation gate but does not establish that the 9V74 observation has disappeared. Do not call the primary regression fixed on 9V74 or discard it. This model also differs from the 7763 competitor anchor, so do not substitute its geomean into that projection.

ARM64 execution and startup ran on Neoverse-N2. Startup ERC20 is -0.957655% NEGLIGIBLE rather than the prior -2.00% REGRESSION. The x64 WASI primary reports LZ4 compress +0.06% PASS rather than the preceding -5.14% regression; this is recovery evidence from the current draw, not a same-host isolated causal measurement.


#### Recovery x64-execute

All 20 rows; geomean +15.286127% versus campaign main f73219f5.

| Metric | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.075977% | PASS |

| execute/counter-param | +0.105488% | PASS |

| execute/counter-global | +0.057153% | PASS |

| execute/fibonacci-rec | +2.145083% | IMPROVEMENT |

| execute/fibonacci-iter | -4.772727% | REGRESSION |

| execute/fibonacci-tail | +71.280712% | IMPROVEMENT |

| execute/sort | +44.255507% | IMPROVEMENT |

| execute/prime_sieve | +3.422461% | IMPROVEMENT |

| execute/matrix_mul | +1.180099% | PASS |

| execute/nbody | +2.557692% | IMPROVEMENT |

| execute/argon2 | +237.609192% | IMPROVEMENT |

| execute/tiny_keccak | -4.127973% | PLACEMENT |

| execute/mandelbrot | -0.004575% | PASS |

| execute/spectralnorm | +2.973595% | IMPROVEMENT |

| execute/compression | +0.244647% | PASS |

| execute/word_count | +1.113574% | IMPROVEMENT |

| execute/json_parse | +2.304641% | IMPROVEMENT |

| execute/reverse_complement | +92.543133% | IMPROVEMENT |

| execute/regex_redux | +0.068105% | PASS |

| execute/bulk-ops | -0.090818% | RECOVERED |


#### Recovery arm-execute

All 20 rows; geomean +4.688572% versus campaign main f73219f5.

| Metric | Delta | Status |
|---|---:|---|

| execute/counter-local | +0.013311% | PASS |

| execute/counter-param | -0.014992% | PASS |

| execute/counter-global | +0.011390% | PASS |

| execute/fibonacci-rec | +0.002557% | PASS |

| execute/fibonacci-iter | +0.003743% | RECOVERED |

| execute/fibonacci-tail | +0.010718% | PASS |

| execute/sort | +0.002179% | RECOVERED |

| execute/prime_sieve | +0.974283% | RECOVERED |

| execute/matrix_mul | -0.009145% | PASS |

| execute/nbody | -0.014215% | PASS |

| execute/argon2 | +152.021608% | IMPROVEMENT |

| execute/tiny_keccak | -0.167369% | NEGLIGIBLE |

| execute/mandelbrot | +0.004922% | PASS |

| execute/spectralnorm | +0.019961% | PASS |

| execute/compression | +0.023035% | PASS |

| execute/word_count | -0.032038% | RECOVERED |

| execute/json_parse | -0.590920% | NEGLIGIBLE |

| execute/reverse_complement | -0.529584% | RECOVERED |

| execute/regex_redux | -0.318406% | RECOVERED |

| execute/bulk-ops | -0.175654% | RECOVERED |


#### Recovery x64-startup

All 7 rows; geomean -0.974938% versus campaign main f73219f5.

| Metric | Delta | Status |
|---|---:|---|

| startup/bz2 | +1.001233% | PASS |

| startup/pulldown-cmark | -1.342217% | NEGLIGIBLE |

| startup/spidermonkey | -0.618154% | NEGLIGIBLE |

| startup/ffmpeg | -0.002240% | PASS |

| startup/coremark | -1.255824% | RECOVERED |

| startup/argon2 | -4.299722% | NOISY-FLOOR |

| startup/erc20 | -0.221767% | RECOVERED |


#### Recovery arm-startup

All 7 rows; geomean -0.341502% versus campaign main f73219f5.

| Metric | Delta | Status |
|---|---:|---|

| startup/bz2 | +0.975394% | IMPROVEMENT |

| startup/pulldown-cmark | -0.678830% | NEGLIGIBLE |

| startup/spidermonkey | +0.124591% | RECOVERED |

| startup/ffmpeg | +0.770346% | IMPROVEMENT |

| startup/coremark | -1.174278% | NEGLIGIBLE |

| startup/argon2 | -1.422326% | NEGLIGIBLE |

| startup/erc20 | -0.957655% | NEGLIGIBLE |


#### Recovery x64-confirm

All 1 rows; geomean +1.027424% versus campaign main f73219f5.

| Metric | Delta | Status |
|---|---:|---|

| execute/fibonacci-iter | +1.027424% | PASS |


### Dense table bounds candidate

Independent candidate e437baa5a22aef7acaf98eb2094694920fbf8d70 on codex/x64-table-index-bounds is based on restored development 20b01c3d. Its bounded backward graph walk follows GP copies, selects and every incoming block argument. Constants/masks/boolean definitions bound the possible low 32 bits. Copy cycles contribute no new bits but all external entries are visited. Unknown definitions, the implicit function-entry ABI edge, runtime/compiled calls, and budget exhaustion retain the clamp. This is a general range proof, with no function names, module hashes, benchmark constants or input-specific rewrites. The br_table index still passes through a 32-bit zero-extending copy even when the clamp is omitted.

Local default x64 557 units plus all integrations passed before adding the fifth proof test; a subsequent targeted run passed all five proof tests (558 total units now). ARM64 569 units plus all integrations passed. Unguarded x64 call-status, integer-division, five memory cases and two table integration tests passed. x64 specification runner passed all 260 files. Fmt, lint policy and source warning scans passed. The first local spec build failed because a fresh target lacked the pinned archive; the prerequisite was fixed by checking testsuite_version.txt and copying the existing pinned archive, then the build/run passed. The archive has no .git: do not use git rev-parse from inside it to verify its version, since that resolves the enclosing Nano worktree.

All 1,285 fixed-corpus MIR functions match be3db3a5. The only module with changed native code size is CoreMark, 17,484 -> 17,468 bytes. Its function 8 dense dispatch retains MOV r32 and indirect JMP but drops bound materialization, CMP and CMOVA. These are static observations, not a measured execution gain. Probe /tmp/sf-table-index-bounds-probe; dumps /tmp/sf-table-index-bounds-dump.

Diagnostic commit 3602f58ec46a89c27a7a3f1dc79642ade6e23a25 starts native CoreMark run 34064039213 (two draws) and startup run 34064039275. Both compare exact parent 20b01c3d with candidate e437baa5 plus campaign main f73219f5. They remain private codex/x64-profiling/manual workflows; PR/main CI is unchanged. Measurement helper tests: 27 passed. Candidate remains unmerged while native execution/startup results are pending.

Next independent hypothesis, not implemented: fibonacci-iter native loop has SUB r8,1 followed by CMP r8,0. The x64 flags tracker currently proves only 32-bit ZF. A width-aware extension could omit adjacent i64 Eq/Ne-zero compares after proven flag-setting ALU operations, while preserving comparisons after LEA/BMI2 shifts, zero-count shifts, calls, and block boundaries. It must distinguish low-32 zero from full-64 zero and retain ordering comparisons. This would be a general integer lowering optimization; it has not been timed or coded.


## Table-index bound proof: rejected after native measurements

CoreMark run 34064039213 completed both native jobs. Relative to restored parent 20b01c3d, e437baa5 is -6.135598% on AMD EPYC 9V74 (job 101569555326, P regression 99.999039%) and -3.335678% on AMD EPYC 7763 (job 101569555203, P regression 99.999583%). Both jobs completed native 559 core units plus integrations, unguarded memory/call/division/table tests, spec260, actual BMI2 execution, and all seven build/source warning audits with zero errors and warnings. Successful measurement steps do not make these performance regressions acceptable. The candidate is rejected, remains unmerged, and adds no gain to standings. Full-20 execution is not warranted for this candidate. Selected data are in tablebounds-coremark-draw-{1,2}.

Startup run 34064039275 job 101569555276 ran on AMD 7763. Candidate/parent throughput: CoreMark -0.594109% (P regression 86.277016%), bz2 -0.292043% (89.203188%), ffmpeg -0.017822% (53.272279%). All three build warning audits were clean; selected data are in tablebounds-startup-{coremark,bz2,ffmpeg}. These small startup changes do not offset the clear execution losses.

## Exact-width i64 result flags: isolated candidate, native measurement pending

Candidate 7fbd1a02c09da19155d3ddc7a4f2030d8448c745 on codex/x64-i64-result-flags has parent 20b01c3dba8b209228bc8ce9a64c92305847ab13. It extends the existing x64 zero-flag proof to track integer operand width and permits i64 equality/inequality against zero to reuse a proven ALU result. It retains explicit comparisons for ordering, mismatched widths, LEA, shifts, multiplication, calls and uncertain emission. A self MOV r32,r32 cannot preserve an i64 proof because zero-extension can change full-width zero. Store forwarding snapshots flags after materialization; block entries reset the proof. The change is limited to x64 lowering and tests; no MIR pass, register-bank or internal ABI change.

Local x64 556 units plus integrations, ARM64 569 plus integrations, x64 unguarded integer_sums/memory_length/call_status/integer_division, and x64 spec260 pass. Formatting, lint policy, diff whitespace, and source build warning checks pass. The first unguarded invocation named a nonexistent feature and stopped before compilation; the corrected --no-default-features --features jit,interp invocation passed. These ARM-host/Rosetta checks are correctness and static-code evidence, never native performance measurements.

All 1,285 fixed-corpus MIR functions are identical to the restored source baseline. Native code sizes change for fibonacci-iter 209 -> 205 bytes, regex_redux 1,731,613 -> 1,731,133, word_count 42,450 -> 42,402. Other 17 wasmi cases and exact suite CoreMark have unchanged sizes. Fibonacci-iter's loop now emits SUB r8,1 then JNE, omitting the redundant CMP r8,0. This is not a speedup claim.

Diagnostic commit ac68e1948c7f15ffbcf16a56ce5059ffc458931d launches full-20 paired Nano execution run 34065098529 (two draws) and startup run 34065098515 (CoreMark/bz2/ffmpeg). Both are confirmed in progress. Native BMI2, full core and spec tests, unguarded boundary tests, and warning audits precede execution timing. Measurement helpers: 26 tests pass; workflow YAML and embedded Bash parse, and only codex/x64-profiling push or manual triggers remain. PR/main/dev triggers and competitor engines are unchanged. Candidate remains unmerged and standings remain the conservative AMD 7763 projection recorded above until native results support retention.


### i64 result flags startup completed; full execution still pending

Run 34065098515 job 101572354803 completed on AMD EPYC 7763. Candidate/parent startup throughput is CoreMark +0.216422% (P improvement 76.802400%), bz2 -0.024743% (P regression 53.439177%), ffmpeg -0.071442% (P regression 68.532484%). No clear incremental startup change is established. All three source build warning audits have zero errors/warnings and every executed job step completed successfully. Selected artifact 9998779164 is saved in i64flags-startup-{coremark,bz2,ffmpeg}; complete archive is /tmp/sf-i64flags-startup.zip. Candidate/main is CoreMark -3.167210%, bz2 -0.496764%, ffmpeg +0.149249%; the inherited CoreMark startup cost remains visible. This is not evidence of full dev startup recovery.

Full execution run 34065098529 jobs 101572354758 and 101572354847 have completed their native candidate validation step and remain confirmed in-progress at the full-20 measurement step. No execution result or retained gain is available yet.

## Internal call-entry alignment: isolated native experiment

Source 6b223c9e963b7beb1d5c9d373b2c38112fc956cc on codex/x64-internal-entry-align is based on restored 20b01c3d. Three files add an architecture hook before internal-entry label binding in both materialized and template pipelines, with a no-op default. x64 emits up to 15 NOP bytes before that label to align the internal entry to 16 bytes. Public entry code and all internal callers resolve past the padding; x64 stack alignment, register allocation, entry guards and loop-header alignment policy are unchanged. Other architectures emit no extra instructions.

Important correction to the layout hypothesis: build.rs and page_align_function already align each FUNCTION start to 64 bytes. Native dumps concatenate function byte slices without inter-function arena gaps, so their packed file offsets are not module-relative runtime addresses. Shrinking one function cannot by itself change the following function's 16/32/64-byte phase. The new experiment concerns the internal entry's offset within its own public-entry-prefixed function, not that disproven cross-function phase hypothesis. It is also distinct from the previously rejected stack-shim omission and DFS loop-header selection.

Local x64 default 553 units plus integrations, ARM64 569 plus integrations, unguarded x64 call_status and memory_length (including template and mixed normal/template calls), and x64 spec260 pass without source warnings. Formatting, lint policy and whitespace checks pass. Static audit of all 1,285 fixed-corpus functions proves unchanged MIR, 16-byte internal-entry alignment relative to each 64-byte-aligned function start, NOP-only padding, and each public stub's actual CALL target beyond the padding. CoreMark native bytes 17,484 -> 17,650; regex_redux 1,731,613 -> 1,738,614. This code-size cost is not a measured speedup.

Probe /tmp/sf-internal-entry-align-probe, dumps /tmp/sf-internal-entry-align-dump, source /tmp/sf-x64-internal-entry-align. Diagnostic 6337ce2a6a93c3f0669e7a4141c14c911cacd528 launches CoreMark run 34065584875, jobs 101573653258 and 101573653402, and startup run 34065584862. These runs are confirmed live. CoreMark captures both parent and candidate profiles on each host. Native core/BMI2/unguarded/spec and warning audits precede timing. Five measurement-helper tests and YAML/Bash checks pass. Triggers remain isolated to codex/x64-profiling/manual and there is no concurrency cancellation of the older i64-flags experiment. Nothing is merged and standings remain unchanged.

### Next independently identified lowering opportunity: immediate multiply

Read-only inspection confirms x64 lower_int_binary Mul always materializes both operands and uses the register/register IMUL form. Generic three-operand IMUL with imm8/imm32 could avoid constant materialization and a distinct-destination copy. Eligibility must allow any low i32 multiplier but require sign-extended imm32 equivalence for an i64 multiplier; arbitrary u64 constants must keep the existing path. Flags remain untrusted for multiply. Fixed-corpus MIR contains constant multiplications in 14 modules, including 834 sites in 152 regex_redux functions, 84 in word_count, 57 in json_parse, and one cold/setup CoreMark function. Some large i64 constants are ineligible, so these counts are an upper bound. This opportunity is not yet implemented or measured and cannot be counted toward the goal.


## i64 result flags: full corpus completed, promoted to development validation

Run 34065098529 completed both native jobs with no failed steps. Native 557 core units plus integrations, unguarded arithmetic/memory/call/division tests, actual BMI2 execution and spec260 pass; all four correctness/build warning audit logs report zero errors and warnings. Archived harness logs contain no source warning/error or actual BMI2 skip. Source 7fbd1a02 is cherry-picked as c76c3765b33281a2713e7fecf369b6415dcf8c7c on dev/x64-hotpaths and pushed. Full dev run 34066095984 is confirmed queued/running; it is not yet a completed gate.

9V74 Fibonacci-iter +6.897656% is a real measured improvement; 7763 -0.072959% RECOVERED shows the benefit is model-specific. Word_count improves +0.796766% and +0.506249% on the two models. The 9V74 reverse_complement -1.649172% RECOVERED observation remains in the data. Its bulk-ops +8.496350% RECOVERED change is not repeated on 7763 (+0.147078%) and its guest code size/operations did not change; do not count that isolated large shift as an established optimization benefit. The conservative competitor projection is unchanged pending full dev evidence.

### i64flags draw 1: AMD EPYC 9V74

All 20 measurements, geometric throughput delta +0.685899% versus 20b01c3d.

| Case | Delta | Status |
|---|---:|---|
| execute/counter-local | -0.03% | PASS |
| execute/counter-param | -0.00% | PASS |
| execute/counter-global | -0.06% | RECOVERED |
| execute/fibonacci-rec | +0.20% | PASS |
| execute/fibonacci-iter | +6.90% | IMPROVEMENT |
| execute/fibonacci-tail | +0.05% | PASS |
| execute/sort | -0.01% | PASS |
| execute/prime_sieve | -0.25% | PASS |
| execute/matrix_mul | +0.00% | PASS |
| execute/nbody | +0.10% | RECOVERED |
| execute/argon2 | -0.03% | PASS |
| execute/tiny_keccak | +0.07% | PASS |
| execute/mandelbrot | +0.04% | PASS |
| execute/spectralnorm | -0.06% | PASS |
| execute/compression | +0.14% | PASS |
| execute/word_count | +0.80% | IMPROVEMENT |
| execute/json_parse | -0.22% | PASS |
| execute/reverse_complement | -1.65% | RECOVERED |
| execute/regex_redux | -0.22% | PASS |
| execute/bulk-ops | +8.50% | RECOVERED |
### i64flags draw 2: AMD EPYC 7763

All 20 measurements, geometric throughput delta +0.023171% versus 20b01c3d.

| Case | Delta | Status |
|---|---:|---|
| execute/counter-local | -0.23% | PASS |
| execute/counter-param | +0.04% | PASS |
| execute/counter-global | +0.05% | PASS |
| execute/fibonacci-rec | +0.03% | PASS |
| execute/fibonacci-iter | -0.07% | RECOVERED |
| execute/fibonacci-tail | +0.49% | RECOVERED |
| execute/sort | -0.12% | RECOVERED |
| execute/prime_sieve | +0.29% | RECOVERED |
| execute/matrix_mul | +0.08% | PASS |
| execute/nbody | +0.26% | PASS |
| execute/argon2 | -0.72% | RECOVERED |
| execute/tiny_keccak | -0.21% | RECOVERED |
| execute/mandelbrot | -0.06% | RECOVERED |
| execute/spectralnorm | +0.06% | PASS |
| execute/compression | +0.00% | PASS |
| execute/word_count | +0.51% | IMPROVEMENT |
| execute/json_parse | -0.02% | RECOVERED |
| execute/reverse_complement | -0.25% | PASS |
| execute/regex_redux | +0.20% | RECOVERED |
| execute/bulk-ops | +0.15% | PASS |


## Internal entry alignment: modest CoreMark benefit, full corpus pending

Draw 1, AMD EPYC 9V45: candidate/parent +0.536069% (P improvement 96.600085%).

Draw 2, AMD EPYC 7763: candidate/parent +1.176155% (P improvement 99.938047%).

Both native CoreMark jobs completed native core554 plus integrations, unguarded boundaries/calls, spec260 and seven source warning audits with zero errors/warnings. The two positive directions justify a full-corpus follow-up, but the 9V45 result is not conclusive and nothing is merged. Archives /tmp/sf-entryalign-draw-{1,2} include paired parent/candidate profiles and native dumps.

Startup run 34065584862 completed on AMD 9V74 with three clean build warning audits.

| Case | Candidate / parent | P regression |
|---|---:|---:|
| coremark | -0.205505% | 75.075538% |
| bz2 | -0.738760% | 85.814295% |
| ffmpeg | -0.361785% | 98.700569% |

Diagnostic b1ed7aec3f7c4844c83c916e0635e82f9c337c05 launches full-20 execution run 34066269048 (two Nano-only draws), confirmed queued. It retains parent20b01c3d and candidate6b223c9e, with native correctness/audits before timing. Startup workflow is unchanged and is not rerun. Full helper tests, YAML/Bash and isolated trigger checks pass.

## Immediate IMUL: isolated source validated, native timing not started

Candidate 47b529c2fe70d288bc45d56ac219159e86fd7a3c on codex/x64-imul-immediate is based on 20b01c3d, excluding both other candidates. x64 emits three-operand IMUL imm8/imm32 for an immediate multiplier in either operand order. All i32 low words qualify; i64 qualifies only if its bit pattern equals a sign-extended i32. Positive 0xffff_ffff and other large positive/negative i64 constants retain register multiplication. The existing flags proof is invalidated by instruction emission and multiply never publishes a ZF proof. Semantics follow the AMD64 Architecture Programmer Manual Volume 3 IMUL entry (https://docs.amd.com/v/u/en-US/24594_3.37 ; older AMD-hosted text https://community.amd.com/sdtpp67534/attachments/sdtpp67534/processors-discussions/29160/1/AMD64-3.pdf).

Two raw encoder tests execute both widths, signed immediate boundaries, extended register fields, all pairings of four volatile source/destination registers, overlapping operands and deterministic randomized full-width inputs. A separate WASM integration verifies both operand orders, live original inputs, wrapped products and zero branches across eligible and ineligible constants. Local x64 core555 plus integrations, ARM64 core569 plus integrations, unguarded integer_products/call_status/memory_length, and x64 spec260 pass without source warnings. Fmt, lint policy and whitespace pass. All1285 MIR functions are identical to baseline. json_parse shrinks101659→101305, regex_redux1731613→1726971, word_count42450→42309 bytes; CoreMark stays17484 bytes because local padding absorbs its cold multiply change. Source/probe/dumps are /tmp/sf-x64-imul-immediate, /tmp/sf-imul-immediate-probe, /tmp/sf-imul-immediate-dump. Source is pushed, unmerged and not timed; do not add gains to standings.


### Immediate IMUL native measurements submitted

Diagnostic6eebb34509a7fcf5adb84f8f502f853e3961f35d launches full-20 paired execution run34066396849 (two draws) and startup run34066396858 (CoreMark/bz2/ffmpeg), both confirmed queued. Source parent20b01c3d and candidate47b529c2 remain isolated from result-flags and entry-alignment changes. Native core/spec/BMI2/warning and unguarded integer_products/integer_sums/memory/call/division checks precede execution timing. All26 measurement-helper tests, YAML/Bash parsing, lint policy and whitespace checks pass. Only private profiling push/manual triggers apply; prior entryalign full20 run34066269048 remains queued and is not cancelled. Result-flags dev run34066095984 has17 primary jobs (1 completed resolver,8 running,8 queued) at the last inspected snapshot, so its full gate remains incomplete. No new competitor projection is claimed.


## Latest c76 dev primaries and pending experiment results

Dev run34066095984 at c76c3765 has finished its primary jobs. Its ARM JIT startup primary has CoreMark -1.98% and argon2 -1.75% REGRESSION; independent confirmation101578646715 is still live. Do not call the dev run passing. The other completed job steps and actual summaries have been inspected; resolver regression rows are measurement-tool test fixtures. All numeric declines remain included.


### c76-x64-execute: AMD EPYC 7763

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate c76c3765b33281a2713e7fecf369b6415dcf8c7c; complete 20-row throughput geomean +13.359287%.

| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | -0.099639% | PASS |
| execute/counter-param | +0.001720% | PASS |
| execute/counter-global | -1.427243% | RECOVERED |
| execute/fibonacci-rec | +2.320584% | IMPROVEMENT |
| execute/fibonacci-iter | +1.193631% | PASS |
| execute/fibonacci-tail | +69.752786% | IMPROVEMENT |
| execute/sort | +42.932365% | IMPROVEMENT |
| execute/prime_sieve | +6.737923% | IMPROVEMENT |
| execute/matrix_mul | +0.867388% | PASS |
| execute/nbody | +1.251242% | IMPROVEMENT |
| execute/argon2 | +155.936749% | IMPROVEMENT |
| execute/tiny_keccak | -4.707680% | PLACEMENT |
| execute/mandelbrot | -0.023264% | PASS |
| execute/spectralnorm | +0.069506% | PASS |
| execute/compression | +0.544628% | IMPROVEMENT |
| execute/word_count | +8.348143% | IMPROVEMENT |
| execute/json_parse | +2.890323% | IMPROVEMENT |
| execute/reverse_complement | +68.103861% | IMPROVEMENT |
| execute/regex_redux | -0.832847% | RECOVERED |
| execute/bulk-ops | -0.141411% | PASS |


### c76-arm-execute: Neoverse N2

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate c76c3765b33281a2713e7fecf369b6415dcf8c7c; complete 20-row throughput geomean +4.662405%.

| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | -0.012875% | PASS |
| execute/counter-param | -0.053424% | PASS |
| execute/counter-global | +0.038357% | RECOVERED |
| execute/fibonacci-rec | +0.019035% | RECOVERED |
| execute/fibonacci-iter | +0.007960% | PASS |
| execute/fibonacci-tail | -0.019752% | PASS |
| execute/sort | -0.839315% | PASS |
| execute/prime_sieve | -0.025066% | PASS |
| execute/matrix_mul | +0.018337% | PASS |
| execute/nbody | +0.030278% | PASS |
| execute/argon2 | +151.879687% | IMPROVEMENT |
| execute/tiny_keccak | +0.343707% | RECOVERED |
| execute/mandelbrot | +0.004445% | PASS |
| execute/spectralnorm | +0.020840% | PASS |
| execute/compression | -0.023886% | PASS |
| execute/word_count | -0.046736% | PASS |
| execute/json_parse | +0.377320% | RECOVERED |
| execute/reverse_complement | -0.562097% | RECOVERED |
| execute/regex_redux | -0.452958% | RECOVERED |
| execute/bulk-ops | -0.055189% | RECOVERED |


### c76-x64-startup: AMD EPYC 7763

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate c76c3765b33281a2713e7fecf369b6415dcf8c7c; complete 7-row throughput geomean -1.799448%.

| Metric | Throughput delta | Status |
|---|---:|---|
| startup/bz2 | -0.335935% | PASS |
| startup/pulldown-cmark | -2.010398% | RECOVERED |
| startup/spidermonkey | -0.388888% | RECOVERED |
| startup/ffmpeg | +0.364908% | RECOVERED |
| startup/coremark | -4.855147% | NOISY-FLOOR |
| startup/argon2 | -4.781689% | RECOVERED |
| startup/erc20 | -0.440323% | PASS |


### entryalign-full20-draw-2: AMD EPYC 7763

Baseline 20b01c3dba8b209228bc8ce9a64c92305847ab13; candidate 6b223c9e963b7beb1d5c9d373b2c38112fc956cc; complete 20-row throughput geomean -0.586990%.

| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | +0.164480% | PASS |
| execute/counter-param | +0.113003% | RECOVERED |
| execute/counter-global | +0.308425% | RECOVERED |
| execute/fibonacci-rec | -8.508782% | REGRESSION |
| execute/fibonacci-iter | -0.053936% | PASS |
| execute/fibonacci-tail | -0.055542% | PASS |
| execute/sort | +0.514829% | PASS |
| execute/prime_sieve | +0.432424% | PASS |
| execute/matrix_mul | -0.237916% | PASS |
| execute/nbody | +0.411526% | PASS |
| execute/argon2 | -0.073512% | RECOVERED |
| execute/tiny_keccak | +2.353375% | PASS |
| execute/mandelbrot | -0.012260% | PASS |
| execute/spectralnorm | -0.038222% | RECOVERED |
| execute/compression | -0.045068% | PASS |
| execute/word_count | +0.519748% | PASS |
| execute/json_parse | -1.166134% | PASS |
| execute/reverse_complement | -2.284174% | RECOVERED |
| execute/regex_redux | -0.017902% | PASS |
| execute/bulk-ops | -3.582963% | PASS |


### imul-full20-draw-1: AMD EPYC 9V74

Baseline 20b01c3dba8b209228bc8ce9a64c92305847ab13; candidate 47b529c2fe70d288bc45d56ac219159e86fd7a3c; complete 20-row throughput geomean +0.045718%.

| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | -0.263662% | PASS |
| execute/counter-param | -0.192345% | PASS |
| execute/counter-global | +0.128069% | RECOVERED |
| execute/fibonacci-rec | +0.092328% | PASS |
| execute/fibonacci-iter | +0.050754% | PASS |
| execute/fibonacci-tail | +0.085250% | RECOVERED |
| execute/sort | +0.027857% | PASS |
| execute/prime_sieve | +0.053908% | PASS |
| execute/matrix_mul | -0.046777% | RECOVERED |
| execute/nbody | +0.377217% | PASS |
| execute/argon2 | -0.454324% | RECOVERED |
| execute/tiny_keccak | +0.009145% | PASS |
| execute/mandelbrot | -0.025513% | PASS |
| execute/spectralnorm | +0.012760% | PASS |
| execute/compression | -0.192518% | PASS |
| execute/word_count | +1.176531% | PASS |
| execute/json_parse | -0.532750% | RECOVERED |
| execute/reverse_complement | +0.449572% | PASS |
| execute/regex_redux | -0.033225% | RECOVERED |
| execute/bulk-ops | +0.204063% | RECOVERED |


Entry alignment6b223c9e is rejected: full20 run34066269048 draw2 job101575454730 has a real -8.508782% fibonacci-rec regression on AMD7763, overwhelming the small CoreMark benefit. Correctness native554 units plus integrations, spec260 and four warning audits completed cleanly. Draw1 remains live; its data must still be collected. No merge and no gain added to standings.

IMUL47b529c2 run34066396849 draw1 job101575791994 (AMD9V74) has no confirmed full20 improvement: geo +0.045718%, word_count +1.176531% PASS below the full-family improvement threshold. Draw2 remains live. Native556 units plus integrations, unguarded integer_products, spec260 and four warning audits were clean. Candidate remains unmerged.

IMUL startup34066396858 completed on AMD7763 (job101575791866, artifact9999250644). Relative to parent: CoreMark +0.185321% (P improvement59.530573%), bz2 -0.216387% (P regression89.327637%), ffmpeg -0.228889% (P regression99.712032%). Versus campaign main: CoreMark -2.640963%, bz2 +0.208225%, ffmpeg +0.024306%. No startup improvement is established; negative observations remain real. Source build logs have no Rust warnings/errors.


## Native-frame values carried over single-predecessor edges

Independent source6201a516df192d4ecbef46e0ba8b8d44a8d3a16b is pushed on codex/x64-edge-frame-values, parent c76c3765. Source/worktree /tmp/sf-x64-edge-frame-values. A bounded MachineIR proof replaces a native GP frame load with the value from a matching predecessor store; it reuses an existing parameter or introduces one dead/free GP lane. Published stores and guest memory operations remain in place. Multiple predecessors, implicit entry, calls/opaque operations, frame changes, overlapping writes, source clobbers, width mismatch and register saturation reject the rewrite. It runs after compare-branch fusion removes dead temporary boolean definitions, before final index-extend relaxation. No benchmark-specific recognition. Unlike the old rematerialization candidate it carries the actual stored word instead of recomputing it.

Local final source: default x64 core566 and ARM64 core579 units plus integrations pass; x64 spec260, unguarded integer_sums/memory_length/call_status/integer_division pass; local CoreMark returns a positive F32 (Rosetta correctness only). Ten proof tests include randomized full64 words, both branch paths, parallel edge arguments, reused/clobbered parameters, aliasing/opaque barriers, saturation and native32 shape. Lint policy/fmt/diff checks pass. CoreMark frame loads452→428, native bytes17484→17513. Its func8/b25 no longer reloads frame72; an extra edge copy/jump carries the word via R14, so lower load count alone is NOT evidence of speed. Full21-module static summary is edgeframe-static-summary.json; native CI not yet submitted.

Additional JIT-only unit-test configuration (no-default-features + jit, x86_64-apple-darwin) exposes two existing shared decoder warnings: op_decoder.rs:300 predecode_fast_disabled and :322 disable_predecode_fast_for_test. Both reproduce on clean parent-equivalent7fbd1a02 in /tmp/sf-x64-i64-result-flags; only consumers are in the sf_interp predecoder and its tests. The default configuration is clean. This is an existing test/engine ownership cluster in a shared file, not fixed or suppressed; that JIT-only warning audit remains red. Evidence /tmp/sf-edgeframe-parent-jitonly-warning.log.


## Completed full20 rejection evidence and next native experiment


### entryalign-full20-draw-1: Intel Xeon Platinum 8370C

Baseline 20b01c3dba8b209228bc8ce9a64c92305847ab13; candidate 6b223c9e963b7beb1d5c9d373b2c38112fc956cc; full 20-row geomean -0.308084%.

| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | +0.016887% | PASS |
| execute/counter-param | -0.141206% | PASS |
| execute/counter-global | -0.036081% | PASS |
| execute/fibonacci-rec | -1.614212% | RECOVERED |
| execute/fibonacci-iter | -0.024978% | PASS |
| execute/fibonacci-tail | +0.507108% | PASS |
| execute/sort | -0.017701% | PASS |
| execute/prime_sieve | +0.392025% | PASS |
| execute/matrix_mul | -0.191656% | NEGLIGIBLE |
| execute/nbody | -0.141265% | RECOVERED |
| execute/argon2 | +0.879155% | RECOVERED |
| execute/tiny_keccak | -0.042341% | PASS |
| execute/mandelbrot | +0.004068% | PASS |
| execute/spectralnorm | -0.612089% | RECOVERED |
| execute/compression | +0.058619% | PASS |
| execute/word_count | +0.329821% | PASS |
| execute/json_parse | -1.123071% | RECOVERED |
| execute/reverse_complement | -0.022220% | PASS |
| execute/regex_redux | -4.649014% | REGRESSION |
| execute/bulk-ops | +0.397549% | RECOVERED |


### imul-full20-draw-2: AMD EPYC 7763

Baseline 20b01c3dba8b209228bc8ce9a64c92305847ab13; candidate 47b529c2fe70d288bc45d56ac219159e86fd7a3c; full 20-row geomean -0.101519%.

| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | +0.019899% | PASS |
| execute/counter-param | +0.175882% | PASS |
| execute/counter-global | +0.924416% | PASS |
| execute/fibonacci-rec | +0.027479% | PASS |
| execute/fibonacci-iter | +0.056062% | RECOVERED |
| execute/fibonacci-tail | +0.176204% | PASS |
| execute/sort | -3.501353% | REGRESSION |
| execute/prime_sieve | -0.220483% | RECOVERED |
| execute/matrix_mul | +0.095108% | RECOVERED |
| execute/nbody | -0.718560% | RECOVERED |
| execute/argon2 | -0.169280% | PASS |
| execute/tiny_keccak | -0.140295% | RECOVERED |
| execute/mandelbrot | +0.002796% | PASS |
| execute/spectralnorm | -0.093741% | RECOVERED |
| execute/compression | +1.260180% | IMPROVEMENT |
| execute/word_count | -0.046985% | PASS |
| execute/json_parse | +0.038563% | RECOVERED |
| execute/reverse_complement | -1.012985% | RECOVERED |
| execute/regex_redux | +1.484455% | RECOVERED |
| execute/bulk-ops | -0.293858% | RECOVERED |


### c76-arm-startup: Neoverse N2

Baseline f73219f56a70b77028f0d79730c7efca29ba3439; candidate c76c3765b33281a2713e7fecf369b6415dcf8c7c; full 7-row geomean -0.791843%.

| Metric | Throughput delta | Status |
|---|---:|---|
| startup/bz2 | +0.939757% | IMPROVEMENT |
| startup/pulldown-cmark | -1.389368% | NEGLIGIBLE |
| startup/spidermonkey | -0.675542% | NEGLIGIBLE |
| startup/ffmpeg | +0.402651% | IMPROVEMENT |
| startup/coremark | -1.976040% | REGRESSION |
| startup/argon2 | -1.745728% | REGRESSION |
| startup/erc20 | -1.062572% | NEGLIGIBLE |

Entry alignment full20 run34066269048 is terminal failure on both jobs: draw1 Intel8370C has regex_redux -4.649014% REGRESSION; draw2 AMD7763 has fibonacci-rec -8.508782% REGRESSION. Both core/spec/BMI2/unguarded and warning-audit steps passed. Candidate6b223c9e remains rejected and unmerged. All40 rows are now retained.

IMUL full20 run34066396849 is terminal failure: draw2 AMD7763 has sort -3.501353% REGRESSION, despite compression +1.260180% IMPROVEMENT; full20 geo -0.101519%. Draw1 AMD9V74 geo +0.045718% does not establish an improvement. Both native correctness/BMI2/spec/unguarded and warning audits passed. Candidate47b529c2 is rejected and unmerged; all40 rows are retained.

Cross-edge frame-value source6201a516 is now measured by diagnostic e7bfde39b8471556ee1b3cfa2fec7de9d04dff01: CoreMark run34067766341 (two native draws) and startup run34067766277 (CoreMark/bz2/ffmpeg). Both runs confirmed queued. Parentc76c3765 and exact source6201a516 are pinned. Five helper tests, Ruby YAML parsing, embedded Bash parsing, lint and diff checks passed. This does not change PR/main/dev triggers and does not invoke V8/Cranelift. Candidate remains unmerged; no runtime gain claimed.

Latest live handles: edgeframe CoreMark34067766341 draw2 job101579450673 in progress, draw1 job101579450774 queued; startup34067766277 job101579450417 queued. Devc76 run34066095984 has33 jobs, with ARM startup confirmation101578646715 still in progress and no final gate verdict. No duplicate runs were submitted.

Dev34066095984 is now a confirmed failure, not a soft pass: ARM Neoverse-N2 startup confirmation101578646715 fails its Cross-run verdict. CoreMark primary -1.976040%, confirmation -1.36%; argon2 primary -1.745728%, confirmation -1.70%. All34 jobs inspected; only this cross-run gate fails and requires correction. c76c3765 versus20b01c3d changes four x64-backend source files and integration tests only, so an ARM causal attribution to that flags change is not established. Investigate inherited JIT compilation cost; keep this failure visible.


## Current standings and edge-frame final verdict

Current retained dev source is c76c3765; its full-20 AMD EPYC7763 execution differential is +13.3592865984% versus campaign main f73219f5 (run34066095984). Its parent/main CoreMark control in run34067766341 draw1 is +8.7746166406% on AMD7763. Applying those Nano-only ratios to the frozen same-model three-engine anchors projects full20 throughput -1.333882% versus V8 and +9.516217% versus Cranelift; CoreMark +0.809433% versus V8 and -8.421654% versus Cranelift. Reaching parity needs a further +1.351915% full20 gain versus V8 and +9.196119% CoreMark gain versus Cranelift. These are cross-run projections, not refreshed same-host three-engine results, and the small CoreMark/V8 margin is not an established lead. Intel does not have an equally complete current projection. The full dev gate remains failed because independent ARM Neoverse-N2 JIT-startup confirmation reproduces CoreMark and argon2 regressions.

Edge-frame source6201a516 is rejected and unmerged. CoreMark run34067766341 measures -2.660057% versus parentc76 on AMD7763 (P regression99.946332%) and -3.515594% on Intel8573C (99.999997%). Startup34067766277 on Intel8573C measures CoreMark -2.011766%, bz2 -1.490251%, ffmpeg -1.993204%, each with strong regression evidence. Correctness and warning audits were clean, which does not excuse these measured regressions. Its gains are excluded from standings; no full20 follow-up is warranted for this version.

## Residency extraction scratch: local proof only, native CI still pending

Independent uncommitted worktree /tmp/sf-jit-residency-scratch, based on c76c3765, reuses extraction DP/order/backtracking storage across regions and reads persistent parent selection instead of cloning it for each child. No change to the objective, order comparator or tie breaking. Default ARM570 and x64557 unit suites plus integrations pass without warnings; the pinned 260-file specification suite also passes. Final MachineIR and per-function native code sizes match the parent across21 modules/1285 functions on each architecture. These checks establish structural equivalence for this corpus, not a native startup speedup.

Temporary ARM64 allocation instrumentation measures three identical counts per variant after warmup, without timing or benchmark execution. Instance construction/drop reduces total allocation-plus-reallocation calls by377 for CoreMark,328 for argon2 and2315 for bz2; requested allocation bytes fall by53014,46095 and1454543 respectively. Reallocations individually increase27,30 and39 and are included in the net counts. Exact argon2 startup module is res/rust/cases/argon2/out.wasm as specified by the pinned suite; CoreMark and bz2 use res/wasm. Native differential CI remains required before retaining this candidate or claiming recovery of ARM startup.


## Residency scratch native verdict: rejected

Sourcef61a0be1, diagnostic a52c7db4, run34068877810. Both native jobs completed without failed steps: ARM job101582444151 on Neoverse-N2, x64 job101582444305 on AMD7763. Every correctness/build audit reports zero errors/warnings; default units, unguarded checks and spec260 pass. Nonetheless scratch reuse does not restore startup and is rejected/unmerged. Local reductions in allocation count did not translate into a startup win. All four measured workloads remain included below.

| Architecture | Workload | scratch/parent | P(regression) | scratch/main |
|---|---|---:|---:|---:|
| arm64 | coremark | -0.452808% | 99.608449% | -1.021830% |
| arm64 | argon2 | -0.108740% | 94.622464% | -0.944176% |
| arm64 | bz2 | -0.768822% | 99.994535% | +0.612491% |
| arm64 | ffmpeg | -0.375885% | 99.817326% | +1.081590% |
| x64 | coremark | -0.057043% | 57.233740% | -2.856669% |
| x64 | argon2 | -0.327204% | 64.990776% | -3.283914% |
| x64 | bz2 | +0.389812% | 18.523897% | +0.726615% |
| x64 | ffmpeg | +0.647624% | 0.116891% | +0.549551% |

Full ZIPs /tmp/sf-residency-startup-{arm64,x64}.zip; artifact IDs9999982011 and9999975986. The prior dev ARM startup cross-run gate remains failed.

## Conditional fallthrough candidate submitted

Source70fdef74fe5539a5b855a9a4035e392945cebaaa on codex/x64-branch-fallthrough, parentc76c3765. Only x64 control lowering changes: emit parallel edge arguments after the condition on the physical fallthrough path; other edges keep their stubs. Equal targets share a path only with equal arguments. No moves are speculated before the branch. Executed proof covers both layouts and no fallthrough, distinct/shared targets and arguments, GP cycles, full64 payloads, source/destination/condition aliasing, constants, i32 value conditions, i64 comparisons and masked tests. Local default557 units plus integrations, five unguarded integration suites and pinned spec260 pass; four warning audits are0/0. Exact CoreMark module returns a positive F32 under Rosetta, used for correctness only.

All21 static module rows saved in branch-fallthrough-static.json. CoreMark native bytes17484→17446 and explicit body unconditional jumps93→89; fibonacci-iter205→184 bytes. Counts deliberately exclude edge/tail/table data for body branch totals. No timing inference from these reductions.

Diagnostic81eb16ddca6167cad53f27626b9f211ed5a91b99 starts CoreMark34069653637(two draws) and startup34069653623. Both currently live, unmerged, no claimed gain. The separate residency workflow is untouched; PR/main/dev triggers remain unchanged.

Next startup hypothesis: cache-layout analysis currently traverses a bank even if the whole function has no cached cells in that bank. Independent local uncommitted worktree /tmp/sf-jit-empty-cache-bank, parentc76, skips those walks. Native ARM profiles put compute_block_entry_cache_params at~5.15% of CoreMark compilation and~2.31% of argon2. Full generated-code equivalence and native measurements still required.


Empty-cache-bank candidate42396c9dba2da406348f4238b4f4dd673af25cd2 is now committed/pushed on codex/jit-empty-cache-bank, parentc76. The only production change is lower_cache_layout.rs: skip root assignment, exit simulation and GP join reconciliation for a bank with no cached cells; unreachable-block traversal marks the absent bank visited. Existing bank ordering/layout semantics are retained. Local ARM569/x64556 units plus integrations pass with zero warnings; the pinned260 specification files pass. Full MachineIR and per-function code sizes are identical across all21 modules/1285functions on both architectures. No native performance claim yet.

Diagnostic0e48dcb6c687ba4ef0e2eaa02159bbc398cdfb18 starts isolated dual-architecture run34070073615, confirmed live: ARMjob101585656849 and x64job101585657023. Sourcesmainf732/currentc76/emptybank42396 are pinned; exact CoreMark/argon2/bz2/ffmpeg startup workloads, both source checks and profiles are retained. No PR/main/dev trigger or competitor benchmark was added.


## Conditional fallthrough native verdict: not retained

Source70fdef74 remains unmerged. CoreMark34069653637 does not establish a repeatable gain: both default/native correctness, BMI2 execution, unguarded tests and spec260 are clean, and all seven audits per job report0 errors/0 warnings. Those correctness results do not prove performance benefit.

| Draw / CPU | fallthrough/parent | P(improvement) | parent/main |
|---|---:|---:|---:|
| 1 / AMD9V45 | +0.748863% | 84.390146% | +5.374019% |
| 2 / AMD7763 | -0.012396% | 46.703982% | +8.875751% |

Startup34069653623 on AMD7763 (artifact10000143541) also gives no supporting recovery. All three build warning audits are0/0; retain every workload, including the negative observations.

| Startup workload | fallthrough/parent | P(regression) |
|---|---:|---:|
| coremark | -1.051425% | 96.832983% |
| bz2 | -0.375718% | 79.335721% |
| ffmpeg | +0.057578% | 15.672830% |

CoreMark artifacts10000159410 and10000180175 are extracted under /tmp/sf-fallthrough-coremark-draw-{1,2}. No candidate gain is added to standings; no full20 run is warranted for this version.

A concrete remaining execution hypothesis comes from rejected6201a516 CoreMark MachineIR. In func8, b24 sends the new frame72 word r8 to a new b25 parameter r9, introducing an edge copy/stub. Its other edge b26→b28 never reads r9 and ends in ReturnScalar r6; the condition reads r6 and other outgoing args do not read r9. A generic bounded dead-register proof might allow the copy before the terminator, turn the new edge argument into identity, and eliminate the extra jump while retaining the forwarded frame word. This is unimplemented; the proof must cover all outgoing arguments/conditions, aliases, calls and longer paths, and it must not treat this particular function or benchmark as a condition. The previous full edge-frame pass also had measurable startup cost, so profitability and compilation cost must both be addressed.


## Empty-cache-bank native results and dev gate

Candidate42396c9d measured by diagnostic0e48dcb6 in run34070073615. Both native jobs are terminal, all actual steps succeeded, and all seven correctness/build warning audits per job report0/0. Source/default tests, native BMI2 on x64, unguarded checks and spec260 are clean. All four measured workloads on both architectures are retained below; this is startup throughput, not an execution speedup.

| CPU | Workload | emptybank/parent | P(improvement) | emptybank/main |
|---|---|---:|---:|---:|
| ARM Neoverse-N2 | coremark | +0.713551% | 99.896162% | -0.185489% |
| ARM Neoverse-N2 | argon2 | +0.346097% | 95.333666% | -0.754333% |
| ARM Neoverse-N2 | bz2 | +0.571633% | 97.733927% | +1.879676% |
| ARM Neoverse-N2 | ffmpeg | +0.240380% | 95.926104% | +1.319310% |
| AMD EPYC9V74 | coremark | +1.526886% | 99.828685% | -0.973833% |
| AMD EPYC9V74 | argon2 | +0.367976% | 87.157292% | -2.003421% |
| AMD EPYC9V74 | bz2 | +1.575498% | 99.989616% | +1.722799% |
| AMD EPYC9V74 | ffmpeg | +0.539654% | 99.997870% | +0.299626% |

Artifacts10000332327(ARM) and10000300462(x64), ZIPs /tmp/sf-emptybank-startup-{arm64,x64}.zip. The local generated-code equivalence and uniformly positive native points support promoting42396 to dev for full gating. dev/x64-hotpaths was fast-forwarded to the exact candidate commit and pushed; full performance-regression34071064270 is confirmed live. No final dev pass or ARM startup recovery is claimed before its individual steps and cross-run summaries finish.

Independent uncommitted edge-frame-without-stubs worktree /tmp/sf-edge-frame-without-stubs is based on rejected6201a516 plus a cherry-pick of42396 as4e3d9711. A new dead-register proof permits a prebranch word copy only when condition/all outgoing arguments retain their inputs and other short straight-line paths overwrite or never read that word. Calls, complex branches and paths beyond64 operations/blocks decline. Reserved bindings are not definitions. The copied conditional argument is identity, so this rewrite introduces no new edge stub. Initial14 proof tests and full x64570/ARM583 plus integrations pass; additional same-instruction read/write/address cases and spec/unguarded validation are being completed.

Actual x64 CoreMark func8 b24 now emits MOVr14,r10 before CMP/JNEdirect-b25. b25 loads the guest byte and stores the carried r14 directly to frame40, with no frame72 reload and no new edge jump. This is generated-code evidence only; native speedup remains unmeasured.


## Retained42396 final gate and current projections
Dev source42396c9dba2da406348f4238b4f4dd673af25cd2, run34071064270, is terminal. All34 jobs and their actual steps/logs were inspected: no failed step, Rust compiler warning, ACTION REQUIRED or SOFT-FAIL marker. Resolver REGRESSION rows belong to its118 passing helper tests. Every confirmation job reports primary flagged rows: none and skips independent remeasurement; these are not extra measurements. ARM startup CoreMark -0.420182% and argon2 -0.129826% are below the current gate, not measured speedups. The preceding c76 run remains a historical confirmed failure. The separate shared JIT-only unit-test decoder warning cluster is unchanged and is not covered by this clean default gate.

### dev423-x64-execute: AMD EPYC 7763
Full 20 rows, throughput geomean +13.615760% versus f73219f56a70b77028f0d79730c7efca29ba3439; source 42396c9dba2da406348f4238b4f4dd673af25cd2.
| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | +0.023622% | PASS |
| execute/counter-param | -0.056111% | PASS |
| execute/counter-global | -0.042671% | PASS |
| execute/fibonacci-rec | +2.821759% | IMPROVEMENT |
| execute/fibonacci-iter | +1.390714% | IMPROVEMENT |
| execute/fibonacci-tail | +71.181219% | IMPROVEMENT |
| execute/sort | +43.355955% | IMPROVEMENT |
| execute/prime_sieve | +6.060444% | IMPROVEMENT |
| execute/matrix_mul | +1.062257% | IMPROVEMENT |
| execute/nbody | +1.513323% | IMPROVEMENT |
| execute/argon2 | +157.752473% | IMPROVEMENT |
| execute/tiny_keccak | -4.565516% | PLACEMENT |
| execute/mandelbrot | +0.013114% | RECOVERED |
| execute/spectralnorm | -0.049658% | PASS |
| execute/compression | +0.997497% | PASS |
| execute/word_count | +8.015383% | IMPROVEMENT |
| execute/json_parse | +2.432783% | IMPROVEMENT |
| execute/reverse_complement | +69.810781% | IMPROVEMENT |
| execute/regex_redux | -0.916898% | NEGLIGIBLE |
| execute/bulk-ops | -0.108596% | PASS |

### dev423-arm-execute: Neoverse-N2
Full 20 rows, throughput geomean +4.684454% versus f73219f56a70b77028f0d79730c7efca29ba3439; source 42396c9dba2da406348f4238b4f4dd673af25cd2.
| Metric | Throughput delta | Status |
|---|---:|---|
| execute/counter-local | +0.044089% | PASS |
| execute/counter-param | -0.029513% | PASS |
| execute/counter-global | +0.008469% | PASS |
| execute/fibonacci-rec | +0.043532% | PASS |
| execute/fibonacci-iter | -0.036129% | PASS |
| execute/fibonacci-tail | -0.008182% | PASS |
| execute/sort | +0.071100% | PASS |
| execute/prime_sieve | -0.005036% | PASS |
| execute/matrix_mul | -0.014533% | PASS |
| execute/nbody | +0.006232% | PASS |
| execute/argon2 | +153.638314% | IMPROVEMENT |
| execute/tiny_keccak | -0.144556% | NEGLIGIBLE |
| execute/mandelbrot | +0.038327% | RECOVERED |
| execute/spectralnorm | +0.033470% | PASS |
| execute/compression | -0.066375% | PASS |
| execute/word_count | -0.035223% | RECOVERED |
| execute/json_parse | -0.246947% | RECOVERED |
| execute/reverse_complement | -0.603851% | NEGLIGIBLE |
| execute/regex_redux | -0.434099% | RECOVERED |
| execute/bulk-ops | -0.130443% | RECOVERED |

### dev423-arm-startup: Neoverse-N2
Full 7 rows, throughput geomean +0.485669% versus f73219f56a70b77028f0d79730c7efca29ba3439; source 42396c9dba2da406348f4238b4f4dd673af25cd2.
| Metric | Throughput delta | Status |
|---|---:|---|
| startup/bz2 | +1.972974% | IMPROVEMENT |
| startup/pulldown-cmark | +0.025971% | PASS |
| startup/spidermonkey | +0.244655% | PASS |
| startup/ffmpeg | +1.685743% | IMPROVEMENT |
| startup/coremark | -0.420182% | RECOVERED |
| startup/argon2 | -0.129826% | RECOVERED |
| startup/erc20 | +0.046664% | PASS |

### dev423-x64-startup: Intel Xeon Platinum 8573C
Full 7 rows, throughput geomean -0.883025% versus f73219f56a70b77028f0d79730c7efca29ba3439; source 42396c9dba2da406348f4238b4f4dd673af25cd2.
| Metric | Throughput delta | Status |
|---|---:|---|
| startup/bz2 | -1.319991% | RECOVERED |
| startup/pulldown-cmark | +0.150702% | PASS |
| startup/spidermonkey | -0.068039% | RECOVERED |
| startup/ffmpeg | -0.501628% | RECOVERED |
| startup/coremark | +0.366904% | RECOVERED |
| startup/argon2 | -2.300500% | PASS |
| startup/erc20 | -2.467393% | RECOVERED |

Current7763 projection uses the latest retained-source measurement, including all20 negative and placement-classified rows. The full20 Nano/main geomean is +13.615760%. With frozen three-engine anchors34026700835 and34028386243, full20 Nano throughput is -1.110651% versus V8 and +9.763996% versus Cranelift; CoreMark is +0.377623% versus V8 and -8.813919% versus Cranelift. CoreMark uses the retained parent42396 control in draw2 of34071886761 (+8.308692% versus campaign main), not the unretained forwarding candidate. Small differences from the preceding standings are measurement variation;42396 preserves generated execution code. All20 per-engine projections are in standings-42396-epyc7763.json. No Intel projection or confirmed cross-engine win is inferred.

## Frame forwarding without conditional stubs: native result, unretained
Final source178cc0896f8d0e5d63034518664b78c692336b33 is on codex/edge-frame-without-stubs, direct parent42396. It supersedes the temporary uncommitted history mentioned above; the final two-file diff is864 insertions including14 proof tests. Diagnostic77f3baa8e3fe23e7e78aae1bfce7bbf95b0ad7fd runs isolated CoreMark34071886761 and dual-architecture startup34071886775, four jobs total. All actual steps passed, all seven correctness/build audits per job are0 errors/0 warnings, and all260 spec files pass. Native x64 BMI2 and unguarded checks are exercised. This does not add competitor builds or alter PR/main/dev triggers.
Local all21 static output is edge-nostub-static.json: CoreMark bytes17484→17516, frame loads452→430, stores479→479, body JMP93→93 and Jcc254→254. Its hot frame72 reload is gone without the previous conditional edge trampoline. These structural facts are not timing evidence.
| CoreMark draw / CPU | candidate/parent | P(improvement) | parent/main |
|---|---:|---:|---:|
| 1 / AMD EPYC 9V74 | +0.145910% | 90.746381% | +5.561445% |
| 2 / AMD EPYC 7763 | +0.619081% | 99.572432% | +8.308692% |

| Startup CPU | Workload | candidate/parent | P(regression) | candidate/main |
|---|---|---:|---:|---:|
| Neoverse-N2 | coremark | -0.780752% | 99.997440% | -1.233343% |
| Neoverse-N2 | argon2 | -1.042670% | 99.472531% | -1.803349% |
| Neoverse-N2 | bz2 | -0.577501% | 99.914207% | +1.425297% |
| Neoverse-N2 | ffmpeg | -0.664224% | 99.982422% | +0.787503% |
| Intel Xeon Platinum 8573C | coremark | -0.534121% | 88.573721% | -2.634875% |
| Intel Xeon Platinum 8573C | argon2 | -1.644758% | 97.475967% | -2.784061% |
| Intel Xeon Platinum 8573C | bz2 | -2.024150% | 99.987589% | -2.188854% |
| Intel Xeon Platinum 8573C | ffmpeg | -1.372880% | 99.999992% | -2.397479% |

Candidate178 remains unmerged and is excluded from standings: modest CoreMark points are accompanied by startup regressions on both architectures. No full20 run is justified for this exact version before its compilation cost is addressed. Native profiles suggest testing a narrower cleanup after Load→Move instead of rerunning all of optimize_block; that possible follow-up is unimplemented and needs code-equivalence/correctness proof and fresh native measurement. Core artifacts10000895650/10000896618 and startup artifacts10000914254/10000891368 are preserved under /tmp/sf-nostub_{core1,core2,startup_arm,startup_x64}.zip.


## Lean frame-forwarding cleanup: native measurement submitted

Source a09afd24b9668bb3fce885c26730feafd8baf880 on codex/edge-frame-lean-cleanup, worktree /tmp/sf-edge-frame-without-stubs, extends unretained178cc089. Relative to retained42396 it includes the generic forward pass; it does not promote178 independently. A 20-insertion/5-deletion follow-up in its single pass file checks predecessor availability before scanning the target prefix and replaces complete optimize_block with ordered copy propagation, stored-value forwarding and instruction selection. The early copy-only draft lost frame-forwarding opportunities in three modules; the final ordered cleanup preserves exact MachineIR and per-function native sizes across all21 modules/1285 x64 functions versus178. Full evidence edge-lean-x64-equivalence.json. This is not a timing result or a claim of universal code equivalence.

Final local ARM583/x64570 core units plus all integrations pass, as do the five unguarded integration suites and pinned spec260. Seven build/correctness warning audits are0/0. Exact suite CoreMark returns a positive F32 under Rosetta (correctness only). Fmt, lint and diff checks pass. No new suppression, cfg exception, or benchmark-specific rewrite. Final logs use /tmp/sf-edge-lean-selected-*.log; initial/final/ordered temporary prefixes are superseded drafts.

Diagnostic01454162476f03680911f15208bc850321bfcbc7 pins mainf732, retained-parent42396 and candidatea09afd24. CoreMark run34073930157: jobs101596317099(draw2),101596317269(draw1). Dualstartup run34073930142: jobs101596317189(x64),101596317295(ARM). Both runs are confirmed live. The workflow edits affect only the diagnostic branch, with no competitor benchmark or daily PR/main/dev change. Five helper tests, YAML/shell parsing, pinned refs and trigger checks passed. Candidate remains unretained until native performance/startup and then full20/ARM execution gates support it.

## New code-reference evidence for loop residency

A local pinned Wasmtime47.0.2 CLI can cross-compile the exact CoreMark module to x64 in under a second; no extra competitor CI or benchmark was started. Its --target x86_64-unknown-linux-gnu and Cranelift znver3 preset produce the CLIF/ELF/disassembly archived under cranelift47-coremark-znver3. This is a structural reference with a CPU preset, not the exact anchor's emitted code and not timing evidence. Compiler settings and module hash are retained.

The retained Nano hot func5/b18 (about8.7% of samples in the preceding native AMD profile) reloads frame104 stride and frame88 loop counter, and stores frame88 every iteration. Its MachineIR carries r10, loaded from frame0 in b56, unchanged through b18 to use on exit b58. The matching Cranelift inner loop at text1e56..1e91 keeps counter and stride inputs in registers and has no stack access inside that loop. Nano's r10 is not a free register; it is a live-through value, so simply overwriting it would be wrong. Existing reuse_loop_frame_values runs before cache_loop_frame_words, giving exit-only reload avoidance first access to spare lanes; the later cache pass requires two static reads and excludes the single-read updated counter. This establishes a concrete generic policy hypothesis, not proof that reordering is faster.

Independent uncommitted prototype /tmp/sf-loop-cache-priority, branch codex/loop-cache-priority, starts at retained42396. It moves exit-only frame reuse after loop-word caching and permits a single-read word only when the loop also writes that exact full native GP word. Read-only single-read caching remains excluded; the previous broad single-read and lowering-reserve candidates stay rejected. The prototype has not passed tests or native timing and is not submitted. First check whether its emitted loop actually removes the identified memory recurrence, then validate semantics and full corpus before considering CI. Its temporary probe is /tmp/sf-loop-cache-priority-probe; initial build is live.


## Lean frame forwarding: final native result, rejected

Runs34073930157 and34073930142 are now complete, superseding their live status above. All four jobs passed their actual correctness/build steps with seven zero-error/zero-warning audits per job. Performance is evaluated separately: all eight startup rows regress, so a09afd24 remains unmerged and excluded from standings. No full20 run is warranted for this version. Exact comparisons, CPU/source manifests and summaries are archived under lean-core1, lean-core2, lean-startup-arm and lean-startup-x64.

| Artifact | CPU | Workload | candidate/parent throughput | P(regression) | parent/main |
|---|---|---|---:|---:|---:|
| lean_core1 | Intel(R) Xeon(R) 6973P-C | coremark-out | +0.881505% | 28.124059% | +17.202234% |
| lean_core2 | AMD EPYC 7763 64-Core Processor | coremark-out | +0.276061% | 3.080875% | +9.303222% |
| lean_startup_arm | Neoverse-N2 | argon2-results | -0.882794% | 99.998175% | -0.969808% |
| lean_startup_arm | Neoverse-N2 | bz2-results | -0.830722% | 99.999004% | +1.730699% |
| lean_startup_arm | Neoverse-N2 | coremark-results | -0.691801% | 99.999293% | -0.585315% |
| lean_startup_arm | Neoverse-N2 | ffmpeg-results | -1.073463% | 99.999727% | +1.701598% |
| lean_startup_x64 | AMD EPYC 7763 64-Core Processor | argon2-results | -1.304305% | 95.710536% | -1.377889% |
| lean_startup_x64 | AMD EPYC 7763 64-Core Processor | bz2-results | -1.860403% | 99.916622% | +0.949849% |
| lean_startup_x64 | AMD EPYC 7763 64-Core Processor | coremark-results | -0.911526% | 97.986475% | -2.050708% |
| lean_startup_x64 | AMD EPYC 7763 64-Core Processor | ffmpeg-results | -1.212263% | 99.988468% | +0.424680% |

The additional retained42396 CoreMark control on AMD7763 is +9.303222% versus main, compared with +8.308692% in the preceding control. The source is unchanged; this spread is measurement variation, not another retained optimization. Using the same frozen anchor implies a roughly8–9% Cranelift deficit, rather than a precise new win. Intel6973P-C is not comparable to the AMD anchor.


## Loop cache priority: validated source and native submission

Source3388ae6786e74306849431b548ca84d9fd350666 on codex/loop-cache-priority is a direct child of retained42396, with two source files changed. The cache policy now allows a single full native-frame read if the loop also stores that exact full word, and schedules loop caching before exit-only frame reuse. Write detection shares the existing alias scan. Single-read read-only words remain excluded; alias barriers, lowering-reserved lanes, frame publication and exit reads are preserved. There are no benchmark identifiers, new fusion instructions or lint/cfg exceptions.

The old rejection fixture6 was a full load plus a narrow load and a full store, previously rejected solely by the two-full-read profitability threshold. It is now covered by an independent small MachineIR evaluator: compare every published store across8 narrow-observer/order cases and12 iteration/seed combinations, including high halves and overflow. Additional tests reject one-read read-only caching and show that both pass orders preserve the exit-only value while the new order uses the spare lane for the loop word. The remaining seven safety-rejection cases stay unchanged.

Final local ARM572 core units/625 total tests and x64559 core units/612 total tests pass (19 suites each,4 existing ignored tests per architecture), plus12 tests in5 unguarded integration suites and pinned260 specification files. Seven compilation/correctness audits are0 errors/0 warnings. Fmt/lint/diff checks pass. Logs /tmp/sf-loop-cache-priority-*-final.log supersede earlier failing draft logs. The earlier draft's exact CoreMark run returned a positive F32 under Rosetta (correctness only); the final production scan consolidation preserves all21 x64 generated MachineIR functions against that draft.

Final all21 structural dumps contain1285 functions per architecture, changing82 x64 functions and18 ARM functions against retained42396. The x64 CoreMark total native buffer is17484→17488 bytes; the target func5/b18 counter frame88 reload is removed while the frame104 stride load remains. This has no measured performance meaning until native CI finishes. Detailed changed-function IDs/sizes are in loop-cache-priority-static.json; raw dumps are /tmp/sf-loop-cache-priority-final-{x64,arm}-dump.

Diagnosticd113f973a1f545b9814e5083644e7934a116b611 submits CoreMark34075452274 and startup34075452287, four jobs total, using pinned Rust1.98.1, parent42396 and mainf732. Both workflow runs are confirmed queued/running. The five helper tests and YAML/shell/source/trigger checks pass; only the dedicated diagnostic branch triggers these workflows. No V8/Cranelift build or PR/main/dev trigger was added. Candidate is unretained pending native measurements and the full20/ARM execution gates if justified.


### Follow-up register-pressure evidence, not an implemented optimization

The retained42396 control in lean CoreMark draw2 (AMD7763) attributes11.48% of samples to func2/b11,10.70% to func8/b25,9.35% to func10/b0 and8.37% to func5/b18. The Nano list reversal/search loops are already compact compared with the local Cranelift znver3 reference; instruction count alone does not establish their relative speed. The matrix loop remains a more concrete frame-traffic target.

Prepared SSA func5/b18 reads old cached c7 into v364, writes temporary product v363 into c7, reads it twice for independent bit extracts, then overwrites c7 with final accumulated v376. Lowered MIR therefore moves old accumulator r6 to r8 and product r7 to r6; the second extraction uses r9 while r4/r5/r10 are live through the loop. This uses an extra physical lane compared with retaining old accumulator in r6, keeping product in r7, extracting first to now-dead r8 and extracting second destructively into r7. A future generic local register-allocation/renaming pass could investigate that reuse. This is not benchmark recognition and no pass has been implemented or timed.

Directly forwarding both CellGetCache values to product SSA is not a valid small fix: middle/ssa_ir/validate.rs enforces at-most-one operation use of each linear SSA value, while the cache getters encode repeated reads. Do not relax that validator or planner discipline to obtain a smaller example. A MachineIR rewrite would need independent value-lifetime proof, preserved block-entry/exit registers, barriers/cached publication correctness and native measurements. The previously rejected destructive linear-input coalescer did not handle cached owners or this case.


## Loop cache priority: final native result, rejected

Both34075452274 and34075452287 are terminal, superseding the live status above. All four actual jobs and their correctness steps pass; each has seven zero-error/zero-warning audits and260 spec files passed. No actual soft-fail, action-required warning, compiler diagnostic or skipped native BMI2 execution is present. Performance is a separate decision: both CoreMark draws regress and all four x64 startup points are negative, so3388ae67 remains unmerged, excluded from standings and will not proceed to full20 measurement in this form. ARM startup improves modestly, but does not rescue the execution regressions.

| Artifact | CPU | Workload | candidate/parent throughput | P(regression) | parent/main |
|---|---|---|---:|---:|---:|
| core1 | AMD EPYC 9V74 80-Core Processor | coremark-out | -0.560971% | 99.486890% | +5.114168% |
| core2 | INTEL(R) XEON(R) PLATINUM 8573C | coremark-out | -0.417025% | 98.496533% | +10.092680% |
| startup_arm | Neoverse-N2 | argon2-results | +0.450807% | 0.212547% | -1.331688% |
| startup_arm | Neoverse-N2 | bz2-results | +0.410617% | 3.999456% | +1.557489% |
| startup_arm | Neoverse-N2 | coremark-results | +0.449104% | 0.031201% | -0.705647% |
| startup_arm | Neoverse-N2 | ffmpeg-results | +0.164407% | 5.500048% | +1.430231% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | argon2-results | -0.549981% | 79.078221% | -2.044651% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | bz2-results | -1.447225% | 99.947031% | +1.177723% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | coremark-results | -0.567648% | 83.500921% | -1.638603% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | ffmpeg-results | -0.740447% | 99.886638% | +0.762654% |

Exact artifacts are archived under loop-priority-core1, loop-priority-core2, loop-priority-startup-arm and loop-priority-startup-x64. Original ZIPs/dumps/profiles remain in /tmp/sf-priority-{core1,core2,startup_arm,startup_x64}.zip and matching directories. Removing a frame reload was insufficient: the extra write-through register copies, exit reloads and other changed loops still require cost/placement assessment. A general block-local register reuse investigation is now better motivated than widening the rejected single-read cache policy again. No new candidate is implemented or live at this point.


## GP temporary reuse: validated independent candidate

Source8f69395be2f1a2c652b9ade0ba17b5f637e4ed74 on codex/gp-island-reuse, worktree /tmp/sf-gp-island-reuse, is a direct child of retained42396. It does not include rejected loop-cache-priority3388. A new MachineIR pass virtualizes value identities inside bounded pure-integer fragments containing cached and linear register moves, treats full native-word copies as aliases, donates dead inputs to results, and restores every live output and owner before any observer. Only registers already defined in the original fragment may be written; fixed registers, lowering reserve, memory, calls, division/remainder, noninteger operations and other unsupported instructions end a fragment. GP-word64 and zero-extending32-def capability predicates exclude pair and sign-extending-word backends without new cfgs. Parallel boundary-copy cycles use only an originally written dead lane; insufficient scratch or no net reduction in emitted instructions abandons the rewrite. No new fusion opcode or benchmark identity is recognized.

Five meaningful proof tests cover old cached snapshots, full64 payloads/I32 overflow, parallel output-copy cycles, frame publication/CFG consumers, capability/fixed/reserved exclusions, and512 generated alias/lifetime programs including arithmetic, variable shifts/rotates, bit tests, signed/unsigned comparisons and shifted operands. Accepted rewrites are compared against an independent evaluator with8 arbitrary input assignments each and output ownership/physical-clobber assertions. The initial proof and final expanded proof pass.

Final local ARM574 units/627 total tests and x64561 units/614 total tests pass in19 suites each (4 existing ignored each). Five unguarded suites pass12 tests, and pinned spec260/260 passes. Seven build/correctness warning audits are0/0. Fmt, lint (including staged new file), and diff checks pass. Final logs /tmp/sf-gp-island-*-final.log; the initial probe's exact CoreMark returned a positive F32 under Rosetta, correctness only. Deferring liveness allocation until an eligible fragment exists preserves exact generated x64 MIR across all21 modules against that initial probe.

All21 final dumps contain1285 functions on each architecture:31 x64 and55 ARM functions change. CoreMark x64 total native buffer remains17484 bytes because of layout/padding; that does not imply its hot regions are unchanged. The func5/b18 prefix loses three native register MOVs and stops using MIRr9/physicalR14 as a temporary. Product stays in its source lane until the second extraction consumes it, while the old accumulator remains in its original register. Both existing frame104 stride and frame88 counter loads remain. No execution or startup improvement is claimed from these structural results. Full changed-function IDs/native sizes are in gp-island-reuse-static.json, raw outputs /tmp/sf-gp-island-final-{x64,arm}-dump.

Diagnostic0f25f06962bd5afc85a9280344995d599feedac3 changes only the two dedicated workflow files, using Rust1.98.1, parent42396, candidate8f69395b and mainf732. Five measurement-helper tests plus YAML/shell/source/trigger checks pass. CoreMark two-draw and dual-architecture startup runs have been submitted; no competitor build or daily PR/main/dev trigger was added. Candidate remains unretained pending native results and full20/ARM execution gates if justified.

Native GP reuse submission is now confirmed running: CoreMark34077414285 jobs101606144602(draw1) and101606144696(draw2); startup34077414291 jobs101606144688(ARM) and101606144797(x64). Actual candidate correctness steps have passed on all four jobs; the workflows remain live, so final warning/log and performance conclusions are pending. All local exec/build/test sessions for this source have completed, and both source/diagnostic branches are pushed and clean.


## GP temporary reuse: final native result, rejected

Both runs 34077414285 and 34077414291 are terminal, superseding the live status above. All four actual jobs and steps succeeded; each has seven zero-error/zero-warning audits and 260 spec files passed. There is no actual soft-fail, action-required compiler warning or native BMI2-execution-skipped marker. See gp-island-reuse-ci-audit.json. These correctness checks do not establish performance acceptance.

Both CoreMark draws landed on AMD EPYC 9V74: candidate/parent is -0.016909% and -0.202342%, with P(regression) 55.669328% and 96.004586%. There is no demonstrated execution gain. All eight startup points regress, including x64 drops of 1.50–2.00% and ARM drops of 0.49–0.78%. Candidate 8f69395b is rejected, remains unmerged and does not enter the full20/ARM execution gate in this form. Retained 42396 standings remain unchanged. Fewer register moves did not establish a runtime benefit.

| Artifact | CPU | Workload | candidate/parent throughput | P(regression) | parent/main |
|---|---|---|---:|---:|---:|
| core1 | AMD EPYC 9V74 80-Core Processor | coremark-out | -0.016909% | 55.669328% | +5.436195% |
| core2 | AMD EPYC 9V74 80-Core Processor | coremark-out | -0.202342% | 96.004586% | +5.557157% |
| startup_arm | Neoverse-N2 | argon2-results | -0.684928% | 99.992939% | -0.738500% |
| startup_arm | Neoverse-N2 | bz2-results | -0.492741% | 99.999422% | +1.900433% |
| startup_arm | Neoverse-N2 | coremark-results | -0.780122% | 99.998611% | -0.567598% |
| startup_arm | Neoverse-N2 | ffmpeg-results | -0.643815% | 99.729249% | +1.675833% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | argon2-results | -1.501077% | 93.990212% | -2.549658% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | bz2-results | -1.862929% | 99.954589% | +1.921502% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | coremark-results | -2.003054% | 99.890784% | -1.333754% |
| startup_x64 | AMD EPYC 7763 64-Core Processor | ffmpeg-results | -1.496720% | 99.998875% | +0.968984% |

Exact results are archived under gp-reuse-core1, gp-reuse-core2, gp-reuse-startup-arm and gp-reuse-startup-x64; all ten points are in gp-island-reuse-native-verdict.json. Original ZIPs, native dumps and profiles remain in /tmp/sf-reuse-{core1,core2,startup_arm,startup_x64}.zip and the matching directories. No candidate CI remains live at this point. Function-inlining/call-overhead investigation is read-only so far; no replacement candidate has been implemented.


## Small-call inlining: independent candidate submitted

Source 7fbc510b0fc6edd06d2bd926c6ba0e49d4b3f4ee on codex/jit-call-inlining, worktree /tmp/sf-jit-call-inlining, is a direct child of retained 42396. It does not include the rejected GP reuse pass. The JIT semantic stage now expands small straight-line local function bodies before either static ABI frame summaries or actual preparation. Parameters are evaluated once and consumed in reverse stack order into fresh locals; non-parameter scalar locals are explicitly initialized at every inline site. Nested ordinary direct/indirect/ref calls remain calls and are not recursively expanded. All original caller control targets, including branch tables/reference branches/EH catch targets, and dynamic result-type rows are relocated. Imported/linked functions have no local spec and are excluded. Callees with structured control, early return with extra stack values, nonnumeric local/result types or unsupported termination are excluded.

Limits are 32 semantic callee operations, 8 callee locals, 256 conservatively counted extra operations and 64 extra locals per caller, plus checked u16 frame arithmetic. A 128-byte callee filter avoids decoding obviously large bodies. A per-caller cache decodes each small callee once, and expansion only visits the original caller operations. Finite compiler-RAM-budget configurations retain their original decode/frame-size policy; the new pass applies with the default unbounded compiler budget. No machine ABI, register bank, fusion opcode, lint exception or feature ownership boundary was changed.

Three unit tests verify single-level expansion with nested call type rows, overflow/control exclusions, and actual decoded call elimination with matching static/prepared frame sizes. Three integration tests exercise parameter side-effect order, repeated zero initialization in a loop, mixed multi-value results with a lower live operand, branch-table/if relocation, host failures, out-of-bounds memory and signed division traps before continuation side effects. The current final local ARM suite passes 572 core units / 628 total tests; x64 passes 559 core units / 615 total tests, with 20 suites and 4 existing ignored tests each. Six unguarded suites pass 15 tests, and the pinned specification suite passes 260/260. Eight audited current compilation/correctness logs contain zero errors/warnings. Earlier /tmp/sf-inline-arm-core.log contains a test-only draft compile error, fixed before the successful full suites. Fmt/lint (including staged new files)/diff checks pass. A full exact-CoreMark invocation returns a positive F32 under Rosetta, correctness only.

Final structural dumps contain 21 modules / 1285 functions per architecture. Both architectures change only CoreMark functions 1,2,4,5,7,14,15 and json_parse function 12. CoreMark x64 native bytes rise 17484→19686 and ARM bytes rise 15364→17060; json_parse x64 101659→102587, ARM 99848→100592. This removes small wrapper call layers but increases code size and frame space, so no runtime or startup gain is claimed. The recursive Fibonacci function is unchanged by this first straight-line inliner. Details are in small-call-inline-static.json; final raw dumps /tmp/sf-inline-final-{x64,arm}-dump.

Diagnostic 96f0cfa764d49b79e5d5be9723b0dab16f1f7db1 submits CoreMark run 34079252500 and startup run 34079252501, four native jobs total, pinned to Rust 1.98.1, main f732, parent 42396 and candidate 7fbc510b. Both runs are confirmed in progress. New call-inlining tests also run without guard pages in these diagnostics. Five helper tests plus YAML/shell/source/isolated-trigger checks pass. No PR/main/dev trigger or competitor-engine build was added. Candidate remains unmerged and unretained; full20 and ARM execution gates remain necessary if its diagnostic performance warrants keeping it. All local build/test/dump/probe processes have completed.


## Small-call inlining: final native result, rejected

Runs 34079252500 and 34079252501 are terminal, superseding the live status above. All four actual jobs/steps and seven compiler/correctness audits per job pass, with 260 specs passed per job and no actual compiler warning, soft-fail, action-required or skipped native BMI2-execution marker. Full step/log findings are in small-call-inline-ci-audit.json. Performance is rejected separately: both AMD 9V74 CoreMark draws improve only +0.172543% and +0.113458%, P(improvement) 81.493113% and 78.876426%, while exact-CoreMark startup throughput drops -7.119528% on 9V74 and -8.210069% on N2. All ten points remain recorded below. Source 7fbc510b stays unmerged and excluded from standings; no full20 gate is justified for this version.

| Artifact | CPU | Workload | candidate/parent throughput | P(regression) | parent/main |
|---|---|---|---:|---:|---:|
| core1 | AMD EPYC 9V74 80-Core Processor | coremark-out | +0.172543% | 18.506887% | +5.468110% |
| core2 | AMD EPYC 9V74 80-Core Processor | coremark-out | +0.113458% | 21.123574% | +5.561483% |
| startup_arm | Neoverse-N2 | argon2-results | -0.981197% | 99.990616% | -0.881882% |
| startup_arm | Neoverse-N2 | bz2-results | -0.368357% | 99.739433% | +1.675199% |
| startup_arm | Neoverse-N2 | coremark-results | -8.210069% | 100.000000% | -0.385276% |
| startup_arm | Neoverse-N2 | ffmpeg-results | -1.365151% | 99.998794% | +1.369286% |
| startup_x64 | AMD EPYC 9V74 80-Core Processor | argon2-results | -0.689454% | 99.078893% | -1.860416% |
| startup_x64 | AMD EPYC 9V74 80-Core Processor | bz2-results | -0.030149% | 56.498533% | +0.936991% |
| startup_x64 | AMD EPYC 9V74 80-Core Processor | coremark-results | -7.119528% | 99.999672% | -1.911929% |
| startup_x64 | AMD EPYC 9V74 80-Core Processor | ffmpeg-results | -1.360813% | 99.999838% | -0.085250% |

Exact results are archived under small-inline-core1, small-inline-core2, small-inline-startup-arm and small-inline-startup-x64. Raw ZIPs/native dumps/profiles are /tmp/sf-small-inline-{core1,core2,startup_arm,startup_x64}.zip and matching directories.

An uncommitted follow-up at /tmp/sf-jit-structured-inlining, branch codex/jit-structured-inlining based on 7fbc510b, now skips straight-line callees that still contain calls: those expand live locals/frame space without exposing control flow. It adds bounded structured-callee expansion with every early return redirected to a copied function End inside a result-typed block. Return drops count only callee operands, internal targets and dynamic type rows are relocated, and recursive output is not expanded again. EH/tail-transfer callees remain excluded; original limits are unchanged. Six targeted execution tests cover early-return stack floors, callee branch tables including function-exit targets, loop parameters/backedges, recursive results and stack exhaustion followed by recovery. Those pass locally on ARM; the new full x64 suite/spec validation are running and this follow-up has not been committed, pushed or measured. No previous CI job remains live.


## Structured-call inlining: validated source and native submission

Source 3b88ca70769f6544d58b6e796839ed6f68c6c7bf on codex/jit-structured-inlining, worktree /tmp/sf-jit-structured-inlining, extends unretained 7fbc510b but disables its unprofitable straight-line non-leaf expansion. A straight-line callee must now be a leaf. Small structured callees can retain ordinary nested calls; they are expanded only once over original call sites. The existing 32-op/8-local/256-added-op/64-added-local limits and checked frame bounds remain. Each structured callee is wrapped in a result-typed Block; its function-closing End is copied. Every earlier Return becomes a Br to that End with the drop count derived from its own operand-stack floor. All internal branch/loop/table targets and dynamic result types move with the body. EH/tail-transfer callees retain their ordinary boundary. Caller EH/reference branch relocation remains supported. No benchmark name, input or fusion opcode is recognized.

The new bounded stack-height analysis handles block/loop parameters, Else/End stack restoration and unreachable regions. Tests now cover parameter evaluation order, per-site local initialization, mixed result values, caller and callee branch tables (including branches to the callee function exit), early returns with extra callee operands above an older caller operand, loop-carried parameters, two recursive call results, stack exhaustion and reuse after a trap, memory faults, division traps and host errors before continuation effects. Unit proof verifies one-level structured expansion with nested call type rows, frame overflow/unsupported control rejection, and non-leaf wrapper exclusion. The existing decoded-call/static-frame/prepared-frame test also passes.

Final local suites pass ARM 573 core units / 632 total tests, x64 560 core units / 619 total tests, 20 suites per architecture and 4 existing ignored tests each. Six unguarded suites pass 18 tests; pinned specs pass 260/260. Seven current compiler/correctness log audits have zero errors/warnings. Fmt, lint and diff checks pass. All local compile/test/dump handles are complete. The actual fixed Fibonacci input15 returns I64(610) under Rosetta; its timing is not performance evidence.

All21 dumps contain1285 functions per architecture;166 functions change on each. The fixed recursive function expands from219 to402 x64 bytes, with two original call sites each expanded one level, leaving four nested call sites. CoreMark is exactly unchanged in generated MachineIR and native buffer size on both architectures after wrapper exclusion. Therefore the next diagnostic targets recursive execution, not a claim of new CoreMark runtime gain. Complete per-module function IDs and sizes are in structured-call-inline-static.json. Raw final dumps are /tmp/sf-structured-final-{x64,arm}-dump.

The dedicated diagnostic workflows have been prepared with pinned Rust1.98.1, retained parent42396, mainf732 and candidate3b88ca70. Two x64 draws run the fixed fibonacci-rec.wat with input35, required result9227465, six alternating rounds and two-second samples; x64/ARM startup retain CoreMark, argon2, bz2 and ffmpeg. All four jobs rerun native correctness, specs and warning audits before timing. Five helper tests, YAML/shell/source and isolated-trigger checks pass. Competitor builds and daily PR/main/dev triggers remain unchanged. Submission IDs will be recorded after GitHub confirms them. Source remains unretained; full20/ARM execution gates are still needed before any retention.


## Structured-call inlining: execution result and resource-design review

GitHub confirmed diagnostic 0a82b1dee72963ed0d3038ca7f11e50b4e748468: recursive run34080463053 jobs101614638800(draw1) and101614638621(draw2), startup run34080463069 jobs101614638500(ARM) and101614638656(x64). Both recursive jobs have completed, with all actual steps successful, seven zero-error/zero-warning audits and260 specifications passed each. There are no actual soft-fail/action-required/BMI2-skip markers. Startup jobs were still running when this note was written.

Fixed Fibonacci input35 produces the required9227465. Candidate/parent throughput rises +55.248248% on AMD9V74 and +53.256245% on AMD7763. These are two same-host paired diagnostic results, not a full20 gain, retention verdict or new competitor comparison. CoreMark generated code remains unchanged. Exact comparisons are in structured-call-inline-execution-results.json; execution job audit is in structured-call-inline-execution-audit.json. Archives are structured-inline-recursive1 and structured-inline-recursive2; original ZIPs/dumps/profiles remain /tmp/sf-structured-inline-recursive{1,2}.zip and matching directories.

The user raised the historical reason for retiring inlining: per-function expansion can make small-memory compilation impossible and worsen algorithm4 cost and cache allocation. The repository's compiler design record explicitly documents both concerns (mcts_mem/silverfir/compiler.md,2026-04-14 and2026-06-14). Inspection confirms current region_solver.rs materializes benefit,call_tax,selected and per-slot DP state over regions × locals, while capacity is shared within each region. Extra locals and copied loops can therefore multiply planning state and change allocation quality; instruction-count limits alone do not bound these costs.

Candidate3b88 already bypasses all inlining when compiler_ram_budget_bytes is finite. That preserves the original bounded-budget decode/template decision; it is not evidence of safe peak memory, code-arena fit, frame footprint or algorithm4 cost in the default unbounded-budget configuration. The pass limits added ops/locals, but does not cap final caller regions × locals or prove allocation quality. It also keeps decoded candidate bodies and old/new caller data during expansion, and frame-summary/preparation both perform expansion. None of these costs has yet been measured as a per-function peak allocation bound.

The candidate remains an isolated experiment. Its large recursive-row gain does not override footprint/startup requirements. Before considering retention, resource evidence must include final caller size/local/region growth, tracked compile peak and joint-plan cost, bounded-memory behavior and final code/frame footprint, in addition to complete execution/startup gates. Do not widen the inlining thresholds to chase the recursive result. Work on allocation and call-boundary overhead that preserves separate function bodies remains a priority; reopening inlining requires a resource-aware acceptance policy rather than the existing callee-size threshold alone.


## Structured-call inlining: completed native startup audit

Both runs34080463053/34080463069 are now terminal. All four actual jobs/steps pass, each with seven zero-error/zero-warning audits and260 specs; see structured-call-inline-ci-audit.json. This does not establish performance acceptance. All ten execution/startup points are retained in structured-call-inline-native-verdict.json.

| CPU | Workload | candidate/parent throughput | P(regression) |
|---|---|---:|---:|
| AMD EPYC 9V74 80-Core Processor | fibonacci-rec.wat | +55.248248% | 0.000000% |
| AMD EPYC 7763 64-Core Processor | fibonacci-rec.wat | +53.256245% | 0.000000% |
| Neoverse-N2 | argon2-results | -8.044637% | 100.000000% |
| Neoverse-N2 | bz2-results | -0.476296% | 99.825400% |
| Neoverse-N2 | coremark-results | -1.147295% | 99.998592% |
| Neoverse-N2 | ffmpeg-results | -1.417527% | 99.997300% |
| AMD EPYC 7763 64-Core Processor | argon2-results | -6.011531% | 99.994125% |
| AMD EPYC 7763 64-Core Processor | bz2-results | -1.260576% | 99.924198% |
| AMD EPYC 7763 64-Core Processor | coremark-results | -0.945229% | 96.322237% |
| AMD EPYC 7763 64-Core Processor | ffmpeg-results | -1.391955% | 99.997461% |

The recursive gains are substantial and reproducible on the two sampled AMD models. All eight startup rows regress, especially argon2 (-6.01% on AMD7763 and -8.04% on N2). These costs require investigation; candidate3b88 is not accepted as submitted and remains unmerged. The user explicitly noted that the +53–55% recursive gain is large, so continue resource-aware investigation rather than dismissing the whole inlining direction. No inlining threshold has been widened. Instrumented release probes are being prepared for parent and candidate to attribute compiler heap growth and phase costs; profiler timings will not be treated as uninstrumented startup evidence. No CI jobs remain live at this point.


## Structured-call inlining: direct heap-counter measurement

Identical standalone probes were built natively for Apple ARM64 against parent42396 and structured3b88, release opt3/codegen16/no LTO, jit+guard-pages, no jit-debug or memprof. A GlobalAlloc wrapper tracks requested live heap bytes and the maximum during serial Instance::new, with bytes/WAT parsing/inert import discovery/Engine setup completed before the interval. Both repeats of every one of the21 modules have byte-identical peak increments. No start function or guest code runs. Both build warning audits are0/0. Probe source /tmp/sf-inline-heap-main.rs, manifests /tmp/sf-inline-heap-{parent,structured}, raw /tmp/sf-inline-heap-results; all42 pairs/repeats are in structured-call-inline-heap-results.json.

Fibonacci-rec's compile peak increment rises6733→12648 bytes (+5915), CoreMark stays198155, compression rises885138→981842 (+10.93%), json_parse598618→639460 (+6.82%), word_count387022→402230 (+3.93%). Argon2's overall peak is essentially flat621347→620387 despite much larger individual expanded functions and the measured6–8% startup throughput loss: a module's maximum heap occupancy is not its total compiler work or each function's footprint. This is native ARM memory-layout evidence only, not x64 memory numbers or MCU fit proof. Code/guard mmap reservations, allocator bookkeeping and realloc-internal temporary overlap are excluded.

Earlier memprof-wrapper probes (/tmp/sf-inline-resource-{parent,structured}, results /tmp/sf-inline-resource-results) produced substantially different apparent peaks even for unchanged CoreMark generated code. They change container representations and use the existing profiler timeline/attribution, so their absolute peaks and instrumented timings are not acceptance evidence. The direct counter is the normal-container control and is the source for the numbers above. No production allocator/profiler has been modified.

A resource-limited follow-up worktree /tmp/sf-jit-inline-resource-limits on codex/jit-inline-resource-limits was created from3b88. The planned eligibility model bounds the entire expanded caller (ops,locals,frame and regions×locals), caps cached callee candidates, and rejects large original callers before decoding callees. The current64-bit cost model must not accidentally enable expansion on32-bit MCU backends: the ESP32-C6 configuration explicitly uses u32::MAX for its compiler budget, so the previous finite-budget check alone was not sufficient to protect that device. This follow-up is not yet implemented or measured.


## Resource-limited inlining: validated source and full-corpus submission

Source049874d05c0dd25d1cadf80a290085df6a0526f7 on codex/jit-inline-resource-limits, worktree /tmp/sf-jit-inline-resource-limits, extends unretained3b88. The final expanded caller is capped at128 conservatively counted semantic ops,8 canonical locals,32 frame slots including call scratch, and32 region×local entries (root plus copied loops; zero locals counted as1 for the limit). Original caller bytecode above256 bytes is excluded before callee resolution. Callees retain the128-byte/32-op/8-local filters, with parameter+local counts now checked before raw decode. A caller caches at most8 distinct callee candidates, including negative results. All original expansion/return/control semantics and single-level recursion rules remain. Limits are generic and do not depend on benchmark or input identity.

The footprint model now requires64-bit GP words;32-bit backends keep original function boundaries even when their compiler budget is u32::MAX. Any finite compiler budget still bypasses inlining. Static summaries and all preparation paths use the same BackendConfig-based decision. Three new proofs cover early rejection without resolving callees, combined local/region bounds despite individually small functions, and bounded negative-callee decoding. The decoded-call/frame-plan proof now checks finite-budget behavior and a32-bit GP configuration as well as matching static/prepared frame layouts. No cfg/lint exception, public API or machine ABI was added.

Local full suites pass ARM576 core units/635 total tests and x64563 core units/622 total,20 suites and4 existing ignores each. Six unguarded suites pass18 tests; pinned specs pass260/260. Eight compiler/correctness/probe audit logs contain0 errors/0 warnings. Fmt/lint/diff checks pass. Both source and diagnostic branches are committed and pushed; all local build/test/dump/probe sessions are complete.

Final dumps contain1285 functions/21 modules per architecture;35 functions change on each, down from166 in3b88. Changes span7 modules: fibonacci-rec,json_parse,mandelbrot,prime_sieve,regex_redux,reverse_complement,tiny_keccak. The full Fibonacci function dump and generated machine code are identical to measured3b88 on both architectures (x64219→402 bytes; ARM348→512). CoreMark remains identical to retained42396. Argon2 and compression no longer change generated code. Complete rows are inline-resource-limits-static.json.

Normal-container heap measurements repeat all21 modules twice: the new candidate matches or reduces parent peak increments on20 modules, while Fibonacci-rec remains6733→12648 bytes. The previous compression/json_parse/word_count peak increases disappear. See inline-resource-limits-heap.json. Finite-budget checks on Fibonacci and argon2 at64KiB/1MiB have identical parent/candidate outcomes and peaks; argon2 at64KiB fails identically with the existing unsupported-template exhaustion, rather than being reported as a success. This is host ARM allocation evidence and a policy-bypass check, not a physical MCU run or a hard whole-module memory guarantee.

Measurement clarification: the prior+53–55% recursive diagnostic used input35; the fixed upstream Criterion suite uses input30. The diagnostic gain must not be directly entered into the20-row standings. The new native execution submission uses the existing ci.wasmi_performance tool over all20 standard execute groups and their original inputs. It runs two independent x64 draws and one ARM draw, with no per-row placement exclusions and no competitor-engine feature. The separate startup workflow measures the same four workloads on x64/ARM. Five jobs total; daily PR/main/dev workflows and triggers remain unchanged. Diagnostic commit572cf5ab changes only these two isolated workflow files. Twenty-two measurement-helper tests plus YAML/shell/pin/trigger checks pass; helper CANNOT-RUN/UNSTABLE text is intentional test-fixture output, not a real candidate failure. New run IDs are pending GitHub confirmation. Source remains unretained, and full dev regression validation is still required if these results support retention.

GitHub confirms diagnostic572cf5ab343059cd11abf8e35e45a6a320bf49b5 is live: full20 run34082432990 (two x64 draws plus ARM) and startup run34082433022 (ARM/x64). These are pending measurements, not successful performance gates.


## Resource-limited inlining: startup completed, full20 still live

Startup run34082433022 is terminal; jobs101620137922(ARM) and101620137994(x64) have all actual steps successful, seven zero-error/zero-warning audits and260 specs passed each. No actual soft-fail/action-required/skipped-native-BMI2 marker occurs. See inline-resource-limits-startup-audit.json. All eight points follow; this is not retention acceptance.

| CPU | Workload | candidate/parent throughput | P(regression) |
|---|---|---:|---:|
| Neoverse-N2 | argon2-results | -0.869931% | 99.993983% |
| Neoverse-N2 | bz2-results | -0.387728% | 99.638358% |
| Neoverse-N2 | coremark-results | -0.449728% | 99.997381% |
| Neoverse-N2 | ffmpeg-results | -0.507196% | 99.622692% |
| AMD EPYC 9V74 80-Core Processor | argon2-results | -0.791445% | 87.008109% |
| AMD EPYC 9V74 80-Core Processor | bz2-results | +0.298915% | 13.758340% |
| AMD EPYC 9V74 80-Core Processor | coremark-results | -0.221052% | 75.329691% |
| AMD EPYC 9V74 80-Core Processor | ffmpeg-results | +0.134657% | 21.831377% |

The earlier argon2 6–8% startup drop is much smaller under the whole-caller limits. ARM still shows four negative points of0.39–0.87%; x64 sampled9V74 has mixed -0.79..+0.30% points with weaker probabilities. Do not call the remaining ARM cost absent. Full20 run34082432990 jobs101620137814/101620137830(x64) and101620137937(ARM) are still executing the standard suite comparison.

An independent static combination of rejected loop-priority3388 and GP-reuse8f exists uncommitted at /tmp/sf-jit-loop-cache-regreuse, branch codex/jit-loop-cache-regreuse based on42396. x64 probe builds warning-free and CoreMark instantiation/dump succeeds; no correctness suite or timing has run for the combination. The func5/b18 product chain loses its three old MOVs, and the counter frame88 load is replaced by two carried-register copies; the frame104 stride load remains. CoreMark total bytes17488. This confirms the passes can compose structurally but does not prove runtime gain or justify their combined compile cost. Keep this as an isolated investigation, not a retained optimization.

Next, inspect the inliner's bounded candidate cache: currently even negative eligibility results allocate a BTreeMap node containing inline-body-sized entries. A compact fixed key/index table plus a vector of positive bodies can preserve all eight-candidate decisions while avoiding heap allocation for rejected candidates. This compiler-only follow-up requires exact all21 generated-code comparison and renewed startup validation; do not infer the cost is gone before measurement.


## Resource-limited inlining: standard full20 completed and execution-first acceptance

Full20 run34082432990 is complete. Jobs101620137814 (AMD7763),101620137830 (Intel8573C),101620137937 (N2) have all actual steps successful, five0-error/0-warning audits and260 specs passed each. Synthetic100ns REGRESSION rows earlier in logs belong to helper tests; actual final60 measured rows are archived in inline-resource-limits-full20-results.json and nano-inline-limited-full20-* directories. The final results include negative rows and no final confirmed regression. See inline-resource-limits-full20-audit.json for steps/log audit.

The standard Fibonacci input30 throughput gain is+50.499204% on AMD7763,+23.036628% on Intel8573C,+1.514453% on N2. Complete20-row throughput geomeans rise+2.124526%,+1.010404%,+0.069735%, respectively. These are same-host Nano candidate049874/retained42396 differentials, not a fresh competitor comparison. Intel iterative Fibonacci is-2.173057% (helper status NEGLIGIBLE with its practical-effect test), tail Fibonacci-2.002379% (PASS under family correction); retain these point estimates rather than saying all rows improved. CoreMark generated code remains unchanged and is outside this20-row aggregate.

The user clarified execution performance is the priority and small startup losses are tolerable; current startup is already around Cranelift. Treat the measured ARM0.39–0.87% startup loss as an explicit acceptable tradeoff for this candidate's large execution gain, not a reason to indefinitely block retention or silently report startup as unchanged. Memory/32-bit protections remain. Larger new regressions still require assessment; no arbitrary universal numerical tolerance was specified. The cache-only follow-up worktree /tmp/sf-jit-inline-cache is clean at049874 with no edits and is deferred.

049874 was pushed to independent dev/x64-bounded-inline for the existing complete dev gate. Daily workflows remain unchanged. Main and retained dev/x64-hotpaths remain42396 pending gate inspection; CI run ID awaits confirmation. The isolated CoreMark loop-cache/register-reuse combination remains uncommitted and unmeasured.


GitHub did not create a run for the new dev/x64-bounded-inline ref (SSH confirms its SHA). The CLI has no authentication and no browser is available for manual dispatch. Source049874 was therefore fast-forward pushed to existing remote dev/x64-hotpaths, and GitHub confirmed full dev run34084103483 is live. This is candidate submission, not completed acceptance; local main-worktree source remains42396 pending the gate. No workflow was changed to trigger it and no secret/token was read.

The CoreMark combination now locally scopes GP alias reassignment to blocks rewritten by successful loop-frame caching and admits only aligned full-word canonical-frame Store instructions as ordered publication nodes. Stores retain order and values, while loads, guest memory, calls, traps and fixed/lowering registers remain barriers. Two new proofs cover repeated same-slot publication with32/64-bit wraparound and rejection of non-frame/partial/unaligned stores; the independent512-program alias evaluator now includes ordered frame stores. All7 focused tests pass natively on ARM. Source remains uncommitted; full native dumps, suites and timings are still pending.


## Loop-cache publication/register-reuse candidate: final local validation

Source937ed3e09fcd959c47acf16e77cf5672c854580b is committed and pushed on codex/jit-loop-cache-regreuse, based on retained42396 (no inliner in this experiment). It composes loop-cache priority with alias reassignment, but reassignment runs only in blocks actually rewritten by loop caching. Canonical aligned whole-word FP stores are ordered publication nodes rather than fragment boundaries; no store is moved or removed, and each published immutable value is preserved. Guest memory, loads, trapping operations and calls remain boundaries. The pass borrows only already-written allocatable GP lanes and restores required output ownership.

A final full-word cached move can also supply an immediately following value-branch condition. Reading that identical cached value prevents a second live alias from forcing restoration of the temporary. The new execution proof compares both branch target/edge arguments and ordered frame stores across wraparound values. This is a generic cache-publication/branch rule; no benchmark/function/input identity is recognized. In CoreMark func5/b18, the product chain loses its three old moves, the frame88 counter load disappears, and decrement/branch now use r10 directly (the remaining self-Move is metadata-only). Frame104 stride remains a loop load. CoreMark x64 total native bytes17484→17456; ARM15364→15380, with no ARM f5 change. Neither number proves speed.

All21-module dumps contain1285 functions per architecture;103 x64 and21 ARM MachineIR functions change, including7/14 module sets listed in loop-cache-register-reuse-static.json. Some code buffers grow, includingjson_parse andregex_redux; complete rows are retained. Raw current dumps are /tmp/sf-loop-joint-terminal-{x64,arm}-dump. Earlier final/published dump directories predate the terminal-condition improvement and are not the final candidate.

Final local default suites pass580 ARM core units/633 total tests and567 x64 units/620 total,19 suites and4 pre-existing ignores each. Five unguarded suites pass12 tests, pinned specs260/260. Seven build/correctness/probe logs audit0 errors/0 warnings. Fmt, lint, diff and five measurement-helper tests pass. All local build/test/dump sessions are complete. Source is not retained pending native timing.

Diagnostic0d07ecd3 changes only .github/workflows/x64-profile.yml to two isolated native x64 CoreMark draws comparing mainf732, parent42396 and candidate937ed3, with native correctness/spec/warning gates and raw code/profile artifacts. Startup workflow and dailyPR/main/dev workflows are unchanged. Given the user's execution-first preference, first measure execution; expanded performance/startup gates follow only if the candidate has useful runtime benefit.

The user explicitly authorized a parallel Agent for interpreter wasmi execution, preserving root JIT work. Agent interpreter_wasmi_execution works in /tmp/sf-interp-wasmi-execution, codex/interp-wasmi-execution based049874. It is independently establishing a same-host Nano/wasmi anchor and investigating common slow-exit operations over all20 workloads. Root coordinates CI and reviews semantic/lifecycle risks; no benchmark-specific fusion, WASI, CoreMark or startup work was added to that agent's scope. Initial candidate is a native memory0.size handler using the exact cached logical length, subject to grow/host/module-switch refresh proofs. No interpreter performance gain has yet been measured.


GitHub confirms CoreMark run34084806606 for diagnostic0d07ecd338f9be5372a4a51aeb4883c1f7292fda, jobs101626680535/101626680810. Both are in native correctness validation; timings pending. Inliner049874 full dev run34084103483 is still live; completed actual jobs/logs inspected so far show no errors or soft-fail. Both full20 interpreter execution jobs and both interpreter startup jobs prove identical compiled runtime to main; all their apparent deltas are drift, not JIT-induced interpreter improvements.

A provisional AMD7763 projection in standings-049874-epyc7763-provisional.json composes frozen competitor anchors with the retained423 differential and049/423 full20 differential. It estimates full20Nano+0.990279% vsV8/+12.095960% vsCranelift, but is explicitly provisional while the full dev gate remains open and is not a fresh same-host three-engine comparison. No Intel extrapolation or new CoreMark gain is claimed.


## Inliner049874 full dev gate: confirmed startup failure

Run34084103483 is terminal. All34 actual jobs/steps/logs have been inspected in dev049-gate-audit.json. The ARM JIT startup cross-run verdict (job101627982175) failed: erc20 primary3.508044→3.770390ms (-6.958050% throughput) and independentN2 confirmation3.476→3.779ms (-8.01%). This is a real failed gate even though dev continue-on-error leaves the workflow conclusion successful. Fifteen other confirmation jobs skipped their measurement because the primary flagged no rows; they are not independent performance confirmations. No compiler-warning or correctness failure was found.

All54 JIT wasmi rows are archived in dev049-wasmi-results.json and sf-dev049-wasmi-performance-* directories. Compared to mainf732, full20 throughput geomean is+16.028388% on AMD7763 and+4.770256% on N2. The x64 geomean includes the existing PLACEMENT-marked tiny_keccak-4.531420% row; no metric is omitted. ARM startup seven-row geo-0.861259%, x64 AMD9V74 startup-2.481873%; all negative/floor/noisy rows are retained. The separate049/423 diagnostic's smaller startup costs covered onlyCoreMark/argon2/bz2/ffmpeg and missederc20. Do not generalize those four to every module. User tolerates small startup losses but did not explicitly accept this newly discovered7–8% regression. Source049 remains an unaccepted candidate on remote dev/x64-hotpaths and its isolated branches; local retained worktree remains42396. No gate exception/suppression or rollback was made.

The previously deferred compact-cache follow-up is now implemented but uncommitted in /tmp/sf-jit-inline-cache. It replaces BTreeMap<u32,Option<InlineBody>> with bounded8-key/optional-u8-body-index arrays and a Vec of positive bodies only. Negative results still consume the same8-candidate budget and are resolved once, including repeats after the cache fills. Whole-caller/callee limits, expansion order and all policy decisions are intended to remain identical. Local validation is in progress; generated-code/heap/startup evidence is still required. The shared build window was coordinated with the interpreter agent after its local timings ended.

## Loop-cache candidate: initial native evidence and longer confirmation

Run34084806606 completed both jobs101626680535/101626680810. Each has seven0-error/0-warning audits and260specs, all actual steps successful. Both CPUs are AMD7763. Source937ed3/parent423 throughput is+0.248602% (Pimp82.482927%) and+0.704797% (Pimp99.853322%). Draw1 retains one negative paired point-0.424277%, alongside three+0.44..0.51%; do not drop it. Parent/main control changes+8.697056/+7.395175 between runners, so cross-run controls are not new source gains. Results and raw scores are loop-cache-register-reuse-execution-results.json, directoriesloop-cache-register-core1/2. Func5/b18 sampled share changes8.30→7.73% in draw1, consistent with reduced loop work but not an independent whole-program speed proof.

Diagnostic0b8ce002 changes only the isolated CoreMark workflow to two12-pair parent/carry confirmations on the same fixed sources. It removes the already-known main control and repeated profiling to spend the work on paired precision. Native correctness/warning/spec and code dumps remain, daily workflows unchanged. Source is not retained; run ID pending confirmation.

Interpreter hardware correction: the current host is Apple M1 Pro (verified by the agent using sysctl), not the historicalM4 in older repository records. Every new local interpreter result is M1 Pro only. The initial memory.size candidate's complete paired rounds are-5.762%/+0.261% geo, with retained anomalouscounter-local and persistent negative rows. Root did not accept it. The agent found that appending the handler before Apple superhandlers shifts the old340-byte superbank by88bytes; it is moving the new handler after the existing banks, checking binary layout and repeating full20 measurement. No PLACEMENT exemption is allowed.
