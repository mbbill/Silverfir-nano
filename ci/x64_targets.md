# x64 tuning targets (branch-only working doc)

Reference cross-engine ratios for the dev/x64-tuning campaign. Measured
ONCE per baseline by `.github/workflows/x64-standings.yml` (dispatch-only);
daily iteration measures nano-vs-nano deltas with the performance-regression
CI and aims for the cuts below. When every row reads met, re-dispatch the
standings lane to confirm against live V8/Cranelift. Delete this file with
the rest of the lane before merge.

Goal (user, 2026-08-03): x64 at the same standing as the arm64 Mac results —
per-case target is the better of wasmtime-cranelift and V8.

## wasmi-benchmarks execute corpus — JIT

Baseline run: 30798311764 / commit `6a959fde` / AMD EPYC 7763 /
rustc 1.97.1 / suite 16a3d7c8. Times are single Criterion mean estimates;
treat gaps under ~5% as parity until re-measured. The full corpus was
re-measured on a second independent runner draw (run 30797021672) and
every ratio reproduced within a few percent — the reference is stable.

`gap` = nano time ÷ best competitor time. `cut needed` = time reduction
on nano to reach that competitor (1 − 1/gap).

| case | nano | best competitor | gap | cut needed |
|---|---|---|---|---|
| fibonacci-rec | 10.6 ms | v8 2.88 ms | 3.68 | 72.8% |
| counter-local | 628.7 µs | cranelift 313.4 µs | 2.01 | 50.2% |
| counter-param | 624.1 µs | v8 313.8 µs | 1.99 | 49.7% |
| fibonacci-tail | 623.9 µs | v8 314.9 µs | 1.98 | 49.5% |
| sort | 60.6 ms | v8 32.9 ms | 1.84 | 45.7% |
| argon2 | 159 ms | cranelift 97.7 ms | 1.63 | 38.6% |
| word_count | 1.46 ms | cranelift 946.4 µs | 1.54 | 35.2% |
| nbody | 16.7 ms | cranelift 11.1 ms | 1.50 | 33.5% |
| json_parse | 9.12 ms | cranelift 6.25 ms | 1.46 | 31.5% |
| reverse_complement | 37.6 µs | cranelift 26.6 µs | 1.41 | 29.3% |
| tiny_keccak | 23.9 µs | cranelift 16.9 µs | 1.41 | 29.3% |
| prime_sieve | 27.0 ms | cranelift 20.0 ms | 1.35 | 25.9% |
| regex_redux | 30.1 µs | cranelift 23.6 µs | 1.28 | 21.6% |
| compression | 12.2 ms | cranelift 10.2 ms | 1.20 | 16.4% |
| bulk-ops | 775.4 µs | v8 684.9 µs | 1.13 | 11.7% |
| spectralnorm | 18.2 ms | v8 17.2 ms | 1.06 | 5.5% |
| matrix_mul | 60.0 ms | cranelift 57.5 ms | 1.04 | 4.2% |
| mandelbrot | 17.4 ms | cranelift 17.4 ms | 1.00 | parity |
| counter-global | 156.3 µs | — | 1.00 | already best |
| fibonacci-iter | 637.0 µs | — | 1.00 | already best |

Rows where nano already leads clearly: fibonacci-tail vs cranelift (0.30)
and spectralnorm vs cranelift (0.60) — do not regress these while fixing
the rest.

Geomeans at baseline: nano/v8 1.34, nano/cranelift 1.21.

Caveats: AMD EPYC draw (the user's own lagging observation was on Intel);
re-confirming on an Intel draw is part of final verification. Ratios are
one snapshot — the performance-regression CI measures the deltas that
count toward these cuts.

## benchmarks/wasi suite — JIT

Baseline run: 30798311764 / commit `6a959fde` / AMD EPYC 7763 /
wasmtime 47.0.2 (prebuilt) / Node 24.18 (V8). Rates, higher is better;
`gap` = best competitor rate ÷ nano rate. This suite is the primary goal
metric: benchmarks/wasi/RESULTS.md holds the arm64 M4 reference where
nano is at parity with Cranelift (best-of 15 metrics: Cranelift 9,
nano 4, V8 2). The `M4 reference` column is that standing.

| metric | gap | best competitor | M4 reference |
|---|---|---|---|
| funcref/exported-table | 2.62 | cranelift | — |
| stream/Triad | 2.36 | v8 (cranelift 2.29) | nano ≈ cranelift |
| stream/Scale | 2.22 | cranelift (v8 2.20) | nano 1.96× OVER v8 |
| stream/Add | 2.18 | cranelift | nano ≈ cranelift |
| lua/fib | 1.91 | cranelift | nano −6..9% of cranelift |
| lua/sunfish | 1.81 | v8 (cranelift 1.75) | nano −6..9% of cranelift |
| lua/json_bench | 1.74 | cranelift | nano −6..9% of cranelift |
| c-ray | 1.61 | v8 (cranelift 1.42) | competitive |
| sqlite | 1.60 | v8 (cranelift 1.30) | competitive |
| lz4/decompress | 1.49 | cranelift | competitive |
| bzip2 | 1.47 | v8 (cranelift 1.23) | nano LED +14% |
| coremark | 1.47 | v8 (cranelift 1.23) | tie for best |
| lz4/compress | 1.25 | v8 (cranelift 1.01) | competitive |
| sha256 | 1.21 | cranelift | nano LED +16% |
| stream/Copy | 1.01 | parity | parity (host memcpy) |
| mandelbrot | best | — | tie for best |

funcref/direct shows v8 at 8.01× — 3.2e9 calls/s smells like V8 optimizing
the call away; use the cranelift ratio (1.34) as the actionable reference
until verified.

The rows where the M4 reference says "led/over" are the purest x64-backend
signal: same engine, same suite, opposite outcome by ISA — sha256, bzip2,
coremark, and the STREAM arithmetic kernels.

## Checkpoint history

### 2026-08-03 — after fix 1 (x86_64 [base+index+disp] addressing, 03089696)

A/B verdicts (AMD, run 30800951711): native suite 14/17 IMPROVEMENT —
coremark +17.5%, sqlite +18.7%, stream Scale/Add/Triad +21/+29/+19%,
lz4 +17/+9%, lua +7-8% (all three), funcref-exported-table +6.9%,
c-ray +3.6%, sha256 +3.5%, bzip2 +2.3%. wasmi corpus: sort +15.6%,
nbody +8.8%, spectralnorm +8.1%, word_count +8.1%, json_parse +7.7%,
compression +7.3%, reverse_complement +4.8%, tiny_keccak +2.6%.
regex_redux flagged −16.6% on the primary runner but showed +12.5%
IMPROVEMENT on the independent confirm runner — dismissed as a
layout/draw artifact.

Standings checkpoint (run 30800992673): wasi suite landed on
**Intel Xeon 8370C** (first Intel datapoint) — geomean 1.36 cranelift /
1.50 v8. wasmi stayed on AMD: geomean 1.19 cranelift / 1.31 v8
(from 1.21/1.34).

Remaining top rows, Intel wasi: funcref 2.2-2.4x, lz4/decompress 1.80,
stream/Triad ~1.46, lua/sunfish 1.55(v8)/1.33(cl), lua/json ~1.4,
sqlite 1.40(v8), bzip2 1.40(v8). Remaining top rows, AMD wasmi:
counter-local/param and fibonacci-tail ~2.0x (untouched by addressing —
different cause), fibonacci-rec 3.7x vs v8 (likely inlining), argon2
1.5-1.6x (unmoved).

### 2026-08-03 — after fix 2 (flags reuse, 06b8fd52) and fix 3 (loop-header
### alignment, 85e493c0)

Fix 3 A/B (run 30806530811, AMD primary): **counter-local +99.97% and
counter-param +100.05%** — both now 312µs, dead even with cranelift/v8;
their 2.0x rows are CLOSED. prime_sieve +7.1%, fibonacci-iter +0.65%.
Native suite cumulative vs main, all confirmed, zero regressions:
stream Add +33.6% / Triad +22.4% / Scale +21.6%, sqlite +17.2%,
lz4-compress +14.8%, coremark +14.0%, lz4-decompress +9.4%, lua-sunfish
+8.7% / lua-json +8.6% / lua-fib +8.4%, bzip2 +4.1%, sha256 +3.7%,
c-ray +3.7%, funcref-exported-table +2.3%.

Known tradeoff, documented and accepted: **execute/regex_redux** runs
~17-20% slower than main on AMD EPYC draws (7763/Zen3 and 9V74/Zen4,
four consistent measurements across three code layouts) while improving
+12.5% on an Intel Xeon 6973P-C draw. The vendor split matches the
store-to-load-forwarding constraint the pre-fix lowering's
"stable-base form" comment guarded: Zen restricts forwarding into
indexed-address loads; Intel does not (leading hypothesis — PMU
counters are unavailable on the runners to confirm directly). The
indexed form wins +2.5-33% on ~27 rows on BOTH vendors and the goal
prioritizes Intel; no vendor-forked codegen for one row.

Remaining open rows: fibonacci-tail 2.0x (return_call chain, untouched
by loop alignment — next), fibonacci-rec 3.7x vs v8 (inlining-class),
argon2 ~1.5x, lua/lz4-decompress/sqlite/c-ray/bzip2 residuals pending a
fresh standings checkpoint.

### 2026-08-03 — after fix 4 (inline jump-edge moves, f6fdb64e) and fix 5
### (scaled table dispatch + out-of-line tables, 5ce64999)

Fix 5 A/B (run 30811613146, AMD-class): every native row IMPROVEMENT —
the Lua trio jumped to +10.5/+11.7/+10.8% (fix 5 targeted its 80-way
dispatch: one-instruction scaled jump, 640 bytes of table data out of
the hot instruction stream), sqlite +21.6%, coremark +19.0%, stream
Add +33.5%. Only red row remains regex_redux on AMD (documented above).

Fix 4 forensics: its apparent lua-fib −9.95% did NOT reproduce in a
controlled same-SKU comparison (EPYC 7763 profiles pre/post fix 4:
524.4 vs 530.3 fib20/s, statistically identical block heat) — the A/B
that flagged it drew a Xeon 6973P-C for both primary and confirm, an
SKU ~40% slower on Lua at baseline. Tracked as SKU sensitivity, not a
code defect.

fibonacci-tail stays ~1.8-2x vs v8: latch is now single-jump and
aligned; the residual is uop count in the loop-carried parameter
rotation (mov shuffle) — regalloc coalescing territory, parked.

### 2026-08-03 — after fix 7 (load+ALU memory-operand fusion, 37987fac)

A/B run 30826610658, zero failed jobs (regex_redux did not reproduce on
this draw). Cumulative vs main: **stream-Scale +93.3%** (fold + fusion
compounding), sqlite +22.2%, stream-Add +33.8%, Triad +21.8%, coremark
+19.6%, lz4-compress +14.6%, lua-json +11.4% / sunfish +10.7% / fib
+10.2%, word_count +9.2%, nbody +8.8%, json_parse +7.9%, prime_sieve
+7.8%, **fibonacci-tail +33.2%** (formerly parked; the fusion chain
reached its latch), counters holding +100%. argon2 +1.7% only: its 600
fused instructions confirmed front-end pressure is not its binding
constraint — the state's store→load round-trips are (register-pressure
design item).

### 2026-08-03 — consolidated standings after all seven JIT fixes
### (run 30830020441)

**Official wasmi corpus (AMD EPYC 9V74): nano/cranelift geomean = 1.01
— statistical parity with wasmtime-cranelift.** nano/v8 = 1.12 (the
residual is fibonacci-rec 2.30, V8's inlining). nano now BEATS
cranelift on fibonacci-rec (0.94), fibonacci-tail (0.22), spectralnorm
(0.54), prime_sieve/json_parse/matrix_mul/compression parity-or-better
vs v8. Remaining >1.2x rows vs cranelift: regex_redux 1.58 (AMD
tradeoff), reverse_complement 1.42, argon2 1.41 (latency-bound,
register-pressure item), prime_sieve 1.29, tiny_keccak 1.26. Combined
with the interpreter lead on every row, the official-corpus standing
now matches the arm64 pattern: interpreter first, JIT at
optimizing-tier parity.

wasi suite (Intel 8370C): geomean 1.27 cranelift / 1.40 v8 (baseline
1.51/1.65). Remaining drivers, in order: funcref 2.2 (shared call ABI
— design decision), lz4/decompress 1.81 (Intel-specific residual,
uninvestigated), lua 1.27-1.37 (register pressure — design decision),
c-ray 1.29, bzip2 1.25, stream/Triad 1.26 (bandwidth-class now;
Scale/Add at 1.04-1.12).

## Next implementation unit: preserved dynamic lanes on x86_64

The register-pressure class (lz4 wildcopy [r1+24]/[r1+120] reloads,
argon2 state round-trips, lua residuals) is NOT gated on a design
decision after all — the design already exists and arm64 uses it:
`BackendConfig::with_volatility` with GP_PRESERVED_DYNAMIC lanes plus
the backend's lazy body save/restore
(`lower_preserved_dynamic_body_save/restore` in arm64/backend.rs).
x86_64 uses plain `BackendConfig::new` = zero preserved lanes, so every
cached local dies at every call boundary — while R14/R15 (callee-saved,
already unconditionally pushed by the prologue) sit in the dynamic pool
classified volatile.

Adoption plan (mechanical but frame-protocol-wide; fresh session):
1. REG_PLAN.gp_dynamic reorder: [RSI, RDI, R8, R9, R10 | R14, R15 |
   R11] = volatile 5 | preserved 2 | internal scratch 1, with arm64's
   compile-time count asserts; add the caller-saved-subset list arm64
   keeps (`gp_dynamic_caller_saved`).
2. compile_backend_config → with_volatility(8, 5, 2, 1, 14, 0,
   /*gp_arg_lanes*/ 4, /*fp*/ 4, /*scalar_return_lanes*/ false,
   SCALAR_CALL_SCRATCH_SLOTS); keep scalar_return_lanes false (out of
   scope).
3. Implement x64 lower_preserved_dynamic_body_save/restore mirroring
   arm64 (body-prelude save of used preserved lanes, restore on every
   return path INCLUDING the trap-exit tail); wire where arm64 wires
   them (body prelude + return sequence).
4. save_caller_clobbered_gp_dynamic must save only the caller-saved
   subset (R14/R15 survive C helper calls).
5. with_preserved_lane_save_overhead: measure; the public prologue
   already saves R14/R15, but SF→SF bodies pay the lazy save, so
   arm64's 3 is the honest starting value.
6. Validate: Rosetta spectest + wasi correctness both callconvs
   (Win64 check builds), A/B watch on lz4-decompress / argon2 / lua /
   sqlite rows and startup lanes.

### 2026-08-03 — fix 8 (preserved dynamic lanes, c1bc0c1d + 5213a917)

x86_64 adopts arm64's preserved-lane design: R14/R15 reclassified from
volatile to preserved, bodies lazy-save exactly what they clobber, all
four frame paths share one save/shim plan. A/B: lua-json +12-14%,
lua-fib +11.3-11.6%, sqlite +12.9%, sha256 +8.9%, c-ray/bzip2 +5.4%,
funcref-exported-table up to +5.4% — call-crossing residency working.

Calibrations from the confirm lanes (5213a917): the induction fold now
pre-scans loops for its pattern (arm64 startup cost cured — no arm64
failures in run 30835024499), and the x64 preserved-save overhead price
is 5 (solver declines nomination in tiny bodies; fibonacci-rec's
prelude verified push-free).

Documented tradeoff #2: **fibonacci-rec −3.4%** on x64 — not the lazy
save (verified absent) but the volatile-lane reduction 7→5 pressuring
a tiny recursive call-tree's arg staging. Bounded and acceptable:
fibonacci-rec remains ahead of cranelift (≈0.97 after this), and the
same reclassification funds the Lua/sqlite/sha256/funcref gains.

### 2026-08-03 — closing standings for this stretch (run 30837737428)

Same-SKU trajectory (EPYC 7763, matched to the campaign baseline):
wasi geomean **1.51 → 1.39 → 1.34** vs cranelift (1.65 → 1.51 → 1.46
vs v8); wasmi corpus **1.21 → 1.11 → 1.10** vs cranelift (1.34 → 1.23
→ 1.22 vs v8). The 1.01-parity and 1.27 readings earlier came from
Zen4 and Intel draws respectively — the ratios are SKU-dependent, so
standings claims are per-SKU; the improvement holds on every SKU
measured.

Remaining open work, all with evidence dossiers: local-cache SELECTION
for lz4's wildcopy — NARROWED: the pointers ARE locals (func17 slots 3
and 15 of 19), the region solver already trip-weights per wasm-loop
regions (SF_CACHE_POLICY=algorithm4:trip=N), and a trip sweep
(8/32/128) leaves func17's 122 spill ops IDENTICAL — the exclusion is
structural, not weight-based. RESOLVED by solver trace (2026-08-03,
local diagnostic, reverted): LZ4_decompress_safe's main-loop region
(45 blocks) has gp_capacity 3 = budget 7 minus headroom 4 (the region
capacity subtracts the WORST owned block's SSA transient peak), while
six locals carry benefit >=120 — the solver correctly selects the top
three (l17=256, l2=200, l4=176) and the profiled wildcopy reload
victims l15=152 and l3=136 rank 4th/5th and lose. Fix class: solver
granularity — split expression-heavy peak blocks out of hot regions,
or per-block spill-around so one peak block does not tax 45 blocks'
residency. This is joint-planner architecture (shared, both ISAs
benefit; arm64 hides it behind 20+ lanes) — sized for a dedicated
session with these numbers as the acceptance test. Then:
the value-stack call ABI (funcref, user decision), fibonacci-rec's v8
gap (inlining class), and the two documented tradeoffs
(regex_redux/AMD, fibonacci-rec lane trade).

### 2026-08-03 — fix 10 (region capacity raising + per-block clamp,
### 545cc24e + 6bdbcecf + b2c82a03)

The register-pressure class fix, inside the solver's own design: a
region may exceed its worst-block-peak capacity when an extra
resident's benefit outside the peak blocks pays its ride-around
crossings; after solving, each block sheds lowest-benefit residents
only where its own transient peak demands (edge repair reconciles).
Calibrations: starvation gate before ranking; post-solve verification
of raised extras with the DP's actual selection (repricing the shared
benefit table was rejected by a planner unit test — it distorts
within-capacity competition); verification/clamp skipped entirely when
nothing was raised.

A/B confirmed on x64, zero execute regressions in the final run:
lua-fib +20.7%, lz4-decompress +17.0% (from +6.1 pre-raise), sqlite
+17.3%, lua-sunfish +17.3%, lua-json +13.6%, coremark +19.1%.
Acceptance: LZ4_decompress_safe spills 122 -> 103.

Open startup question: x64 startup/pulldown-cmark and startup/ffmpeg
-4.3% (harness-only modules; CLI cannot compile them for local
analysis — they need stub imports). Local proxies show flat analysis
cost (lua compile identical pre/post; 73 raises) and arm64-native
coremark startup measures FASTER at HEAD (+1.6%) contradicting the CI
runner's -2.26% — the small-module delta is machine-dependent noise at
the floor. The suite doctrine (execute wins outrank startup costs;
not vice versa) supports the trade; the targeted follow-up is a
CI-side instrumented run or CLI stub-import support to profile the
big-module compile path.

### 2026-08-03 — post-fix-10 standings (run 30849884018, EPYC 9V74)

**wasmi corpus: nano/cranelift geomean = 1.00 — exact parity** (v8
1.10). wasi suite: **1.24 cranelift / 1.38 v8** (campaign baseline
1.51/1.65). Row movement from the register-pressure fix: lua/fib 1.15
cl / **1.02 v8** (baseline 1.91), lua/sunfish 1.15/1.24 (1.81),
lua/json 1.22/1.27 (1.74), lz4/decompress 1.31 (1.49-1.81),
stream/Scale 1.09 (2.22), sha256 1.06/**1.00** , lz4/compress 1.00,
coremark 1.07/1.25 (1.47), mandelbrot ahead 0.94/0.64.

Remaining wasi rows above ~1.3: funcref 2.26 (value-stack call ABI —
the design decision), stream Add/Triad 1.68-1.71 on this draw
(bandwidth-class, SKU-variable — 7763 read them 1.12-1.26 post-fold),
c-ray 1.32, sqlite/v8 1.45. With the interpreter leading every row and
the official corpus at cranelift parity, the arm64-pattern standing
holds on the goal's primary metrics; the wasi residual is concentrated
in the ABI decision and SKU-variable bandwidth rows.

### 2026-08-03 — terminal decomposition of the wasi tail (profiles
### 30850932814/34495/36342)

- **sha256 (1.06 cl / 1.00 v8)**: hot round block = 41 ops, 16 state
  spills — the irreducible floor: 8-word state + 16-word schedule
  exceeds 15 GPRs for every x64 engine; the capacity raise correctly
  declines (every round block equally dense, second-peak == peak) and
  load+ALU fusion correctly declines (loads feed multi-use Ch/Maj).
  Cranelift pays the same physics; we tie V8. The M4 "+16% lead" was
  cranelift-arm64 relative weakness, not headroom we are missing.
- **bzip2 (1.14)**: flat profile, hottest block 4.8% — no dominant
  mechanism; scheduling-quality territory.
- **c-ray (1.32)**: heat spread across FP blocks at 8-13% — no
  concentration; same territory.

These three are diminishing-returns rows for a baseline JIT's toolkit:
no single mechanism remains, and the residual is optimizing-tier
scheduling/CSE quality. The wasi geomean's substantive remaining
levers are exactly funcref (ABI decision) and the SKU-variable stream
bandwidth rows.

### 2026-08-03 — SUPERSEDED: the tail is not terminal. arm64 reference
### run + v8-normalized cross-ISA deficits

The decomposition above compared nano-x64 only against competitors on
x64. Measuring the same wasi suite natively on the arm64 Mac (the
goal's reference point; wasmtime 42.0.1 / node v25.9.0, no pinning)
gives the arm64 standing: **nano/cranelift geomean 0.76 (nano leads),
nano/v8 1.18** — so "arm64 level" on wasi means beating cranelift
outright, and x64's 1.24 is not there yet.

Normalizing each row by v8's own cross-ISA scaling
(`deficit = (v8/nano)_x64 ÷ (v8/nano)_arm64`; the cranelift column is
unusable as a normalizer — wasmtime-42-on-mac is anomalously slow,
e.g. 585 fib20/s vs 1,170 on x64 CI) isolates what is genuinely
x64-specific. Geomean deficit **1.17**. Sanity rows mandelbrot,
lz4/compress, stream/Copy sit at 1.00±0.03; stream/Scale (0.65) and
funcref/exported-table (0.26) are already better than their arm64
standing.

The x64-specific worklist, by cluster:

| cluster | row | deficit |
|---|---|---|
| call path | funcref/direct | 2.68 |
| call path | lua/fib | 1.48 |
| call path | sqlite | 1.40 |
| call path | coremark | 1.30 |
| call path | lua/json_bench | 1.23 |
| call path | lua/sunfish | 1.13 |
| 2-load stream kernels | stream/Add | 1.83 |
| 2-load stream kernels | stream/Triad | 1.42 |
| byte/bit-dense loops | lz4/decompress | 1.40 |
| byte/bit-dense loops | sha256 | 1.36 |
| byte/bit-dense loops | c-ray | 1.34 |
| byte/bit-dense loops | bzip2 | 1.20 |

Corrections to the section above forced by this data:

- **funcref/direct is not design-gated.** nano-arm64 does 1.35B
  calls/s (~3 cycles/call) with the same value-stack ABI; nano-x64
  does 355M (~10.5 cycles/call at 3.7 GHz). ~7 cycles per SF->SF call
  are x64-backend cost, not ABI cost. (exported-table remains the
  ABI-shaped row: 2.10 on arm64 vs 2.26 on x64.)
- **sha256's "shared 15-GPR floor" is not the whole story**: nano-arm64
  beats v8-arm64 1.35x on sha256; on x64 we only tie v8. v8 handles
  the 15-GPR floor better than nano-x64 does relative to arm64.
- **stream Add/Triad are not (only) SKU bandwidth rows**: Copy and
  Scale are clean, arm64 wins Add outright — the 2-load kernels are
  a codegen gap.
- bzip2/c-ray "scheduling territory" still carries a 1.20-1.34
  x64-specific component worth a look before writing it off.

Artifacts: `do_not_scan/x64-standings-test/arm64-ref/standings.{md,json}`
+ `cross_isa.py` beside it. Next: profile the call-path cluster
(funcref/direct first — smallest reproducer, biggest deficit).

## Interpreter

Baseline run: 30819701182 / commit `8d7261de` / AMD EPYC 9V74 /
standings lane `tier=interp` (nano-interp vs stitch, wasm3.eager,
wasmi-v2.eager.checked on the official corpus).

**nano-interp already leads the x64 interpreter field: geomean 0.64 vs
stitch, 0.54 vs wasm3, 0.64 vs wasmi-v2 (below 1.00 = nano faster),
winning 18 of 20 rows** — matching its Apple-Silicon standing except:

| case | vs stitch | vs wasm3 | vs wasmi-v2 |
|---|---|---|---|
| spectralnorm | 2.74 | 2.31 | 3.33 |
| bulk-ops | 1.07 | 1.08 | 0.96 |

spectralnorm is the single real interpreter target (bulk-ops is
parity-noise). mandelbrot — also FP-heavy — wins at 0.70/0.53/0.86, so
this is not a general FP weakness; spectralnorm leans on f64 division
and int→f64 conversion in its inner loop. Mechanism unidentified yet;
first experiment: same-engine arm64-vs-x64 comparison to classify
x64-specific vs engine-general.

### 2026-08-03 — interpreter fix 1 (inline u64→f64 conversion, 55aeeeaa)

Root cause: the x86_64 interp generator declined F64_ConvertI64U (no
SSE2 instruction), so every execution fell back to the interpreter
core; spectralnorm converts once per matrix element. Native arm64
measurement (same engine 2.8x AHEAD of wasmi there) proved the ISA
split. Fixed with the branch-free split-halves sequence (exact halves,
one final rounding).

A/B: **execute/spectralnorm interp +246.2%** (99.99%+). Standings
re-run (run 30823393515, EPYC 7763): spectralnorm now 0.68 vs stitch,
0.59 vs wasm3, **1.00 vs wasmi-v2**; interpreter geomeans 0.60 / 0.52 /
0.68 — **nano-interp leads or ties every peer on every row. The
interpreter phase is at its arm64-level standing.** (bulk-ops wobbled
1.10-1.22 on this draw; single-run noise on a parity row.)
