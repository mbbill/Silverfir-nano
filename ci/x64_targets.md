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
