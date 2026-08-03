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

## Interpreter

Second phase per the goal ordering; snapshot with the same lanes after
the JIT work.
