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

Baseline run: 30797021672 / commit `a1aa2a38` / AMD EPYC 7763 /
rustc 1.97.1 / suite 16a3d7c8. Times are single Criterion mean estimates;
treat gaps under ~5% as parity until re-measured.

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

Pending: first CI run of the `wasi` standings job (in flight on this
branch). Append its reference table here when it lands.

## Interpreter

Second phase per the goal ordering; snapshot with the same lanes after
the JIT work.
