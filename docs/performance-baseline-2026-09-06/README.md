# x64 baseline — 2026-09-06

Source runtime: `main@f73219f5`; measured checkout: `fe334819` (CI-only changes).
[CI run 34026700835](https://github.com/mbbill/Silverfir-nano/actions/runs/34026700835).

## WASI (17 metrics, 12 benchmark workloads)

Ratios below are competitor throughput / Nano throughput. A value above 1
means Nano needs that factor of throughput improvement to match. These are
same-run point estimates, not statistically confirmed cross-engine wins.

| Metric | 7763 / CL | 7763 / V8 | 9V74 / CL | 9V74 / V8 |
|---|---:|---:|---:|---:|
| bzip2 | 1.099 | 1.310 | 1.058 | 1.212 |
| c-ray | 1.246 | 1.413 | 1.234 | 1.371 |
| coremark | 1.032 | 1.214 | 1.019 | 1.200 |
| funcref/direct | 1.334 | 7.976 | 1.166 | 6.973 |
| funcref/exported-table | 1.777 | 1.357 | 1.774 | 1.444 |
| lua/fib | 1.393 | 1.048 | 1.044 | 1.000 |
| lua/json_bench | 1.062 | 1.215 | 0.899 | 1.084 |
| lua/sunfish | 1.100 | 1.251 | 0.894 | 1.012 |
| lz4/compress | 0.940 | 1.169 | 1.008 | 1.029 |
| lz4/decompress | 1.269 | 1.200 | 1.292 | 1.294 |
| mandelbrot | 0.992 | 0.663 | 0.940 | 0.635 |
| sha256 | 1.116 | 1.081 | 1.007 | 0.953 |
| sqlite | 1.018 | 1.256 | 0.999 | 1.220 |
| stream/Add | 1.275 | 1.224 | 1.304 | 1.339 |
| stream/Copy | 0.992 | 0.998 | 1.005 | 0.998 |
| stream/Scale | 1.100 | 1.090 | 1.102 | 1.095 |
| stream/Triad | 1.405 | 1.429 | 1.381 | 1.554 |
| **Metric geomean** | **1.169** | **1.306** | **1.108** | **1.258** |
| Workload geomean | 1.146 | 1.251 | 1.062 | 1.180 |

Workload geomean first combines the submetrics of LZ4, funcref and STREAM;
the three Lua programs remain three separate workloads. Both totals are shown
so a large microbenchmark gap or four STREAM rows cannot be hidden.

## Fast iteration against this anchor

1. Keep the fixed comparison source SHA and per-SKU target ratios above.
2. Use the ordinary `dev/**` paired regression CI for compiler candidates.
3. For a confirmed throughput gain `p` against the anchor, project the new
   gap as `old_gap / (1 + p/100)`. If elapsed time changes by `d%`, use
   `old_gap * (1 + d/100)` instead. Do not mix the two conventions.
4. When comparing adjacent revisions, multiply their speedup factors. When
   each candidate is compared to the same main/merge-base, use only the newest
   comparison; adding those percentages double-counts earlier changes.
5. A projection is a planning estimate. CPU differences and performance
   interactions do not disappear when numbers are chained. Use one final
   same-host competitor snapshot to substantiate the closing claim.

For example, LZ4 decompress on 9V74 starts at 1.291× Cranelift time. A
confirmed +20% Nano throughput gain would project 1.076×; it still needs
about another 7.6% throughput to match. Keep every original row in the report.

The workflow is now manual-only. No PR, main, or ordinary dev push rebuilds
V8/Cranelift for these snapshots.

## Measurement limits

- The two WASI jobs drew AMD EPYC 7763 (Zen 3) and 9V74 (Zen 4); neither is
  an Intel result. Do not infer all-x64 parity from either.
- Engine order is Nano/CL/V8 on 7763 and V8/CL/Nano on 9V74. Order and CPU
  vary together, so their effects cannot be separated from these two draws.
- Wasmtime CLI is pinned to 47.0.2, Node to 24.7.0 (V8 13.6.233.10). This
  is a reproducible comparison set, not a claim to use the newest engines.
- Node uses its normal tiering configuration. The guest workload is timed;
  compilation/startup is a separate question. This instrument does not prove
  a particular optimized-tier machine-code shape without a code dump.
- The legacy Node harness retains official-duration CoreMark while the Python
  harness uses its bounded mode. Their score semantics are comparable, but
  measurement duration differs; treat a small gap on that row as provisional.
- Missing measurements are failures, not an invitation to reduce the denominator.

## Official execution corpus (20 cases)

| CPU | Nano / Cranelift elapsed time | Nano / V8 elapsed time |
|---|---:|---:|
| AMD EPYC 7763 | 1.0351 | 1.1489 |
| Intel Xeon Platinum 8573C | 1.0377 | 1.1673 |

The complete table follows. No cases are omitted. Ratios above 1 mean Nano is
slower. The resolved suite uses Wasmtime 47.0.2 and the `v8` Rust crate 150.4.0;
this V8 integration differs from the Node version in the WASI comparison.
Both jobs resolved the same lockfile, preserved as `corpus-Cargo.lock`
(SHA-256 `7265151339807b2e3acd52f8a16490e9a773efe31653fe4cf3e8e5a299fa215b`).

| Case | 7763 / CL | 7763 / V8 | Intel / CL | Intel / V8 |
|---|---:|---:|---:|---:|
| argon2/10000 | 1.580 | 1.502 | 1.160 | 1.265 |
| bulk-ops/5000 | 1.160 | 1.143 | 1.321 | 1.243 |
| compression/152089 | 1.098 | 0.972 | 1.059 | 0.947 |
| counter-global/500000 | 1.001 | 0.994 | 0.974 | 1.000 |
| counter-local/1000000 | 0.994 | 0.995 | 0.986 | 0.994 |
| counter-param/1000000 | 1.004 | 0.998 | 0.996 | 1.015 |
| fibonacci-iter/2000000 | 0.974 | 0.991 | 1.000 | 0.994 |
| fibonacci-rec/30 | 0.654 | 1.507 | 0.867 | 1.442 |
| fibonacci-tail/1000000 | 0.297 | 1.987 | 0.738 | 1.961 |
| json_parse/1727205 | 1.284 | 1.054 | 1.089 | 0.855 |
| mandelbrot/150 | 1.000 | 0.927 | 1.007 | 0.989 |
| matrix_mul/400 | 1.015 | 0.811 | 1.049 | 0.753 |
| nbody/400 | 1.356 | 1.137 | 1.252 | 1.011 |
| prime_sieve/10000000 | 1.236 | 0.871 | 1.237 | 0.918 |
| regex_redux/10245 | 1.502 | 1.365 | 0.997 | 0.970 |
| reverse_complement/10245 | 1.298 | 1.197 | 1.452 | 1.391 |
| sort/1000000 | 1.184 | 1.460 | 1.004 | 1.432 |
| spectralnorm/500 | 0.573 | 1.026 | 0.733 | 3.791 |
| tiny_keccak | 1.327 | 1.292 | 0.976 | 1.146 |
| word_count/152089 | 1.367 | 1.326 | 1.138 | 1.055 |

All four jobs completed successfully, with complete fields and no failed
non-skipped steps. No project compiler warning was emitted. The first
superseded run was cancelled after repairing the lint-audit checkout order.
The successful final run is 34026700835, source `fe334819`.

## First optimization targets

- **Argon2 and data movement:** Nano/CL is 1.58 on 7763 and 1.16 on Intel;
  inspect residency, state spills, and scheduling. This is not a universal
  58% gap across x64.
- **Indirect calls:** WASI exported-table needs about +77–78% throughput to
  match CL on both measured AMD generations. Separate resolution/type checks
  from frame/argument transfer, and keep cross-instance/reentry semantics.
- **LZ4 decode and c-ray:** both AMD draws show material gaps (about 27–29%
  and 23–25% vs CL). These are useful real-workload targets for native dumps.
- **Tail loops and sort vs V8:** tail Fibonacci is about 1.96–1.99× V8 time;
  sort is 1.43–1.46×. Investigate parallel moves and hot dependency chains;
  the already integrated compare/select fusion is not a new candidate.
- **FP on Intel:** spectralnorm is 3.79× V8 time on Intel versus 1.03× on
  7763, while Nano is ahead of CL on both. Preserve that SKU distinction
  when choosing codegen experiments.
- **Direct-call microbenchmark:** V8's 6.97–7.98× throughput reflects a loop
  calling a trivial `+1` helper. C `noinline` does not itself establish V8's
  emitted Wasm code shape. Inlining/elimination is a hypothesis to verify,
  not evidence that ordinary Nano calls can simply become eight times faster.

The scores are a baseline, not a delivered optimization. The next compiler
experiment should isolate one mechanism and use dev differential CI; the
competitor builds do not run on ordinary pushes.
