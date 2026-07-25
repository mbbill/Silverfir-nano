# WASI Benchmark Results

Measured 2026-07-24 on an Apple M4 (macOS), via `run_tests.py` for the native
runtimes and `run_v8.mjs` for Node.

| Runtime | Version | Class |
|---|---|---|
| Silverfir (JIT) | this tree, `--release` | compiled |
| Silverfir (interpreter) | this tree, `--interp` | interpreted |
| Cranelift / wasmtime | 47.0.2 (built from source, rustc 1.97.0) | compiled |
| V8 / Node.js | Node 25.9.0, V8 14.1.146 | compiled (TurboFan) |
| wasm3 | `~/Dev/wasm3` build-release | interpreted |
| wasmi | 1.1.0 (`cargo install wasmi_cli`) | interpreted |

Charts: [README.md](README.md) · Internal tracking dashboard: https://mbbill.github.io/Silverfir-nano/dev/bench/

## How to read these numbers

**Every metric is a rate, and higher is always better.** No time-based metrics
remain: each benchmark self-times to a wall-clock target (2 s by default) and
reports work per second, so the same binary costs about the same wherever it
runs. That is also why a runtime being 7× slower no longer means a 7× longer
benchmark run.

Silverfir JIT, Cranelift and V8 are the **mean of two interleaved rounds**, with
the machine allowed to cool between runs; run-to-run spread was ≤3% on every
benchmark and ≤1% on most. The three interpreters are single runs.

**These numbers share no baseline with any earlier revision of this file.** All
benchmark binaries were regenerated from source on 2026-07-24 with the current
wasi-sdk clang, and several changed their working-set sizes; same-engine figures
moved substantially for that reason alone. Do not compare across that line —
re-measure both sides.

## Integer / Control Flow

<!-- chart file="benchmark_integer.svg" title="Integer / Control Flow" -->
| Benchmark | SF (JIT) | Cranelift | V8 | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|
| CoreMark (score) | 44,958 | **45,874** | 44,378 | **7,556** | 4,746 | 3,813 |
| SHA-256 (MB/s) | **250.6** | 216.8 | 212.4 | **38.08** | 29.14 | 18.42 |
| bzip2 (MB/s) | **30.68** | 26.86 | 29.66 | **4.99** | 3.10 | 2.61 |
| LZ4 compress (MB/s) | **926.0** | 916.3 | 911.2 | **314.9** | 206.7 | 154.8 |
| LZ4 decompress (MB/s) | 3,241 | **3,478** | 3,046 | **620.0** | 408.7 | 311.3 |
| sqlite speedtest1 (size/s) | 59.88 | 62.66 | **65.66** | **10.00** | 7.13 | 6.86 |
<!-- endchart -->

## Lua

<!-- chart file="benchmark_lua.svg" title="Lua Benchmarks" -->
| Benchmark | SF (JIT) | Cranelift | V8 | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|
| lua / fib (fib20/s) | 2,747 | **2,915** | 1,876 | **356.9** | 221.6 | 181.5 |
| lua / sunfish (score) | 10,494 | **11,558** | 10,847 | **1,191** | 782 | 734 |
| lua / json (score) | 28,996 | **31,019** | 30,280 | **2,498** | 1,609 | 1,939 |
<!-- endchart -->

## Floating Point

<!-- chart file="benchmark_fp.svg" title="Floating-Point Benchmarks" -->
| Benchmark | SF (JIT) | Cranelift | V8 | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot (Kpixel/s) | **503.3** | 502.6 | 298.9 | **156.1** | 134.0 | 57.59 |
| c-ray (Kpixel/s) | 9,809 | 10,515 | **10,705** | **1,380** | 942.4 | 490.8 |
<!-- endchart -->

## Memory Bound

<!-- chart file="benchmark_memory.svg" title="Memory-Bound Benchmarks (STREAM)" note="STREAM Copy=lowered to memory.copy — measures the engine's bulk copy, not dispatch" -->
| Benchmark | SF (JIT) | Cranelift | V8 | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|
| STREAM Copy (MB/s)¹ | 88,302 | **89,610** | 88,012 | 83,092 | 88,992 | **89,288** |
| STREAM Scale (MB/s) | 61,116 | **65,655** | 31,193 | **7,779** | 6,328 | 3,848 |
| STREAM Add (MB/s) | 68,640 | **69,730** | 38,162 | **7,369** | 6,188 | 3,968 |
| STREAM Triad (MB/s) | 60,211 | **62,247** | 37,802 | **6,617** | 5,600 | 3,602 |
<!-- endchart -->

¹ The current clang lowers STREAM's copy loop to a bulk `memory.copy`, so this
row measures each engine's bulk-copy path — not its dispatch. That is why three
very different compilers land within 1% of each other here. It is kept because
it is still a fair comparison of that path, and it is how we caught the
Silverfir interpreter's bulk copy sitting 28% below wasm3 and wasmi: its
handler moved 8 bytes per iteration. Widening it to 64-byte NEON blocks took
this row from 64,127 to 83,092 MB/s (~7% off the pack) and `memory.fill` from
30,650 to 67,630 MB/s — 2.2×, which puts fill ahead of wasm3's 55,658. Scale,
Add and Triad are arithmetic loops that cannot become `memcpy`, so they remain
the dispatch-sensitive kernels.

## Summary

**Against the other compilers, Silverfir is at parity, not ahead.** Counting
best-of-three across the 15 metrics: Cranelift 9, Silverfir 4, V8 2.

- Silverfir JIT wins: SHA-256 (+16% over both), bzip2 (+14% vs CL), LZ4
  compress (+1%), mandelbrot (+0.1% — a tie in practice).
- Roughly tied: CoreMark (−2% vs CL), STREAM Copy.
- Behind: the three Lua benchmarks (−6% to −9% vs CL), LZ4 decompress (−7%),
  STREAM Scale (−7%), c-ray (−8% vs V8).
- Clear wins over V8 on numeric code: mandelbrot 1.68×, STREAM Scale 1.96×,
  Add 1.80×, Triad 1.59×, Lua fib 1.46×.

**Against the other interpreters, Silverfir's interpreter wins every
dispatch-sensitive benchmark** — 14 of the 15 metrics, the exception being the
`memory.copy` row above.

| vs | worst | best | median |
|---|---:|---:|---:|
| wasm3 | 1.17× (mandelbrot) | 1.61× (lua fib) | ~1.49× |
| wasmi 1.1.0 | 1.29× (lua json) | 2.81× (c-ray) | ~1.97× |

The float-heavy pair carries the largest margin against wasmi (mandelbrot
2.71×, c-ray 2.81×) — the domain-split float residency work paying off.

## Caveats worth knowing

- **The Lua benchmarks on wasmtime run on a 1-second clock.** wasmtime 47
  returns `0.0` from Lua's `os.clock()` (no process-CPU clock), so the scripts
  fall back to `os.time()`. Read naively that is up to 1 s of error — 50% at a
  2 s target. `bench.lua` therefore aligns the start of every measurement to a
  tick edge, bounding the error by the check interval instead: Lua fib reads
  2,863 at a 2 s target vs 2,858 at 10 s, 0.2% apart. A side effect is that
  Cranelift's Lua scores are quantised and repeat exactly, so repeated runs
  cannot estimate their noise.
- **sqlite reports `size/s`, not its `TOTAL … s` line.** Its work is set by
  `--size`, which the harness picks to hit the target, so elapsed time is ~the
  target on every runtime and carries no information. Work per second does.
- **The interpreter column is a single run**, taken on an idle machine after
  an earlier pass was discarded for CPU interference (that one read CoreMark
  7,187 against 7,556 here — the contamination was real and cost ~5%, mostly on
  CoreMark; every other metric moved under ~2%).
- `sunfish` scores rise with a longer target because its cold first game gets
  amortised away. All runtimes are measured at the same target so the
  comparison is fair, but never compare a 2 s sunfish score to a 10 s one.
- wasmi needs a `--` separator before guest arguments (`--cli-args "--dir . --"`)
  or its argument parser consumes flags like `--memdb` itself.
