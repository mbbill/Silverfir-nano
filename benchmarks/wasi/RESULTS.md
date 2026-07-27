# WASI Benchmark Results

Measured 2026-07-24 on an Apple M4 (macOS), via `run_tests.py` for the native
runtimes and `run_v8.mjs` for Node. Winch was added 2026-07-25 — see
[the note on its column](#the-winch-column) for why it is comparable. The
Silverfir interpreter column was re-measured 2026-07-25 after the dispatch
work (absolute branch targets, and the dispatch counter compiled out); no
other column moved. The wasmi column moved from 1.1.0 to 2.0.0-beta.7 on
2026-07-26 — see [the note on it](#the-wasmi-20-column), which changes the
interpreter-tier story more than any engine change in this file has.

| Runtime | Version | Class |
|---|---|---|
| Silverfir (JIT) | this tree, `--release` | compiled |
| Silverfir (interpreter) | this tree, `--interp` | interpreted |
| Cranelift / wasmtime | 47.0.2 (built from source, rustc 1.97.0) | compiled (optimizing) |
| V8 / Node.js | Node 25.9.0, V8 14.1.146 | compiled (TurboFan) |
| Winch / wasmtime | 47.0.2, same build, `-C compiler=winch` | compiled (baseline) |
| wasm3 | `~/Dev/wasm3` build-release | interpreted |
| wasmi | 2.0.0-beta.7 (`cargo install wasmi_cli --version 2.0.0-beta.7`) | interpreted |

Winch is wasmtime's **baseline** compiler: single-pass, built to compile fast
rather than to run fast. It shares the compiled chart with the optimizing tier
because it is a compiler, but it is not trying to win that comparison — it
trades roughly half the throughput for much lower compile latency, and the
charts show the throughput half of that trade only.

Charts: [README.md](README.md) · Internal tracking dashboard: https://mbbill.github.io/Silverfir-nano/dev/bench/

## How to read these numbers

**Every metric is a rate, and higher is always better.** No time-based metrics
remain: each benchmark self-times to a wall-clock target (2 s by default) and
reports work per second, so the same binary costs about the same wherever it
runs. That is also why a runtime being 7× slower no longer means a 7× longer
benchmark run.

Silverfir JIT, Cranelift, V8 and Winch are the **mean of two interleaved
rounds**, with the machine allowed to cool between runs; run-to-run spread was
≤3% on every benchmark and ≤1% on most, the exceptions being Winch's CoreMark
(5.7%) and Lua fib (4.1%). The three interpreters are single runs.

### The Winch column

Winch was measured a day after the rest of the table, so it was run with a
**Cranelift control** in the same session — the same binary, target and
harness that produced the published Cranelift column — to test whether the two
sessions are comparable rather than assume it. The control reproduced 14 of 15
metrics within ±2.7%, and `lua/sunfish` reproduced exactly (11,558), which is
the quantisation described in the caveats below showing up as expected. The one
row that did not reproduce is `lua/json`, which came back 10% low (27,917 vs
31,019) — so treat the Lua json row as the noisiest in this file, for every
engine, and do not read small differences there.

### The wasmi 2.0 column

wasmi was upgraded from 1.1.0 to 2.0.0-beta.7 on 2026-07-26 and re-measured
the same way Winch was: **four interleaved rounds** of the whole suite, wasmi
1.1.0 and 2.0.0-beta.7 alternating within each round, so both saw the same
machine state. Each cell is the best of its four rounds — interference on this
machine is one-sided, so the maximum is the least-contaminated estimate, and
the four rounds agreed within 1.00–1.03× on every metric.

The 1.1.0 control reproduced its own published column at **0.99–1.02× on all
15 metrics**, which is what licenses dropping the new numbers into a table
measured two days earlier. An earlier attempt that day did not clear this bar:
CoreMark read 28% low with a 1.51× spread across rounds while the machine was
loaded, and was discarded. If you re-measure, check the control before reading
the column.

2.0.0-beta.7 is a **pre-release**. It was built with `cargo install wasmi_cli
--version 2.0.0-beta.7`, whose default features include `auto-dispatch`; on
aarch64 at `opt-level` 3 that resolves to wasmi's tail-call dispatch, so this
is its fast configuration and not a portable-dispatch fallback. A nightly
build with `--no-default-features --features run,wasi,wat,validate,memory64,
auto-dispatch,unstable` — the `unstable` variant of that dispatch — was also
measured over four rounds and landed within 0.99–1.03×, so there is no faster
wasmi build being left on the table here.

Against 1.1.0, 2.0 is faster on 8 of 15 metrics and slower on 4, median 1.23×.
The gains are concentrated in the STREAM arithmetic kernels (Scale 1.92×, Triad
1.74×, Add 1.67×), the float pair (mandelbrot 1.65×, c-ray 1.61×) and SHA-256
(1.61×); the regressions are LZ4 decompress (0.78×), CoreMark (0.90×) and
sqlite (0.93×). The `wasmi_cli` binary is named `wasmi` in 2.0, not
`wasmi_cli`.

**These numbers share no baseline with any earlier revision of this file.** All
benchmark binaries were regenerated from source on 2026-07-24 with the current
wasi-sdk clang, and several changed their working-set sizes; same-engine figures
moved substantially for that reason alone. Do not compare across that line —
re-measure both sides.

## Integer / Control Flow

<!-- chart file="benchmark_integer.svg" title="Integer / Control Flow" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoreMark (score) | 44,958 | **45,874** | 44,378 | 21,379 | **8,188** | 4,746 | 3,456 |
| SHA-256 (MB/s) | **250.6** | 216.8 | 212.4 | 128.0 | **40.69** | 29.14 | 30.05 |
| bzip2 (MB/s) | **30.68** | 26.86 | 29.66 | 13.88 | **5.21** | 3.10 | 2.68 |
| LZ4 compress (MB/s) | **926.0** | 916.3 | 911.2 | 587.4 | **313.8** | 206.7 | 150.2 |
| LZ4 decompress (MB/s) | 3,241 | **3,478** | 3,046 | 1,200 | **590.1** | 408.7 | 245.2 |
| sqlite speedtest1 (size/s) | 59.88 | 62.66 | **65.66** | 29.89 | **10.19** | 7.13 | 6.51 |
<!-- endchart -->

## Lua

<!-- chart file="benchmark_lua.svg" title="Lua Benchmarks" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| lua / fib (fib20/s) | 2,747 | **2,915** | 1,876 | 1,253 | **377.1** | 221.6 | 223.5 |
| lua / sunfish (score) | 10,494 | **11,558** | 10,847 | 4,356 | **1,221** | 782 | 951 |
| lua / json (score) | 28,996 | **31,019** | 30,280 | 10,857 | **2,548** | 1,609 | 1,939 |
<!-- endchart -->

## Floating Point

<!-- chart file="benchmark_fp.svg" title="Floating-Point Benchmarks" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandelbrot (Kpixel/s) | **503.3** | 502.6 | 298.9 | 184.0 | **166.3** | 134.0 | 95.42 |
| c-ray (Kpixel/s) | 9,809 | 10,515 | **10,705** | 2,792 | **1,413** | 942.4 | 791.0 |
<!-- endchart -->

## Memory Bound

<!-- chart file="benchmark_memory.svg" title="Memory-Bound Benchmarks (STREAM)" note="STREAM Copy=lowered to memory.copy — measures the engine's bulk copy, not dispatch" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| STREAM Copy (MB/s)¹ | 88,302 | **89,610** | 88,012 | 87,845 | 86,816 | **88,992** | 87,234 |
| STREAM Scale (MB/s) | 61,116 | **65,655** | 31,193 | 33,297 | **8,646** | 6,328 | 7,523 |
| STREAM Add (MB/s) | 68,640 | **69,730** | 38,162 | 29,205 | **7,167** | 6,188 | 6,709 |
| STREAM Triad (MB/s) | 60,211 | **62,247** | 37,802 | 29,154 | **7,084** | 5,600 | 6,313 |
<!-- endchart -->

¹ The current clang lowers STREAM's copy loop to a bulk `memory.copy`, so this
row measures each engine's bulk-copy path — not its dispatch. That is why three
very different compilers land within 1% of each other here. It is kept because
it is still a fair comparison of that path, and it is how we caught the
Silverfir interpreter's bulk copy sitting 28% below wasm3 and wasmi: its
handler moved 8 bytes per iteration. Widening it to 64-byte NEON blocks took
this row from 64,127 to 83,092 MB/s and `memory.fill` from 30,650 to 67,630
MB/s — 2.2×, which puts fill ahead of wasm3's 55,658. It reads 86,816 now,
2.4% off the best interpreter on this row; that last step is within this row's
noise, so read it as "joined the pack", not as a further win. Scale,
Add and Triad are arithmetic loops that cannot become `memcpy`, so they remain
the dispatch-sensitive kernels.

## Summary

**Against the other optimizing compilers, Silverfir is at parity, not ahead.**
Counting best-of-four across the 15 metrics: Cranelift 9, Silverfir 4, V8 2,
Winch 0 — Winch takes no row, which is the expected shape for a baseline
compiler and the reason the count reads the same as it did before it was added.

- Silverfir JIT wins: SHA-256 (+16% over both), bzip2 (+14% vs CL), LZ4
  compress (+1%), mandelbrot (+0.1% — a tie in practice).
- Roughly tied: CoreMark (−2% vs CL), STREAM Copy.
- Behind: the three Lua benchmarks (−6% to −9% vs CL), LZ4 decompress (−7%),
  STREAM Scale (−7%), c-ray (−8% vs V8).
- Clear wins over V8 on numeric code: mandelbrot 1.68×, STREAM Scale 1.96×,
  Add 1.80×, Triad 1.59×, Lua fib 1.46×.

**Winch sits about halfway between the two classes**, which is what a baseline
compiler is for: a median 0.47× of Cranelift and 0.46× of Silverfir's JIT, and
a median 2.93× of Silverfir's interpreter. The spread matters more than the
median — it is 0.27–0.98× of Cranelift, so how much the optimizing tier buys
depends entirely on the kernel:

- Closest to the optimizing tier where the work is not in the compiled code:
  STREAM Copy 0.98× (that row is `memcpy` for everyone).
- Furthest behind on float and pointer-chasing code: c-ray 0.27×, Lua json
  0.35×, LZ4 decompress 0.35×.
- It clears V8 on exactly one row, STREAM Scale (1.07×), where V8 is itself
  well off the pace.
- Its narrowest margin over Silverfir's *interpreter* is mandelbrot, 1.11×
  (it was 1.18× before the interpreter was re-measured) — the float-residency
  work is what makes that row close, and it is the one place a baseline
  compiler's advantage over this interpreter nearly vanishes.

**Against the other interpreters, Silverfir's interpreter wins every
dispatch-sensitive benchmark** — 14 of the 15 metrics, the exception being the
`memory.copy` row above.

| vs | worst | best | median |
|---|---:|---:|---:|
| wasm3 | 1.16× (STREAM Add) | 1.73× (CoreMark) | ~1.47× |
| wasmi 2.0.0-beta.7 | 1.07× (STREAM Add) | 2.41× (LZ4 decompress) | ~1.63× |
| whichever of the two is faster per row | 1.07× (STREAM Add) | 1.73× (CoreMark) | ~1.39× |

wasmi 2.0 splits the field rather than shifting it: it is now the closer rival
on the STREAM arithmetic kernels (Add 1.07×, Triad 1.12×, Scale 1.15×) and on
Lua (sunfish 1.28×, json 1.31×), while wasm3 remains the one to beat on
CoreMark, bzip2, LZ4 and the float pair. Neither interpreter is second place
everywhere, so the honest margin is the third row: **1.07–1.73× over the better
of the two on each benchmark**, median ~1.39×.

The largest margins against wasmi are no longer the float pair but the two
benchmarks where 2.0 regressed against 1.1.0 — LZ4 decompress 2.41× and
CoreMark 2.37×. Mandelbrot and c-ray, which led this comparison at 2.89× and
2.88× against 1.1.0, now read 1.74× and 1.79×.

## Caveats worth knowing

- **The Lua benchmarks on wasmtime run on a 1-second clock.** wasmtime 47
  returns `0.0` from Lua's `os.clock()` (no process-CPU clock), so the scripts
  fall back to `os.time()`. Read naively that is up to 1 s of error — 50% at a
  2 s target. `bench.lua` therefore aligns the start of every measurement to a
  tick edge, bounding the error by the check interval instead: Lua fib reads
  2,863 at a 2 s target vs 2,858 at 10 s, 0.2% apart. A side effect is that
  **both** wasmtime compilers' Lua scores are quantised and repeat exactly —
  Cranelift's sunfish came back at 11,558 in two sessions a day apart, and
  Winch's sunfish and json were identical across both its rounds — so repeated
  runs cannot estimate their noise. That is a property of the clock, not
  evidence of unusually stable engines.
- **sqlite reports `size/s`, not its `TOTAL … s` line.** Its work is set by
  `--size`, which the harness picks to hit the target, so elapsed time is ~the
  target on every runtime and carries no information. Work per second does.
- **The interpreter column is a single run**, re-measured 2026-07-25 after the
  dispatch work. Single runs on this suite have been contaminated before: an
  earlier pass was discarded for CPU interference, reading CoreMark 7,187
  against 7,556 — real, and worth ~5%, mostly on CoreMark. Treat any single
  interpreter figure as ±5% until a second round agrees with it.
- **Two interpreter rows moved backwards in that re-measure** and a single run
  cannot say whether that is real: LZ4 decompress 620.0 → 590.1 (−4.8%) and
  STREAM Add 7,369 → 7,167 (−2.7%). The rest moved up, several well past the
  noise band — CoreMark 7,556 → 8,188 (+8.4%), STREAM Scale +11.1%, STREAM
  Triad +7.1%, SHA-256 +6.9%, mandelbrot +6.5%. If the two regressions matter,
  re-run those two rows specifically rather than reading them from this column.
- `sunfish` scores rise with a longer target because its cold first game gets
  amortised away. All runtimes are measured at the same target so the
  comparison is fair, but never compare a 2 s sunfish score to a 10 s one.
- wasmi needs a `--` separator before guest arguments (`--cli-args "--dir . --"`)
  or its argument parser consumes flags like `--memdb` itself.
