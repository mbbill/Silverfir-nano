# WASI Benchmark Results

Silverfir-nano 0.7.0 (`d3e4edec`), measured 2026-11-12 (UTC) on two machines:
an Apple M5 MacBook Air and an Intel Core i9-14900K desktop. The Apple M4
record of July 2026 that this file used to hold is in its git history. Nothing
below shares a baseline with it: the machines, every competitor version and two
of the benchmark programs changed, so do not read one against the other.

| Runtime | Version | Class |
|---|---|---|
| Silverfir (JIT) | 0.7.0, `d3e4edec`, `--release` | compiled |
| Silverfir (interpreter) | same binary, `--interp` | interpreted |
| Cranelift / wasmtime | 50.0.1, official release binary | compiled (optimizing) |
| V8 / Node.js | Node 26.13.1, V8 14.6.202.34 | compiled (TurboFan) |
| Winch / wasmtime | 50.0.1, same binary, `-C compiler=winch` | compiled (baseline) |
| wasm3 | 0.9.0 at `0cd38327`, clang, tail-call build | interpreted |
| wasmi | 2.0.0 | interpreted |

| | Apple M5 | Intel Core i9-14900K |
|---|---|---|
| Machine | MacBook Air, 4 P + 6 E cores, 24 GB, no fan | desktop, bare metal, 8 P + 16 E cores |
| OS | macOS 27.0.1 | Ubuntu 26.04.1, Linux 7.0.0 |
| rustc | 1.99.0 | 1.99.1 |
| wasm3 compiler | Apple clang 21.0.0 | clang 22.1.8 |
| Power and placement | mains power, Low Power Mode off, unpinned | governor `performance`, `taskset -c 2` (a P-core), ASLR off (`setarch -R`) |
| Rounds | 2: forward order, then reverse | 3: forward, reverse, forward |
| Gap between sweeps | 75 s idle | 5 s |

Winch is wasmtime's **baseline** compiler: single-pass, built to compile fast
rather than to run fast. It shares the compiled chart with the optimizing tier
because it is a compiler, but it is not trying to win that comparison.

Charts: [README.md](README.md)

## How to read these numbers

**Every metric is a rate, and higher is always better.** Each benchmark
self-times to a wall-clock target (2 s) and reports work per second, so a
sweep of the suite costs about the same on any engine. CoreMark runs in the
harness's `--target-seconds` mode in every cell; that is CoreMark's explicit
non-standard regression mode, not an official CoreMark result.

**Every cell is the mean of interleaved rounds.** One sweep is one engine
running the whole suite, about 30 to 50 s on one core. The engines take turns,
and the next round runs them in the opposite order, so no engine always runs
first or on a warmer machine.

**Apple M5.** The machine has no fan, so each sweep is followed by 75 s idle.
After the two rounds a control sweep repeated the JIT and Cranelift: the JIT
control read 0.94 to 1.02 of its own mean per metric (geomean 0.99) and the
Cranelift control 1.00 to 1.10 (geomean 1.03), so the machine did not drift
downwards over the 28 minutes. The largest ratio between an engine's two
rounds on any metric was 1.03 for the JIT, 1.10 for Cranelift, 1.04 for V8,
1.08 for Winch, 1.06 for the interpreter, 1.06 for wasmi and 1.11 for wasm3.
Read differences inside those bands as ties.

**Intel Core i9-14900K.** The machine also hosts self-hosted CI runners; none
ran a job during any sweep (checked at the start of each). The largest ratio
between an engine's three rounds on any metric was 1.17 for the JIT (STREAM
Add and Triad read 6 to 15% low in the first round), 1.13 for Cranelift, 1.10
for V8, 1.18 for Winch, 1.08 for wasmi and 1.02 for wasm3. One cell is an
outlier: the interpreter's exported-table funcref rate read 114 M calls/s in
the first round and 265 to 267 in the other two, so its mean, 215.1, is the
least certain number in this file.

## Apple M5

### Integer / Control Flow

<!-- chart file="benchmark_integer.svg" title="Integer / Control Flow" machine="Apple M5" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoreMark (score) | **52,125** | 51,455 | 50,109 | 22,332 | **10,021** | 5,209 | 6,502 |
| SHA-256 (MB/s) | **246.9** | 239.9 | 206.9 | 127.2 | **45.88** | 28.48 | 33.77 |
| bzip2 (MB/s) | 31.76 | 30.11 | **33.07** | 16.00 | **5.57** | 3.17 | 4.80 |
| LZ4 compress (MB/s) | 891.9 | 936.0 | **962.7** | 682.6 | **369.7** | 204.0 | 236.5 |
| LZ4 decompress (MB/s) | **3,343** | 3,275 | 3,148 | 1,555 | **648.6** | 405.3 | 490.9 |
| sqlite (iteration/s) | 173,672 | 164,969 | **176,639** | 66,876 | **16,522** | 7,272 | 12,241 |
<!-- endchart -->

### Lua

<!-- chart file="benchmark_lua.svg" title="Lua Benchmarks" machine="Apple M5" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| lua / fib (fib20/s) | 3,102 | **3,347** | 2,074 | 1,541 | **417.5** | 232.3 | 320.2 |
| lua / sunfish (score) | 16,328 | 15,374 | **16,916** | 6,125 | **1,750** | 965.5 | 1,348 |
| lua / json (score) | 35,109 | 31,552 | **35,618** | 14,056 | **3,514** | 1,901 | 2,617 |
<!-- endchart -->

### Floating Point

<!-- chart file="benchmark_fp.svg" title="Floating-Point Benchmarks" machine="Apple M5" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandelbrot (Kpixel/s) | **563.2** | 563.1 | 328.8 | 194.6 | **176.6** | 127.0 | 170.3 |
| c-ray (Kpixel/s) | 11,182 | 11,483 | **12,210** | 3,194 | **1,457** | 811.7 | 973.6 |
<!-- endchart -->

### Memory Bound (STREAM)

<!-- chart file="benchmark_memory.svg" title="Memory-Bound Benchmarks (STREAM)" machine="Apple M5" note="STREAM Copy=lowered to memory.copy — measures the engine's bulk copy, not dispatch" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| STREAM Copy (MB/s)¹ | **165,912** | 156,219 | 165,176 | 165,777 | 124,644 | 164,821 | **165,092** |
| STREAM Scale (MB/s) | 63,772 | 59,544 | **89,186** | 31,306 | **11,000** | 6,891 | 9,488 |
| STREAM Add (MB/s) | 90,763 | 67,453 | **97,113** | 29,839 | **12,076** | 6,767 | 9,048 |
| STREAM Triad (MB/s) | 78,466 | 61,653 | **88,866** | 29,765 | **10,898** | 6,123 | 8,166 |
<!-- endchart -->

### Call Overhead

<!-- chart file="benchmark_calls.svg" title="Call Overhead (funcref)" machine="Apple M5" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| funcref direct (M calls/s) | **4,350** | 865.3 | 4,280 | 287.4 | **378.2** | 172.1 | 246.9 |
| funcref exported-table (M calls/s) | 1,197 | 885.7 | **4,294** | 388.3 | **386.7** | 163.5 | 139.6 |
<!-- endchart -->

¹ The current clang lowers STREAM's copy loop to a bulk `memory.copy`, so this
row measures each engine's bulk-copy path, not its dispatch. That is why very
different compilers land within a few percent of each other here. On the M5
the Silverfir interpreter is the exception: its copy handler reads 124,644
MB/s where every other engine reads 156,000 to 166,000, in both rounds. On the
14900K the same row is level. This is a gap in the interpreter's ARM64 bulk
copy on this core, not noise.

## Intel Core i9-14900K

### Integer / Control Flow

<!-- chart file="benchmark_integer_x64.svg" title="Integer / Control Flow" machine="Intel Core i9-14900K" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoreMark (score) | 40,858 | 36,416 | **42,616** | 20,155 | **7,647** | 4,328 | 5,561 |
| SHA-256 (MB/s) | 189.8 | 182.5 | **191.9** | 146.5 | **34.51** | 19.08 | 24.31 |
| bzip2 (MB/s) | 17.69 | 18.30 | **19.58** | 11.29 | **4.87** | 2.71 | 3.63 |
| LZ4 compress (MB/s) | 752.7 | 838.7 | **866.3** | 577.6 | **284.5** | 179.4 | 223.9 |
| LZ4 decompress (MB/s) | 3,219 | **3,314** | 2,782 | 1,320 | **476.4** | 265.0 | 329.3 |
| sqlite (iteration/s) | 104,685 | 93,194 | **123,765** | 51,712 | **5,062** | 3,594 | 4,368 |
<!-- endchart -->

### Lua

<!-- chart file="benchmark_lua_x64.svg" title="Lua Benchmarks" machine="Intel Core i9-14900K" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| lua / fib (fib20/s) | **2,655** | 2,152 | 1,741 | 1,367 | **412.7** | 245.6 | 299.5 |
| lua / sunfish (score) | 11,839 | 10,902 | **13,458** | 5,676 | **1,100** | 582.7 | 980.0 |
| lua / json (score) | 25,842 | 21,907 | **26,813** | 12,311 | **1,942** | 1,211 | 1,583 |
<!-- endchart -->

### Floating Point

<!-- chart file="benchmark_fp_x64.svg" title="Floating-Point Benchmarks" machine="Intel Core i9-14900K" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandelbrot (Kpixel/s) | **534.2** | 534.1 | 400.3 | 276.4 | **213.2** | 126.0 | 165.3 |
| c-ray (Kpixel/s) | 6,575 | 6,867 | **7,387** | 3,688 | **1,283** | 724.9 | 885.3 |
<!-- endchart -->

### Memory Bound (STREAM)

<!-- chart file="benchmark_memory_x64.svg" title="Memory-Bound Benchmarks (STREAM)" machine="Intel Core i9-14900K" note="STREAM Copy=lowered to memory.copy — measures the engine's bulk copy, not dispatch" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| STREAM Copy (MB/s)¹ | 63,959 | 64,219 | 63,518 | **64,315** | **64,492** | 64,282 | 64,409 |
| STREAM Scale (MB/s) | 58,890 | 53,619 | **62,963** | 26,249 | **11,558** | 4,238 | 5,662 |
| STREAM Add (MB/s) | 56,556 | 59,863 | **68,186** | 31,414 | **9,867** | 4,189 | 5,195 |
| STREAM Triad (MB/s) | 55,553 | 57,111 | **68,087** | 29,130 | **10,819** | 3,677 | 4,783 |
<!-- endchart -->

### Call Overhead

<!-- chart file="benchmark_calls_x64.svg" title="Call Overhead (funcref)" machine="Intel Core i9-14900K" -->
| Benchmark | SF (JIT) | Cranelift | V8 | Winch | SF (interp) | wasm3 | wasmi |
|---|---:|---:|---:|---:|---:|---:|---:|
| funcref direct (M calls/s) | 4,988 | 1,864 | **4,993** | 886.2 | **360.6** | 101.0 | 154.8 |
| funcref exported-table (M calls/s) | 1,091 | 1,167 | **4,992** | 520.8 | **215.1** | 124.9 | 119.2 |
<!-- endchart -->

## Summary

**Against the other optimizing compilers, the JIT is level on this suite: a
little ahead of Cranelift, and level with V8, which takes the most rows.**

| | Apple M5 | i9-14900K |
|---|---|---|
| Best-of-four over the 17 metrics | V8 10, Silverfir 6, Cranelift 1, Winch 0 | V8 13, Silverfir 2, Cranelift 1, Winch 1 |
| Silverfir / Cranelift, geomean of all 17 | 1.18 | 1.09 |
| Silverfir / Cranelift, geomean of the 14 application metrics | 1.06 | 1.04 |
| Silverfir / V8, geomean of all 17 | 0.95 | 0.90 |
| Silverfir / V8, geomean of the 14 application metrics | 1.03 | 0.98 |
| Metrics where Silverfir is ahead of Cranelift | 14 of 17 | 9 of 17 |
| Metrics where Silverfir is ahead of V8 | 7 of 17 | 4 of 17 |

The 14 application metrics leave out the two funcref rows, which are call
micro-benchmarks with ratios from 0.2 to 5.0, and STREAM Copy, which is the
host's `memcpy`. Winch's one row on the 14900K is STREAM Copy, a four-way tie
within 1.3%.

- **CoreMark.** M5: Silverfir 52,125, Cranelift 51,455, V8 50,109, a 1 to 4%
  lead that is inside the round-to-round band. 14900K: V8 42,616, Silverfir
  40,858, Cranelift 36,416.
- **Ahead of V8:** mandelbrot (1.71× on the M5, 1.33× on the 14900K), Lua fib
  (1.50×, 1.52×), LZ4 decompress (1.06×, 1.16×), and SHA-256 on the M5 (1.19×).
- **Behind V8:** the exported-table funcref call, where V8 is 3.6× and 4.6×
  ahead; the STREAM arithmetic kernels (Scale 0.72× on the M5, Add and Triad
  0.82 to 0.93×); c-ray (0.92×, 0.89×); LZ4 compress (0.93×, 0.87×); and on the
  14900K sqlite (0.85×), Lua sunfish (0.88×) and bzip2 (0.90×).
- **Ahead of Cranelift:** the direct funcref call (5.0×, 2.7×); the Lua json
  benchmark (1.11×, 1.18×); on the M5 STREAM Add and Triad (1.35×, 1.27×); on
  the 14900K Lua fib (1.23×), CoreMark (1.12×) and sqlite (1.12×).
- **Behind Cranelift:** LZ4 compress (0.95×, 0.90×) and c-ray (0.97×, 0.96×) on
  both; Lua fib on the M5 (0.93×); the exported-table funcref call and STREAM
  Add on the 14900K (0.93×, 0.94×).

The picture differs on the wasmi-benchmarks corpus, where the same release
leads both compilers on x64 Linux, x64 Windows and ARM64 macOS; see
[the standings there](../wasmi_benchmarks/README.md).

**Winch sits about halfway between the two classes**, which is what a baseline
compiler is for: a median 0.40× of Silverfir's JIT and 0.45× of Cranelift on
the M5, 0.51× and 0.54× on the 14900K, and a median 2.5× and 2.7× of
Silverfir's interpreter.

**Against the other interpreters, Silverfir's interpreter is the fastest on
every dispatch-sensitive benchmark on both machines**: 16 of the 17 metrics on
the M5, the exception being the `memory.copy` row above, and all 17 on the
14900K, where that row is a tie.

| vs | Apple M5: worst | best | median | i9-14900K: worst | best | median |
|---|---:|---:|---:|---:|---:|---:|
| wasm3 | 1.39× (mandelbrot) | 2.36× (funcref exported-table) | 1.80× | 1.41× (sqlite) | 3.57× (funcref direct) | 1.78× |
| wasmi 2.0 | 1.04× (mandelbrot) | 2.77× (funcref exported-table) | 1.34× | 1.12× (lua / sunfish) | 2.33× (funcref direct) | 1.40× |

These ranges leave out STREAM Copy. wasmi 2.0 is the closer rival on every
row but the exported-table call, so the margin over whichever of the two is
faster is wasmi's row in practice. CoreMark reads 1.54× wasmi and 1.92× wasm3
on the M5, and 1.38× and 1.77× on the 14900K. The JIT is a median 6.5× of the
interpreter on the M5 and 5.4× on the 14900K.

## Caveats worth knowing

- **The funcref rows and the sqlite row are new since the July record.** The
  suite gained `funcref.wasm`, two call micro-benchmarks, and replaced
  speedtest1 with `sqlite_bench.wasm`: a fixed in-memory database and a
  repeatable select/update/restore work unit, reported as `iteration/s`.
- **The Lua benchmarks may run on a 1-second clock under wasmtime.** wasmtime
  47 returned `0.0` from Lua's `os.clock()` (no process-CPU clock), so the
  scripts fall back to `os.time()`; whether 50.0.1 still does was not checked.
  `bench.lua` calibrates an approximately 10 ms work chunk, then uses a
  separate tick-aligned window and checks the clock after each chunk, so the
  endpoint overshoot is about one chunk rather than one second either way.
- **The M5 is a fanless laptop and ran unpinned.** macOS has no core pinning,
  and the cooling gaps stand in for a fan; the control sweeps above are the
  evidence that this was enough. Absolute scores on it will not match a
  machine with active cooling under sustained load.
- **The 14900K is a hybrid CPU.** Every engine ran on logical CPU 2, a
  performance core whose sibling thread was idle. A run that lets the
  scheduler place work on an efficiency core will read lower.
- **V8 runs through Node's WASI** (`run_v8.mjs`), the other engines through
  their own command lines, so process startup is not comparable across them;
  the self-timed window excludes it.
- wasmi needs a `--` separator before guest arguments (`--cli-args "--dir . --"`)
  or its argument parser consumes flags like `--memdb` itself.
