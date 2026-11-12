# wasmi-benchmarks: recorded runs

Everything about running Silverfir-nano inside the external
[`wasmi-benchmarks`](https://github.com/wasmi-labs/wasmi-benchmarks) suite: the
integration document, the raw measurements of a run, the generator that turns
them into a report, and the rendered report.

| Path | What it is |
|---|---|
| `WASMI_BENCHMARKS.md` | How the suite is wired to this repository, what each score means, how the regression CI uses it. |
| `run.sh` | Collects a full run, one Criterion group per process, into a single JSON stream. |
| `make_report.py` | Turns that stream into a self-contained HTML page. |
| `report.html` | The rendered page for the run in `data/`. No external assets; open it directly. |
| `data/criterion-<host>-<date>.json` | The raw cargo-criterion stream for one run. |
| `data/environment-<host>-<date>.json` | Host, toolchain, revisions and sampling parameters for that run. |

`report.html` renders the newest pair (`2026-08-05`, Silverfir-nano `1850f2f9`).
The `2026-08-04` pair is kept beside it as the last run before the interpreter
startup rework; see the note at the end of "The recorded run" about what may and
may not be concluded by comparing the two.

**[Open the report →](https://mbbill.github.io/Silverfir-nano/benchmarks/wasmi_benchmarks/report.html)** — every engine, every workload, in one page.
That is `report.html` in this directory, served by GitHub Pages; the file is
self-contained, so a local copy opens the same way with no network.

`WASMI_BENCHMARKS.md` is the reference for the suite itself. This file covers
only how the recorded runs here were produced and what they say.

## Release 0.7.0 standings

The execute corpus, measured 2026-11-12 (UTC) for release 0.7.0 (`d3e4edec`)
against the engines each tier is judged by, at their current releases:
Wasmtime 50.0.1, the `v8` crate 152.2.0 and wasmi 2.0.0, with wasm3 and
`makepad-stitch` 0.1.0 as the suite has them. Every run is wired the way
`.github/workflows/standings.yml` does it: suite `wasmi-benchmarks@16a3d7c8`,
this checkout's `sf-nano-core`, `ci/wasmi_adapter.patch`, and then
`ci/standings_engines.patch`, which moves the three requirements above off
the suite's July lockfile.

Each figure is the geometric mean, over the cases both engines ran, of that
engine's time divided by Silverfir-nano's: above 1.00× the other engine is
slower. Where a run has two draws both are given. The last column counts the
cases on which Silverfir-nano is the fastest engine of its tier.

**JIT tier** (`silverfir-nano.jit`, `v8`, `wasmtime.cranelift`; 20 cases)

| Run | V8 | Cranelift | Silverfir fastest |
|---|---:|---:|---:|
| x64 Linux CI, EPYC 9V74 | 1.18×, 1.17× | 1.32×, 1.31× | 12 of 20, 11 of 20 |
| x64 Windows CI, EPYC 7763 and EPYC 9V74 | 1.07×, 1.15× | 1.21×, 1.30× | 6 of 20, 13 of 20 |
| ARM64 macOS CI, Apple M1 (virtual) | 1.30×, 1.29× | 1.31×, 1.35× | 12 of 20, 14 of 20 |
| Intel Core i9-14900K, bare metal | 1.18× | 1.32× | 15 of 20, 16 of 20 |
| Apple M5 MacBook Air | 1.25× | 1.30× | 14 of 20 |

**Interpreter tier** (`silverfir-nano.interpreter`, `wasmi-v2.eager.checked`,
`wasm3.eager`, `stitch`; 20 cases, of which wasm3 and stitch run 19 because
neither supports the tail call in `fibonacci-tail`)

| Run | wasmi 2.0 | wasm3 | stitch | Silverfir fastest |
|---|---:|---:|---:|---:|
| x64 Linux CI | 1.76×, 1.52× | 2.41×, 2.08× | 2.11×, 1.89× | 19 of 19, 18 of 19 |
| x64 Windows CI | 1.52×, 1.60× | 2.37×, 2.52× | 1.88×, 2.04× | 19 of 19 |
| ARM64 macOS CI | 1.33×, 1.22× | 1.87×, 1.78× | 2.42×, 2.22× | 19 of 19, 17 of 19 |
| Intel Core i9-14900K, bare metal | 1.37×, 1.36× | 1.97×, 1.95× | 1.73×, 1.72× | 19 of 19 |
| Apple M5 MacBook Air | 1.41× | 1.98× | 2.60× | 19 of 19 |

How each run was made:

- **CI** is the manual `standings` workflow as this change leaves it:
  [JIT run 43158583560](https://github.com/mbbill/Silverfir-nano/actions/runs/43158583560) and
  [interpreter run 43160487110](https://github.com/mbbill/Silverfir-nano/actions/runs/43160487110),
  two draws per platform. Linux pins one core with ASLR off; Windows and
  macOS run unpinned. The interpreter run is red: its measurements completed,
  and its "Write the standings table" step then failed, because
  `ci/standings_report.py` requires every engine to have run every case and
  two of these engines cannot run `fibonacci-tail`. The interpreter figures
  above were computed from the Criterion estimates that run uploaded.
- **The 14900K** is bare metal on Ubuntu 26.04.1, pinned with `taskset -c 2`
  (a performance core) and ASLR off, two draws per tier.
- **The M5** has no fan, so it ran one case at a time, a burst of 10 to 20 s
  with 30 s idle after each, in a single pass on mains power.

**Why the current releases.** The suite's own lockfile has Wasmtime 47.0.2,
`v8` 150.4.0 and wasmi 2.0.0-beta.8. On the 14900K the JIT tier reads the
same either way: 1.16× and 1.33× with the lockfile against 1.18× and 1.32×
here. The interpreter tier does not: the released wasmi takes 1.36× the
interpreter's time where beta.8 took 1.92×, so a ranking against the beta
overstates the lead over wasmi by about 40%.

The JIT's largest margins hold in every CI and 14900K run: `fibonacci-tail`
against Cranelift (2.8 to 6.8×), `argon2` against both (1.9 to 4.5×),
`reverse_complement` against both (1.9 to 2.8×), `sort` against both (1.3 to
2.0×) and `fibonacci-rec` against Cranelift (1.5 to 2.6×). Its recurring
losses are small: `prime_sieve` to Cranelift (0.86 to 1.05, behind in 7 of 8
runs), `fibonacci-rec` to V8 (0.85 to 1.16, 6 of 8) and `tiny_keccak` to
both (0.81 to 1.09). The Windows draw that landed on an EPYC 7763 is the
weakest run for the JIT, 1.07× and 1.21× with 6 of 20 cases fastest; the
other draw on that platform, on an EPYC 9V74, reads like the Linux runs.

This corpus and the repository's own WASI suite do not give the same margin:
there the same release is level with the same versions of both compilers
([WASI results](../wasi/README.md)). Quote each with its suite.

Per case on the 14900K, mean of the two draws, bold marking the fastest
engine:

| case | Silverfir JIT | V8 | Cranelift | V8 / nano | Cranelift / nano |
|---|---:|---:|---:|---:|---:|
| spectralnorm/500 | **8.68 ms** | 9.47 ms | 50 ms | 1.09 | 5.76 |
| argon2/10000 | **16.9 ms** | 55.8 ms | 48.4 ms | 3.30 | 2.87 |
| fibonacci-tail/1000000 | **200 µs** | 201 µs | 556 µs | 1.00 | 2.77 |
| reverse_complement/10245 | **6.38 µs** | 14 µs | 12.3 µs | 2.19 | 1.92 |
| sort/1000000 | **14.5 ms** | 19.4 ms | 25.1 ms | 1.34 | 1.73 |
| fibonacci-rec/30 | 2.07 ms | **1.9 ms** | 3.11 ms | 0.92 | 1.50 |
| json_parse/1727205 | **2.74 ms** | 3.61 ms | 3.08 ms | 1.32 | 1.12 |
| nbody/400 | **4.92 ms** | 6.23 ms | 5 ms | 1.27 | 1.02 |
| matrix_mul/400 | 23.3 ms | 29.1 ms | **23.1 ms** | 1.25 | 0.99 |
| prime_sieve/10000000 | 14.4 ms | 16.8 ms | **12.8 ms** | 1.17 | 0.90 |
| compression/152089 | **6.61 ms** | 7.18 ms | 6.64 ms | 1.09 | 1.00 |
| tiny_keccak | 9.2 µs | **8.73 µs** | 9.89 µs | 0.95 | 1.08 |
| regex_redux/10245 | **16.6 µs** | 17.2 µs | 17.1 µs | 1.04 | 1.03 |
| word_count/152089 | **651 µs** | 667 µs | 665 µs | 1.02 | 1.02 |
| bulk-ops/5000 | **243 µs** | 248 µs | 243 µs | 1.02 | 1.00 |
| mandelbrot/150 | **11 ms** | 11.1 ms | 11.1 ms | 1.01 | 1.00 |
| fibonacci-iter/2000000 | 401 µs | 404 µs | **401 µs** | 1.01 | 1.00 |
| counter-global/500000 | **100 µs** | 101 µs | 101 µs | 1.00 | 1.01 |
| counter-local/1000000 | **200 µs** | 201 µs | 201 µs | 1.00 | 1.00 |
| counter-param/1000000 | **200 µs** | 201 µs | 201 µs | 1.00 | 1.00 |
| **geomean** |  |  |  | **1.18** | **1.32** |

| case | Silverfir interp | wasmi 2.0 | wasm3 | stitch | wasmi 2.0 / nano | wasm3 / nano | stitch / nano |
|---|---:|---:|---:|---:|---:|---:|---:|
| fibonacci-rec/30 | **8.42 ms** | 17.7 ms | 23.7 ms | 16.9 ms | 2.11 | 2.81 | 2.01 |
| regex_redux/10245 | **60.4 µs** | 86.9 µs | 129 µs | 154 µs | 1.44 | 2.13 | 2.56 |
| counter-param/1000000 | **1.27 ms** | 1.56 ms | 2.97 ms | 2.86 ms | 1.23 | 2.35 | 2.26 |
| spectralnorm/500 | **48.8 ms** | 59.5 ms | 114 ms | 109 ms | 1.22 | 2.33 | 2.23 |
| counter-global/500000 | **801 µs** | 1 ms | 1.87 ms | 1 ms | 1.25 | 2.33 | 1.25 |
| argon2/10000 | **324 ms** | 431 ms | 753 ms | 521 ms | 1.33 | 2.32 | 1.61 |
| fibonacci-iter/2000000 | **3.26 ms** | 3.95 ms | 7.45 ms | 4.44 ms | 1.21 | 2.29 | 1.36 |
| reverse_complement/10245 | **87.7 µs** | 117 µs | 199 µs | 147 µs | 1.33 | 2.27 | 1.68 |
| prime_sieve/10000000 | **59 ms** | 73.7 ms | 133 ms | 129 ms | 1.25 | 2.25 | 2.18 |
| matrix_mul/400 | **176 ms** | 264 ms | 297 ms | 384 ms | 1.50 | 1.69 | 2.18 |
| sort/1000000 | **118 ms** | 148 ms | 249 ms | 207 ms | 1.25 | 2.10 | 1.75 |
| json_parse/1727205 | **18.1 ms** | 22.3 ms | 37.4 ms | 33.1 ms | 1.23 | 2.06 | 1.83 |
| nbody/400 | **39.1 ms** | 61.6 ms | 65.7 ms | 76.1 ms | 1.57 | 1.68 | 1.94 |
| tiny_keccak | **61.2 µs** | 87.6 µs | 109 µs | 99.9 µs | 1.43 | 1.78 | 1.63 |
| fibonacci-tail/1000000 | **3.64 ms** | 6.46 ms | – | – | 1.78 | – | – |
| mandelbrot/150 | **27.7 ms** | 40 ms | 48.5 ms | 48.5 ms | 1.44 | 1.75 | 1.75 |
| word_count/152089 | **2.92 ms** | 3.59 ms | 4.9 ms | 4.43 ms | 1.23 | 1.68 | 1.52 |
| compression/152089 | **17.8 ms** | 23 ms | 29 ms | 29.3 ms | 1.29 | 1.63 | 1.64 |
| counter-local/1000000 | **1.27 ms** | 1.6 ms | 1.82 ms | 1.4 ms | 1.26 | 1.44 | 1.11 |
| bulk-ops/5000 | **238 µs** | 302 µs | 273 µs | 284 µs | 1.27 | 1.15 | 1.19 |
| **geomean** |  |  |  |  | **1.37** | **1.96** | **1.73** |

## The recorded run

This is the last full-field run, 29 engine configurations, and it predates
release 0.7.0; the standings above are the current record for the two tiers.


Apple M4 (4 P-cores + 6 E-cores), 16 GB, macOS 26.5.2, `rustc` 1.97.1 stable.
Silverfir-nano `1850f2f9` (main), wasmi-benchmarks `ee13941`, bench profile
(opt-level 3, fat LTO, one CGU), Criterion at 10 samples with a 1 s warm-up and
a 2 s measurement window. 29 engine configurations built; 22 of them appear in
the execution benchmarks, all 29 in startup. 21 execution and 7 startup
workloads, collected 2026-08-05.

Execution, both Silverfir-nano tiers lead their class. Each ratio below is the
geometric mean of the per-case time ratio against the class leader, over the
cases every engine in the class ran:

| Engine | × the class leader | Cases won |
|---|---:|---:|
| `silverfir-nano.jit` | 1.00× | 10/20 |
| `wasmer.cranelift` | 1.20× | 2/20 |
| `v8` | 1.20× | 5/20 |
| `wasmtime.cranelift` | 1.25× | 3/20 |
| `wasmtime.winch` | 2.93× | 0/20 |
| `wasmer.singlepass` | 4.36× | 0/20 |
| `wasmtime.pulley` | 19.09× | 0/20 |

`silverfir-nano.interpreter` is the fastest interpreter on **all 18** cases the
whole interpreter class completed; `wasm3.eager` averages 1.73× its time,
`wasmi-v2.eager.checked` 1.94× and `stitch` 2.45×.

Startup (compile + instantiate, serial compilation) is the other side of that
trade, because both Silverfir-nano tiers compile eagerly and completely before
the first call, while V8 and the `.lazy*` interpreter configurations defer
function bodies to first use. `silverfir-nano.interpreter` is 20th of 27 at
21.5× the fastest interpreter to start (`wasmi-v1.lazy.unchecked`), and
`silverfir-nano.jit` is 5th of 6 in its class.

Among the engines that, like this one, translate every function up front, that
tier now reads:

| Engine | Startup geomean | × Silverfir-nano |
|---|---:|---:|
| `wasmi-v1.eager.checked` | 1.270 ms | 0.68× |
| `wasmi-v2.eager.checked` | 1.742 ms | 0.94× |
| **`silverfir-nano.interpreter`** | **1.859 ms** | **1.00×** |
| `wasm3.eager` | 2.283 ms | 1.23× |
| `wamr` | 3.153 ms | 1.70× |

The interpreter startup path was reworked on 2026-08-05 (PR #30). Comparing
*absolute* times against the earlier recorded run below would overstate it —
unchanged third-party engines move by tens of percent between runs with their
position in the session. Against peers measured in the same run,
`silverfir-nano.interpreter` went from 1.37× `wasm3.eager`'s startup time to
0.81×, from 1.74× to 1.07× against `wasmi-v2.eager.checked`, and from 2.46× to
1.46× against `wasmi-v1.eager.checked` — a consistent ~1.65× peer-relative
gain. `ci/wasmi_performance.py` measured the same change as +58% to +76% on
arm64 and +28% to +64% on x64, which is the number to quote: it compares two
revisions on one machine minutes apart under a probability gate.

`report.html` carries the rest: a per-class summary for execution and for
startup, then every execution workload as a pair of charts — JITs on one axis,
interpreters on their own — with the measured times printed on each bar.
Colours are fixed across every chart and follow the engine, not its rank:
Silverfir-nano is blue, and four reference engines each keep a hue of their own
so the field has landmarks — `wasmtime.cranelift` orange and `v8` green among
the JITs, `wasm3.eager` amber and `wasmi-v2.eager.checked` pink among the
interpreters. Five hues in total, but never more than three in one chart, which
is what keeps each chart inside the colour-vision separation gates. Every startup number is in the page's table rather than 7 more
charts.

## Reproducing it on a Mac

The suite builds nineteen third-party engines from source, several of them C or
C++. Four host-specific things have to be right, or the run fails in ways that
look unrelated to Wasm. `run.sh` handles all four:

**1. Do not let a cross-compiler win `PATH`.** With `wasi-sdk/bin` ahead of
`/usr/bin`, a bare `clang` is a `wasm32-wasi` cross compiler, so `bindgen`
parses WAMR's headers with 32-bit pointers and emits layout assertions that
cannot hold for the 64-bit Rust structs:

```
error[E0080]: index out of bounds: the length is 1 but the index is 16
  ["Size of NativeSymbol"][::core::mem::size_of::<NativeSymbol>() - 16usize];
```

Pin the host toolchain instead — `CC`, `CXX`, `LIBCLANG_PATH`, and a `PATH`
that starts at `/usr/bin`.

**2. Delete a stale `wamrx-sys` build directory after fixing 1.** `cargo clean
-p wamrx-sys` does *not* remove the CMake tree the build script generated under
`target/<profile>/build/wamrx-sys-*/out`. A WAMR C library configured by the
wrong compiler survives the clean and gets linked into the fresh bindings; the
symptom is a run that dies at `execute/fibonacci-tail` with
`ModuleLoad("unsupported opcode 12")` — opcode `0x12` is `return_call`, so the
library was built without `WAMR_BUILD_TAIL_CALL`. Remove the directory:

```sh
rm -rf target/release/build/wamrx-sys-*
```

**3. Fizzy needs one deprecation demoted.** Fizzy uses
`std::basic_string_view<uint8_t>` and compiles with `-Werror`; current Apple
libc++ marks `char_traits<unsigned char>` deprecated, so every translation unit
fails. `CXXFLAGS=-Wno-error=deprecated-declarations` clears it.

**4. Raise the main-thread stack.** With every engine linked into one binary,
WAMR overflows the default 8 MB main-thread stack on `execute/counter-local`
and the process dies with SIGSEGV part-way through the run. A WAMR-only build
of the same revision does not. `ulimit -s 65520` before launching.

**5. Do not measure straight after building.** The suite's `target/` is ~11 GB
and, on a managed Mac, is not in the antivirus exclusion path, so a fresh build
leaves a scan running through the first groups. One run collected this way had
28 engines nobody had touched come out 1.1×–2.5× slower than the previous run;
it was thrown away. Build, let the machine settle, then measure — and record
the load with each group so the conditions can be checked afterwards instead of
assumed. Even then, expect the machine to drift over the ~80 minutes the suite
takes: compare engines *within* a group, never absolute times across runs.

Then:

```sh
WASMI_BENCH_REPO=/path/to/wasmi-benchmarks OUT=/tmp/wasmi-run ./run.sh
```

`run.sh` writes one JSON file per group under `$OUT/groups/`, concatenates them
into `$OUT/criterion.json`, and records progress in `$OUT/status`. Re-running it
skips groups already collected, so a crashed group can be retried on its own.

## Regenerating the page

```sh
python3 make_report.py \
    data/criterion-apple-m4-2026-08-05.json \
    data/environment-apple-m4-2026-08-05.json \
    report.html --standalone
```

Drop `--standalone` to emit a fragment without the `<!doctype>`/`<head>`
skeleton, for embedding. The environment JSON carries the page's title, the
fact chips in its header, and the methodology paragraph, so a new run only
needs its own pair of files.

## What these numbers are, and are not

Each value is a Criterion point estimate from ten samples, on a laptop, in one
session. That is enough to place engines in an order and to see structural
differences such as the eager/lazy startup split. It is **not** the protocol
`ci/wasmi_performance.py` uses to decide whether a change regressed: alternating
A/B process pairs, a probability gate, and a family correction across all 27
metrics. Do not promote a few percent of movement on this page into a claim
about a commit — run the CI job for that.

One deviation from upstream's own method is worth recording: upstream runs the
entire suite in a single process, and this run used one process per group. The
same binary and the same engine ordering are used either way, but a per-group
run gives each group a cold process. Compare runs collected the same way.
