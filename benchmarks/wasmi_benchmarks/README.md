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

## The recorded run

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
