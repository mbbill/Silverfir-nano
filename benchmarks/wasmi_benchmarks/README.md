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

**[Open the report → `report.html`](report.html)** — every engine, every
workload, in one self-contained page. GitHub will not render it in the browser;
clone the repository, or download the raw file and open it locally.

`WASMI_BENCHMARKS.md` is the reference for the suite itself. This file covers
only how the recorded runs here were produced and what they say.

## The recorded run

Apple M4 (4 P-cores + 6 E-cores), 16 GB, macOS 26.5.2, `rustc` 1.97.1 stable.
Silverfir-nano `6d26e34c` (tag `0.5`), wasmi-benchmarks `ee13941`, bench profile
(opt-level 3, fat LTO, one CGU), Criterion at 10 samples with a 1 s warm-up and
a 2 s measurement window. 29 engine configurations built; 22 of them appear in
the execution benchmarks, all 29 in startup. 21 execution and 7 startup
workloads, collected 2026-08-04.

Execution, both Silverfir-nano tiers lead their class. Each ratio below is the
geometric mean of the per-case time ratio against the class leader, over the
cases every engine in the class ran:

| Engine | × the class leader | Cases won |
|---|---:|---:|
| `silverfir-nano.jit` | 1.00× | 7/20 |
| `wasmer.cranelift` | 1.08× | 5/20 |
| `wasmtime.cranelift` | 1.12× | 4/20 |
| `v8` | 1.19× | 4/20 |
| `wasmtime.winch` | 2.86× | 0/20 |
| `wasmer.singlepass` | 4.32× | 0/20 |

Startup (compile + instantiate, serial compilation) is the other side of that
trade. V8 leads the JIT class because it compiles lazily; `silverfir-nano.jit`
is 4th of 6 at 34.4× its time, behind `wasmer.singlepass` (4.5×) and
`wasmtime.winch` (8.2×) — but ahead of both optimizing compilers it competes
with on execution, `wasmer.cranelift` (48.2×) and `wasmtime.cranelift` (48.4×).
The interpreter tier is 21st of 23, at 38.3× the fastest interpreter to start
(`wasmi-v1.lazy.unchecked`). Both Silverfir-nano tiers compile eagerly and
completely before the first call, while V8 and the `.lazy*` interpreter
configurations defer function bodies to first use.

`silverfir-nano.interpreter` was the fastest interpreter on all 18 cases the
whole interpreter class completed; `wasm3.eager` averages 1.83× its time,
`wasmi-v2.eager.checked` 1.96× and `stitch` 2.49×.

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
    data/criterion-apple-m4-2026-08-04.json \
    data/environment-apple-m4-2026-08-04.json \
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
