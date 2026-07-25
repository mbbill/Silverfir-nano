# Running Silverfir-nano in wasmi-benchmarks

We use the external
[`wasmi-benchmarks`](https://github.com/wasmi-labs/wasmi-benchmarks)
suite to compare Silverfir-nano with Cranelift, Winch, Singlepass, V8, and
other WebAssembly engines. This document explains how to connect any local
Silverfir-nano checkout to the suite and collect all of its scores.

The external benchmark checkout is measurement infrastructure. Compiler and
runtime changes belong in the Silverfir-nano repository.

## Scores produced by the suite

| Category | Command filter | What is timed | Direction |
|---|---|---|---|
| Execution | `execute/` | Calls to an already instantiated module | Lower is better |
| Startup | `startup/` | Parsing, validation, compilation, linking, and instantiation | Lower is better |
| CoreMark | `coremark` binary | CoreMark's own elapsed-time score | Higher is better |

At the time of writing, a complete Silverfir-nano run contains:

- 20 Criterion execution timings;
- seven Criterion startup timings;
- one dedicated CoreMark score.

Startup is only one part of the suite. Run the execution benchmarks after
compiler or code-generation changes: a change that reduces compilation time
must not be retained if it degrades generated-code performance.

## Choose checkout locations

The two repositories may live anywhere. Choose locations with enough space for
the external suite's build artifacts:

```sh
export SF_NANO_REPO="/absolute/path/to/Silverfir-nano"
export WASMI_BENCH_REPO="/absolute/path/to/wasmi-benchmarks"

git clone https://github.com/wasmi-labs/wasmi-benchmarks.git \
  "$WASMI_BENCH_REPO"
```

Use a `wasmi-benchmarks` revision that contains the Silverfir-nano adapter, such
as the work from
[`wasmi-benchmarks` PR #51](https://github.com/wasmi-labs/wasmi-benchmarks/pull/51).
The commands below assume the checkout exposes a Cargo feature named
`silverfir-nano` and a runtime ID named `silverfir-nano`.

Record both revisions and the toolchain with every result:

```sh
git -C "$SF_NANO_REPO" rev-parse HEAD
git -C "$WASMI_BENCH_REPO" rev-parse HEAD
cargo +1.97.0 -V
rustc +1.97.0 -Vv
```

Silverfir-nano currently pins Rust 1.97.0. Use the repository's pinned
toolchain unless a benchmark campaign deliberately changes it.

## Connect the local Silverfir-nano checkout

In
`$WASMI_BENCH_REPO/runtimes/silverfir-nano/Cargo.toml`, point
`sf-nano-core` at the source being measured. Cargo does not expand shell
variables in TOML, so write the checkout's actual absolute path:

```toml
sf-nano-core = { path = "/absolute/path/to/Silverfir-nano/sf-nano-core" }
```

Do not leave the adapter on an old pinned Git revision when evaluating local
work. Verify the resolved dependency before running:

```sh
cd "$WASMI_BENCH_REPO"
cargo +1.97.0 tree -p rt-silverfir-nano | rg sf-nano-core
```

If the local compiler changes, Cargo will rebuild the adapter and benchmark
runner as needed.

## Adapter requirements

The Silverfir-nano adapter must satisfy all of the following before a result is
called complete.

### Enable every supported case

`SilverfirNano::can_run` must allow:

- all 20 Criterion execution cases;
- all seven startup cases, including FFmpeg and startup CoreMark;
- `ExecuteTestId::CoreMark`, which is used by the dedicated CoreMark runner.

An exclusion left over from an older Nano revision silently removes that score
from the run. Treat a missing result as an adapter/setup problem until the
engine is known not to support the workload.

### Forward real host functions

The adapter's `RuntimeInstance::link_func` records the benchmark's host
callback. When `instantiate` builds Nano `Import` values, it must wrap and call
that recorded callback, translating arguments and results between
`benchmark_utils::Val` and `sf_nano_core::Value`.

Do not replace every registered callback with an inert stub:

- inert callbacks are sufficient for startup cases because they are not
  invoked;
- the dedicated CoreMark run invokes `env.clock_ms`;
- discarding that callback prevents a valid CoreMark score.

Silverfir-nano's `Import::func` accepts capturing closures, so the adapter can
capture the function pointer recorded by the benchmark linker's entry.

### Allow explicit serial compilation

For intrinsic compiler-throughput comparisons, the adapter should install a
Nano `RuntimeConfig` once, before any other sf-nano-core API is used:

```rust
use sf_nano_core::{Config, Engine};

let config = Config::new()
    .parallel_compilation(std::env::var_os("SF_NANO_BENCH_SERIAL").is_none());
let engine = Engine::new(config).expect("engine configuration");
// then instantiate through it: Instance::new(&engine, wasm, &imports)
```

`SF_NANO_BENCH_SERIAL=1` then selects the serial implementation of the same
eager compiler pipeline. It does not enable lazy compilation.

This switch is only needed for compile/startup analysis. Instantiation occurs
outside the timed loop in execution benchmarks.

## Select engines

To measure Silverfir-nano alone:

```sh
export BENCH_FEATURES="silverfir-nano"
```

For the JIT comparison used during performance work:

```sh
export BENCH_FEATURES="silverfir-nano,wasmtime-cranelift,wasmtime-winch,wasmer-cranelift,wasmer-singlepass,v8"
```

V8 is useful for execution comparisons. Its lazy/tiered startup result is not
equivalent to an engine that eagerly compiles every function.

Build the Criterion runner:

```sh
cd "$WASMI_BENCH_REPO"
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  --no-run
```

Building all engines can take several minutes.

## Run all Criterion scores

With `BENCH_FEATURES=silverfir-nano`, this command collects every Criterion
execution and startup score for Nano:

```sh
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- --noplot
```

When comparison engines are enabled, filter to Nano only with:

```sh
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- silverfir-nano --noplot
```

To run all enabled engines and generate the complete comparison dataset, omit
the `silverfir-nano` filter:

```sh
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- --noplot
```

## Run execution scores

Run all execution cases for the enabled engines:

```sh
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- execute/ --noplot
```

Run one case by using its group name:

```sh
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- execute/matrix_mul --noplot
```

The execution timer excludes module compilation and instantiation. Workload
specific setup and teardown are also outside `b.iter`; only the repeated
operation is timed.

## Run startup scores

Run all startup cases with Nano's normal hosted worker policy:

```sh
unset SF_NANO_BENCH_SERIAL
cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- startup/ --noplot
```

The Wasm bytes are loaded and required host imports are registered before the
timed loop. Each iteration times:

```rust
rt.instantiate(&wasm[..]);
```

The result is therefore **compile + instantiate**, not a compiler-only timer.

The seven startup filters are:

```text
startup/bz2
startup/pulldown-cmark
startup/spidermonkey
startup/ffmpeg
startup/coremark
startup/argon2
startup/erc20
```

Run an individual case by replacing `startup/` with its full filter.

### Serial eager compiler comparison

Use serial compilation when comparing intrinsic compiler throughput across
machines:

```sh
SF_NANO_BENCH_SERIAL=1 cargo +1.97.0 bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bench criterion \
  -- startup/ --noplot
```

For a valid cross-engine serial table:

- Nano uses `SF_NANO_BENCH_SERIAL=1`;
- Wasmtime must be built without its `parallel-compilation` feature; the
  benchmark adapter currently uses `default-features = false`, so verify this
  remains true;
- Wasmer Cranelift and Singlepass must call `num_threads(1)` on their compiler
  configuration before constructing the engine.

Report this table separately from normal parallel wall-clock startup. Never
compare parallel Nano against serial Cranelift as a statement about intrinsic
compiler efficiency.

Prefer isolated startup runs for final numbers. SpiderMonkey and FFmpeg can
heat the machine enough to distort the small CoreMark, Argon2, and ERC20 cases
that follow them. Let the machine cool, close unrelated CPU-heavy programs,
and rerun small cases individually.

## Run the CoreMark score

CoreMark is a dedicated score runner, not a Criterion timing:

```sh
cargo +1.97.0 run \
  --profile bench \
  --no-default-features \
  --features "$BENCH_FEATURES" \
  --bin coremark
```

The runner prints one score per enabled engine followed by a JSON summary.
Higher is better. If Silverfir-nano is absent, verify that:

- `ExecuteTestId::CoreMark` is allowed by the adapter;
- `env.clock_ms` is forwarded to the recorded host callback rather than an
  inert stub.

## Read Criterion results

Criterion stores current estimates below:

```text
target/criterion/execute_<case>/<runtime-and-input>/new/estimates.json
target/criterion/startup_<case>/<runtime>/new/estimates.json
```

The point estimates are nanoseconds. For example:

```sh
jq '.mean.point_estimate / 1000000' \
  target/criterion/startup_bz2/silverfir-nano/new/estimates.json
```

To locate every Nano result:

```sh
find target/criterion -path '*/new/estimates.json' \
  | rg '/(execute_|startup_).*silverfir-nano' \
  | sort
```

Use the same Criterion estimate for all engines in a table. Record:

- both repository commit hashes;
- Rust version and host CPU/OS;
- enabled Cargo features;
- serial or parallel compilation policy;
- Criterion point estimate and confidence interval;
- unavailable engine/workload pairs;
- whether a result was an isolated cooldown rerun.

## Validate a performance change

For a focused change:

1. Run the affected Nano benchmark on revision A.
2. Build revision B and repeat the same benchmark.
3. Alternate A/B again after cooldown.
4. Keep the external benchmark revision, toolchain, features, inputs, and
   thread policy identical.
5. Run all 20 execution cases to detect generated-code regressions.
6. Run all seven startup cases to detect workload-specific compile regressions.
7. Run CoreMark when host calls or CoreMark-generated code may be affected.
8. Run Silverfir-nano's correctness checks before retaining the change.

Do not infer a compiler improvement from one warm-machine sample. Keep
parallel startup, serial compiler throughput, execution timings, and CoreMark
scores as separate result categories.
