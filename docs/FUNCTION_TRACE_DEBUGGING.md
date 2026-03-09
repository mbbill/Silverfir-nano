# Function Trace Debugging

`function-trace` is a sparse debug feature for comparing `base` and `native`
execution without changing runtime behavior.

It is intentionally narrow:

- disabled in normal builds
- dormant unless `SF_FUNCTION_TRACE` is set
- records only function-boundary events today
- compares two runs afterward instead of running both backends in lockstep

## Invariants

- The trace is observational only.
- It must not materialize extra runtime state just for debugging.
- It records only backend-independent compare data:
  - function id
  - call depth
  - entry / exit / trap kind
  - return values on exit
  - globals hash
  - optional memory hash
  - trap text
- It does not compare backend-private cache or register state.

## Build

```bash
cargo build --release -p sf-nano-cli --features function-trace
```

## Record

Record a `base` trace:

```bash
SF_FUNCTION_TRACE=/tmp/base.trace \
/opt/homebrew/bin/timeout 30 \
./target/release/sf-nano-cli --backend base benchmarks/wasi/coremark/coremark.wasm \
> /dev/null 2>/tmp/base.err
```

Record a `native` trace:

```bash
SF_FUNCTION_TRACE=/tmp/native.trace \
/opt/homebrew/bin/timeout 30 \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm \
> /dev/null 2>/tmp/native.err
```

If you also want memory hashing in each event, set:

```bash
SF_FUNCTION_TRACE_MEMORY=1
```

This is more expensive and should only be used when needed.

## Compare

```bash
./target/release/sf-nano-cli trace-compare /tmp/base.trace /tmp/native.trace
```

For workloads that use nondeterministic host state such as time, compare with:

```bash
./target/release/sf-nano-cli trace-compare --ignore-results /tmp/base.trace /tmp/native.trace
```

This still compares function structure, call depth, globals, memory hashes, and
trap state while ignoring returned values that may legitimately vary across
independent runs.

Successful output looks like:

```text
match: 1234 events (left backend=fast, right backend=native, ignore_results=0)
```

On divergence, the command reports the first mismatching event from each side.

## Current Scope

Today the trace is function-boundary only. The intended workflow is:

1. compare the whole program sparsely
2. identify the first bad function / invocation
3. add narrower trace points later only for that function

This keeps the feature low-overhead and minimally intrusive.
