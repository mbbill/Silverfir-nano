# Debugging Guide

This page is the practical entry point for debugging `sf-nano` today:

- how to run each backend
- how to run spec tests
- what `native` and `reference` mean
- how to get static native dumps and runtime profiles
- where the other debug helpers fit

## Quick Start

Build a normal release CLI:

```bash
cargo build --release --bin sf-nano-cli
```

Run the interpreter path:

```bash
./target/release/sf-nano-cli --backend base benchmarks/wasi/coremark/coremark.wasm
```

Run the native path:

```bash
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Run the debug-only native reference path:

```bash
cargo run --bin sf-nano-cli -- --backend native --emu benchmarks/wasi/coremark/coremark.wasm
```

Run native spectest:

```bash
cargo run --bin sf-nano-spectest -- --backend native
```

Run the core library regression loop used most often during bring-up:

```bash
cargo test -p sf-nano-core --features micro-jit --lib
cargo run --bin sf-nano-spectest -- --backend native
```

## Backend Modes

The CLI accepts:

- `--backend base`
- `--backend fusion`
- `--backend native`
- `--backend auto`
- `--emu` for the debug-only native emulator backend

Important current behavior:

| Mode | What it does today | Notes |
|---|---|---|
| `base` | Interpreter path | This is the stable “just run through interpreter lowering/finalization” mode. |
| `fusion` | Currently behaves like `base` | The flag still exists, but interpreter backend normalization currently maps fusion back to base until fusion is re-enabled on the refactored pipeline. |
| `native` | Native backend | On AArch64 release builds this is the real ARM64 backend. |
| `auto` | Resolve best available backend | Today that means `native` if `micro-jit` is compiled in, otherwise `base`. |

Two details matter:

1. The CLI default is `native`, not `auto`.
2. In normal workspace builds, `sf-nano-core` enables `micro-jit` by default, so `native` is usually available without extra feature flags.
3. `--emu` is only accepted in debug builds. Release builds reject it.

## Native vs Reference

### Native backend

Goal:

- execute through the shared frontend pipeline
- lower through target-independent `NativeIR`
- use a real architecture backend where available

Today that means:

- on AArch64, normal `--backend native` execution uses the ARM64 backend
- on non-AArch64, `native` is only available in debug builds through the emulator backend

### Reference backend

Goal:

- validate `NativeIR` semantics
- provide a non-ISA fallback implementation
- serve as a correctness oracle while real backends come up

What it is not:

- it is not a public CLI backend mode
- there is no `--backend reference`
- it is not enabled in release builds

How to enable it:

```bash
cargo run --bin sf-nano-cli -- --backend native --emu path/to/module.wasm
cargo run --bin sf-nano-spectest -- --backend native --emu
```

If you need to reason about reference behavior, treat it as a debug-only native-validation backend.

## Runtime Line

Both CLI and spectest print one runtime line before execution:

```text
[runtime] interpreter
[runtime] micro-jit backend=arm64
[runtime] micro-jit backend=reference
```

This is the intended high-level signal:

- interpreter vs micro-jit
- if micro-jit, which backend is active

## Interpreter Path

Use the interpreter explicitly with:

```bash
./target/release/sf-nano-cli --backend base path/to/module.wasm
```

This path is always useful when you want:

- a non-native baseline
- to compare behavior against `native`
- to narrow a regression to “shared lowering/finalization” vs “ARM64 lowering/runtime”

Current fusion status:

- `--backend fusion` is accepted only if the `fusion` feature is compiled in
- even then, the interpreter build path currently normalizes fusion back to base

So if you see a difference between `base` and `native`, do not assume there is a live fusion backend in the middle today.

## Spectest

Normal command:

```bash
cargo run --bin sf-nano-spectest -- --backend native
```

Useful variants:

```bash
cargo run --bin sf-nano-spectest -- --backend base
cargo run --bin sf-nano-spectest -- --backend native if
cargo run --bin sf-nano-spectest -- --backend native --emu if
cargo run --bin sf-nano-spectest -- --backend native path/to/test.wast
```

Notes:

- If `TESTSUITE_DIR` is set, spectest uses it.
- Otherwise it falls back to `target/webassembly-testsuite-2.0`.
- `--log-level trace|debug|info|warn|error` controls runner verbosity.
- `RUST_BACKTRACE=1` is useful when chasing an unexpected panic inside spectest.

Example:

```bash
TESTSUITE_DIR=$PWD/target/webassembly-testsuite-2.0 \
RUST_BACKTRACE=1 \
cargo run --bin sf-nano-spectest -- --backend native --log-level info if
```

## Static Native Dump

The native backend can now emit a static compile-time dump with exactly two files:

- `native_index.txt`
- `native_code.bin`

Enable it with `SF_NATIVE_DUMP_DIR`:

```bash
SF_NATIVE_DUMP_DIR=/tmp/coremark-native-dump \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

This writes:

- `/tmp/coremark-native-dump/native_index.txt`
- `/tmp/coremark-native-dump/native_code.bin`

What they contain:

- `native_index.txt`
  - module header
  - region table for all emitted native regions
  - symbol names used by profiling tools
  - runtime address ranges
  - file offsets into `native_code.bin`
  - per-function planned groups
  - per-function full LIR
  - per-function full NativeIR
- `native_code.bin`
  - concatenated machine code bytes for the compiled module

Recommended workflow:

1. record or inspect a hotspot symbol in `samply-for-ai`
2. search that symbol in `native_index.txt`
3. read the function’s LIR and NativeIR sections
4. if needed, query the assembly for the same symbol from the profile

Example symbols now look like:

- `jit::main::func6::b80__helper_t_i32load_move_helper_t_i32load_branch`
- `jit::main::func6::b80_call21_cont_f9`

## JIT Map and Jitdump

### Address map

Set `SF_JIT_MAP` to write a simple address-to-symbol map while native code is recorded:

```bash
SF_JIT_MAP=/tmp/coremark.map \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

This is useful for quick grepping and rough address correlation.

### Jitdump for samply

Set `SF_JITDUMP=1` when recording with `samply-for-ai`:

```bash
SF_JITDUMP=1 \
samply-for-ai record --save-only --output /tmp/coremark-native-profile.json.gz -- \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Optional:

- `SF_JITDUMP_DIR=/path/to/dir` changes where the `jit-<pid>.dump` file is written

Useful follow-up queries:

```bash
samply-for-ai query --profile /tmp/coremark-native-profile.json.gz hotspots --limit 40
samply-for-ai query --profile /tmp/coremark-native-profile.json.gz asm "jit::main::func6::b80_call21_cont_f9"
```

Use `native_index.txt` together with these symbols. `samply` gives runtime hotness; `native_index.txt` explains what the generated code means.

## Function Trace

For backend-vs-backend trace comparison, use the dedicated function trace workflow in [FUNCTION_TRACE_DEBUGGING.md](./FUNCTION_TRACE_DEBUGGING.md).

Build:

```bash
cargo build --release -p sf-nano-cli --features function-trace
```

Record:

```bash
SF_FUNCTION_TRACE=/tmp/base.trace \
./target/release/sf-nano-cli --backend base benchmarks/wasi/coremark/coremark.wasm

SF_FUNCTION_TRACE=/tmp/native.trace \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Compare:

```bash
diff -u /tmp/base.trace /tmp/native.trace
```

Extra knob:

- `SF_FUNCTION_TRACE_MEMORY=1` also hashes memory in each event; use only when needed because it is more expensive

## Common Debug Loops

For a disciplined performance-improvement process, including measurement rules,
IR/assembly proof requirements, and landing criteria, see
[NATIVE_OPTIMIZATION_WORKFLOW.md](./NATIVE_OPTIMIZATION_WORKFLOW.md).

### Validate native correctness first

```bash
cargo test -p sf-nano-core --features micro-jit --lib
cargo run --bin sf-nano-spectest -- --backend native
```

### Compare base vs native on one workload

```bash
./target/release/sf-nano-cli --backend base benchmarks/wasi/coremark/coremark.wasm
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

### Profile a native regression

```bash
SF_NATIVE_DUMP_DIR=/tmp/coremark-native-dump \
SF_JITDUMP=1 \
samply-for-ai record --save-only --output /tmp/coremark-native-profile.json.gz -- \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Then use:

- `/tmp/coremark-native-dump/native_index.txt`
- `/tmp/coremark-native-dump/native_code.bin`
- `samply-for-ai query ... hotspots`
- `samply-for-ai query ... asm "<symbol>"`

## Useful Environment Variables

| Variable | Purpose |
|---|---|
| `TESTSUITE_DIR` | Override the WABT/spec testsuite location for `sf-nano-spectest` |
| `RUST_BACKTRACE=1` | Show backtraces on unexpected panics |
| `SF_NATIVE_DUMP_DIR` | Write `native_index.txt` and `native_code.bin` |
| `SF_JIT_MAP` | Write the native address map |
| `SF_JITDUMP=1` | Emit jitdump records for profiling tools |
| `SF_JITDUMP_DIR` | Override jitdump output directory |
| `SF_FUNCTION_TRACE` | Record sparse function-boundary traces |
| `SF_FUNCTION_TRACE_MEMORY=1` | Add memory hashing to function traces |

## Practical Rules

- Use `base` first when you need a semantic baseline.
- Use `native` when debugging ARM64 codegen, native runtime, or performance.
- Treat `reference` as an internal NativeIR validation backend, not a normal user mode.
- Use `native_index.txt` for static meaning and `samply-for-ai` for runtime hotness.
- If `fusion` comes up in old notes or scripts, remember that current interpreter normalization still maps it back to base.
