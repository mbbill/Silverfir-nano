# Debugging Guide

This page is the practical entry point for debugging `sf-nano` today:

- how to run the JIT backend
- how to run spec tests
- what `native` and `reference` mean
- how to trace startup/JIT memory usage and spikes
- how to get static native dumps and runtime profiles
- where the other debug helpers fit

## Quick Start

Build a normal release CLI:

```bash
cargo build --release --bin sf-nano-cli
```

Run the native path:

```bash
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Run native spectest:

```bash
cargo run --bin sf-nano-spectest -- --backend native
```

Run the memory profiler:

```bash
cargo memprof --memprof-report /tmp/run.html \
  --backend native benchmarks/wasi/lua/lua.wasm benchmarks/wasi/lua/fib_small.lua
```

Run the core library regression loop used most often during bring-up:

```bash
cargo test -p sf-nano-core --lib
cargo run --bin sf-nano-spectest -- --backend native
```

Run the usual local validation:

```bash
cargo build --workspace
cargo test --workspace
```

The exhaustive multi-platform gates run in GitHub Actions from `ci/`.

## Engines

Which engine runs a module, and which ISA the JIT emits for, are two
different axes. The engine is the runtime choice; the ISA is fixed at build
time by the target triple.

The CLI accepts:

- `--engine jit` (aliases: `--engine native`, `--backend native`)
- `--engine interp` (shorthand: `--interp`)
- `--engine auto`

| Engine | What it does |
|---|---|
| `jit` | Compiles each function to native code before running it. |
| `interp` | Runs the threaded dispatch chain generated for this target at build time. |
| `auto` | The JIT where it is compiled in, the interpreter otherwise. |

Details that matter:

1. `--engine` only accepts engines the binary was built with. Asking for one
   that was left out is an error that lists what this build has; the
   corresponding `Engine` variant does not exist at all in the API, so an
   embedder gets a compile error rather than a runtime one.
2. `jit` and `interp` are both default features of `sf-nano-core`, so both
   are usually available without extra feature flags. `--no-default-features
   --features interp` builds an interpreter-only binary with no JIT in it.
3. The previous `base` (interpreter) and `fusion` backends have been removed;
   the interpreter was rewritten as the folded stack machine.

## Native vs Reference

### Native backend

Goal:

- execute through the shared frontend pipeline
- lower through target-independent `NativeIR`
- use a real architecture backend where available

Today that means:

- on AArch64, normal `--backend native` execution uses the ARM64 backend
- on x86_64, normal `--backend native` execution uses the x86_64 backend
- on RV64GC Linux, normal `--backend native` execution uses the RV64 backend
- on ARMv7-A Linux, normal `--backend native` execution uses the ARMv7-A backend
- on an ISA no backend covers, the build is refused outright: `build.rs`
  names the target and lists the supported ISAs, rather than producing an
  engine that would fail at instantiation

## Runtime Line

Both CLI and spectest print one runtime line before execution:

```text
[runtime] jit backend=arm64
[runtime] jit backend=x86_64
[runtime] jit backend=riscv64
```

This line tells you which concrete backend the JIT resolved to for this run.

## Spectest

Normal command:

```bash
cargo run --bin sf-nano-spectest -- --backend native
```

Useful variants:

```bash
cargo run --bin sf-nano-spectest -- --backend native if
cargo run --bin sf-nano-spectest -- --backend native path/to/test.wast
```

Notes:

- If `TESTSUITE_DIR` is set, spectest uses it.
- Otherwise it falls back to `target/webassembly-testsuite`.
- `--log-level trace|debug|info|warn|error` controls runner verbosity.
- `RUST_BACKTRACE=1` is useful when chasing an unexpected panic inside spectest.

Example:

```bash
TESTSUITE_DIR=$PWD/target/webassembly-testsuite \
RUST_BACKTRACE=1 \
cargo run --bin sf-nano-spectest -- --backend native --log-level info if
```

## Memory Profiling

The CLI's `memprof` feature records allocation ownership and compiler phases,
then writes a self-contained HTML report. Use the cargo alias:

```bash
cargo memprof --memprof-report /tmp/coremark-mem.html \
  --backend native benchmarks/wasi/coremark/coremark.wasm
```

`--memprof-report` enables recording and chooses the output path. Plain
`--memprof` writes to the system temporary directory and prints the final path.
The equivalent manual build and run is:

```bash
cargo build --release -p sf-nano-cli --features memprof --bin sf-nano-cli

target/release/sf-nano-cli \
  --memprof \
  --memprof-report /tmp/coremark-mem.html \
  --backend native \
  benchmarks/wasi/coremark/coremark.wasm
```

The report contains the allocation curve, compiler-phase overlays, executable
and guard-page memory, and point-in-time live allocations grouped by type and
size. It is generated directly by `sf-nano-memprof-report`; there is no raw
JSONL analyzer or separate viewer to run.

## Static Native Dump

The native backend can now emit a static compile-time dump with exactly two files:

- `native_index.txt`
- `native_code.bin`

Release builds need the core `jit-debug` feature; hosted dev builds compile
the IR exporter automatically. Build and enable it with:

```bash
cargo build --release -p sf-nano-cli --features sf-nano-core/jit-debug

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
  - per-function full SSA-IR
  - per-function full NativeIR
- `native_code.bin`
  - concatenated machine code bytes for the compiled module

Recommended workflow:

1. record or inspect a hotspot symbol in `samply-for-ai`
2. search that symbol in `native_index.txt`
3. read the function’s SSA-IR and NativeIR sections
4. if needed, query the assembly for the same symbol from the profile

## Post-Processed Per-Function View

For function-by-function debugging, use:

```bash
python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-native-dump \
  --out-dir /tmp/coremark-postprocessed
```

To restrict output to one function:

```bash
python3 scripts/postprocess_native_dump.py \
  --wasm benchmarks/wasi/coremark/coremark.wasm \
  --dump-dir /tmp/coremark-native-dump \
  --out-dir /tmp/coremark-postprocessed \
  --function 6
```

This writes:

- `module.json`
- `function_map.json`
- `functions/<index>/summary.json`
- `functions/<index>/overview.txt`
- `functions/<index>/wasm_disasm.txt`
- `functions/<index>/wasm_text.wat`
- `functions/<index>/ssa_ir.txt`
- `functions/<index>/machine_ir.txt`
- `functions/<index>/native_disasm.txt`

Recommended workflow:

1. generate the native dump with `SF_NATIVE_DUMP_DIR`
2. run `postprocess_native_dump.py`
3. open `functions/0006/overview.txt` for one stitched view of:
   - Wasm disassembly
   - Wasm text
   - native dump metadata
   - SSA-IR
   - MachineIR
   - native disassembly
4. if you want cleaner diffs, compare the individual files instead of the stitched overview

Notes:

- `wasm_disasm.txt` comes from `wasm-objdump -d`
- `wasm_text.wat` comes from `wasm2wat --generate-names`
- `native_disasm.txt` is generated from `native_code.bin` plus `code_file_off` / `code_size`
- on macOS with Homebrew LLVM, the script wraps `native_code.bin` into a temporary ELF with `llvm-objcopy`, then disassembles per-function ranges with `llvm-objdump`
- on systems with GNU `objdump` / `gobjdump`, the script can also use raw-binary disassembly directly

Example symbols now look like:

- `jit::main::func6::b80__helper_t_i32load_move_helper_t_i32load_branch`
- `jit::main::func6::b80_call21_cont_f9`

## Jitdump for samply

Jitdump emission lets external profilers (samply, perf) resolve JIT-compiled
code regions to symbols. It is one of the two exporters compiled by the
core `jit-debug` feature; `SF_JITDUMP` selects it at runtime.

Build with the feature:

```bash
cargo build --release -p sf-nano-cli --features sf-nano-core/jit-debug
```

Then set `SF_JITDUMP=1` when recording with `samply-for-ai`:

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

## Function (Call) Trace

To record sparse JIT function-boundary events, enable the core `call-trace`
feature. The CLI does not duplicate core-only diagnostic features.

Build:

```bash
cargo build --release -p sf-nano-cli --features sf-nano-core/call-trace
```

Record:

```bash
SF_FUNCTION_TRACE=/tmp/coremark.trace \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
```

Extra knob:

- `SF_FUNCTION_TRACE_MEMORY=1` also hashes memory in each event; use only when needed because it is more expensive

## Common Debug Loops

### Validate native correctness first

```bash
cargo test -p sf-nano-core --lib
cargo run --bin sf-nano-spectest -- --backend native
```

### Compare the two engines on one workload

```bash
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
./target/release/sf-nano-cli --engine interp benchmarks/wasi/coremark/coremark.wasm
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

### Inspect a memory spike

```bash
cargo memprof --memprof-report /tmp/coremark-mem.html \
  --backend native benchmarks/wasi/coremark/coremark.wasm
```

Then inspect:

- the allocation curve and compiler-phase overlays in `/tmp/coremark-mem.html`
- a selected point's live allocations, grouped by type and size

## Useful Environment Variables

| Variable | Purpose |
|---|---|
| `TESTSUITE_DIR` | Override the WABT/spec testsuite location for `sf-nano-spectest` |
| `RUST_BACKTRACE=1` | Show backtraces on unexpected panics |
| `SF_NATIVE_DUMP_DIR` | Write `native_index.txt` and `native_code.bin` (`jit-debug` in release; auto-on in hosted dev builds) |
| `SF_JITDUMP=1` | Emit jitdump records for profiling tools (requires `jit-debug`) |
| `SF_JITDUMP_DIR` | Override jitdump output directory |
| `SF_FUNCTION_TRACE` | Record sparse function-boundary traces (requires core `call-trace`) |
| `SF_FUNCTION_TRACE_MEMORY=1` | Add memory hashing to function traces |

## Cross-Architecture Testing

The native backend targets RV64GC, RV32GC, and ARMv7-A in addition to ARM64
and x86_64. Exhaustive cross-runtime validation runs in GitHub Actions: each
target gets an independent x64 Linux runner and QEMU-user job.

```bash
python3 -m ci.correctness cross armv7
python3 -m ci.correctness cross riscv64
python3 -m ci.correctness cross riscv32
```

Those entry points intentionally support x64 Linux only; CI installs the
required Rust target, QEMU, nightly Rust, and Zig.

RV32 uses `riscv32gc-unknown-linux-musl` with `cargo +nightly -Z build-std`
and `ci/zig-riscv32-linux-musl-cc.sh`; rustup does not ship a prebuilt
standard library for this target.

For WASI validation, the RV32 job passes
`--skip-rv32-qemu-timestamp-tests` to `sf-nano-wasitest`. This skips only
`fd_filestat_set`, `path_filestat`, and `symlink_filestat`: qemu-riscv32-static
returns ENOSYS for both timestamp-setting syscall paths observed in this
runner.

### ARMv7 Step 1: Verify the environment with the interpreter

`--engine interp` runs the dispatch chain generated for ARMv7 at build time.
It exercises the build, the cross toolchain, and the QEMU setup without
involving the JIT's code emission — so a failure here is an environment
problem, not a codegen one.

```bash
# Cross-compile (static musl, no default features to skip guard-pages)
cargo build --target armv7-unknown-linux-musleabihf -p sf-nano-cli --no-default-features --features interp

# Linux/WSL: run under QEMU
qemu-arm-static -cpu cortex-a15 \
  target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
  --engine interp benchmarks/wasi/coremark/coremark.wasm

# macOS: run inside the Colima VM
colima ssh -- qemu-arm-static \
  /Users/$USER/Dev/Silverfir-nano/target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
  --engine interp /Users/$USER/Dev/Silverfir-nano/benchmarks/wasi/coremark/coremark.wasm
```

If this passes, the toolchain and runtime environment are working.

### ARMv7 Step 2: Test the real ARMv7 JIT

The goal is to validate the ARMv7-A native codegen. Drop `--engine interp` so
the CLI uses the JIT backend:

```bash
# Linux/WSL
qemu-arm-static -cpu cortex-a15 \
  target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
  benchmarks/wasi/coremark/coremark.wasm

# macOS
colima ssh -- qemu-arm-static \
  /Users/$USER/Dev/Silverfir-nano/target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
  /Users/$USER/Dev/Silverfir-nano/benchmarks/wasi/coremark/coremark.wasm
```

Run spectests the same way:

```bash
cargo build --target armv7-unknown-linux-musleabihf -p sf-nano-spectest --no-default-features

# Linux/WSL
TESTSUITE_DIR=$PWD/target/webassembly-testsuite \
qemu-arm-static -cpu cortex-a15 \
  target/armv7-unknown-linux-musleabihf/debug/sf-nano-spectest \
  --backend native

# macOS
colima ssh -- qemu-arm-static \
  /Users/$USER/Dev/Silverfir-nano/target/armv7-unknown-linux-musleabihf/debug/sf-nano-spectest \
  --backend native
```

### Notes

- Use `--no-default-features --features jit` to disable `guard-pages`
  (which requires a 64-bit virtual address space) while keeping the JIT
  compiled in.
- The binary is statically linked (musl), so no shared libraries are needed
  inside QEMU.
- Colima mounts the macOS home directory, so host paths work directly there.
- QEMU user-mode translates ARMv7 instructions but uses the host kernel for
  syscalls, so mmap/mprotect behavior may differ from real hardware.

## Practical Rules

- Use `native` for normal runs; it's the only real execution backend today.
- Use `--engine interp` as a second opinion when you suspect a JIT codegen bug: the interpreter runs the same module through an entirely separate execution path.
- Use `native_index.txt` (via `SF_NATIVE_DUMP_DIR` + `jit-debug` in release) for static meaning and `samply-for-ai` with jitdump for runtime hotness.
- If `base`, `fusion`, `micro-jit`, `function-trace`, or `SF_JIT_MAP`
  come up in old notes, scripts, or external docs, they are obsolete. The
  current engine and diagnostic feature names are `native`, `jit`,
  `call-trace`, and `jit-debug`; jitdump is the profiler file format.
