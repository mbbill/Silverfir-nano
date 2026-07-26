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

Run the memory tracer:

```bash
cargo memprof --memtrace-output /tmp/run.jsonl -- \
  --backend native benchmarks/wasi/lua/lua.wasm benchmarks/wasi/lua/fib_small.lua
```

Run the core library regression loop used most often during bring-up:

```bash
cargo test -p sf-nano-core --lib
cargo run --bin sf-nano-spectest -- --backend native
```

Run the project validation gates:

```bash
# The gate: workspace tests, full feature matrix, target matrix, spectests, and WASI tests.
python3 scripts/check.py

# Narrow it when iterating.
python3 scripts/check.py --release-only --default-targets
```

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

## Memory Trace

Use the CLI's `memtrace` feature when you want exact raw allocation tracing
from process startup through wasm parsing, validation, instantiation, JIT
compilation, and execution.

What it records:

- `alloc::` heap traffic through the process global allocator
- raw alloc/free/realloc events with timestamps
- interned stack tables with raw PCs
- JIT executable buffer usage
- guard-page linear-memory reservation and committed bytes

Build and run through the cargo alias:

```bash
cargo memprof --memtrace-output /tmp/coremark-mem.jsonl -- \
  --backend native benchmarks/wasi/coremark/coremark.wasm
```

Equivalent manual build/run:

```bash
cargo build --release -p sf-nano-cli --features memtrace --bin sf-nano-cli

target/release/sf-nano-cli \
  --memtrace \
  --memtrace-output /tmp/coremark-mem.jsonl \
  --backend native \
  benchmarks/wasi/coremark/coremark.wasm
```

Useful flags:

- `--memtrace` enables raw tracing
- `--memtrace-output <path>` writes the raw trace log to `<path>`
- `--memtrace-help` shows memtrace-specific CLI help

Outputs:

- one raw JSONL trace file
- one short stderr line that prints the final trace path

Example:

```bash
cargo memprof --memtrace-output /tmp/lua-mem.jsonl -- \
  --backend native benchmarks/wasi/lua/lua.wasm benchmarks/wasi/lua/fib_small.lua
```

Notes:

- `cargo memprof` is just a convenience alias for `sf-nano-cli --features memtrace`
- the runtime trace is intentionally raw; peak finding, curves, categorization,
  and flamegraphs belong in post-processing tools, not in the CLI
- the raw trace can get large because it logs every allocation event
- the raw trace includes:
  - `meta` with command line and schema version
  - `image` records for offline symbolization on macOS
  - `stack` records with interned raw PCs
  - `alloc` / `free` / `realloc`
  - `exec` / `exec_drop`
  - `guard` / `guard_drop`
- if you want cleaner native call stacks, rebuild with frame pointers:

```bash
RUSTFLAGS="-C force-frame-pointers=yes" \
cargo build --release -p sf-nano-cli --features memtrace --bin sf-nano-cli
```

## Memory Trace Analysis

Use the offline analyzer in `tools/memtrace/analyze.py` to turn the raw JSONL
trace into spike candidates, bucketed timeline data, or one selected snapshot.

Best current workflow: run the trace and launch the local viewer in one command:

```bash
python3 tools/memtrace/analyze.py record-view -- \
  --backend native benchmarks/wasi/lua/lua.wasm benchmarks/wasi/lua/fib_small.lua
```

That command will:

1. run `sf-nano-cli` with `memtrace`
2. write one raw trace file
3. build the bucketed curve data
4. start a localhost viewer
5. open the browser

Inside the viewer:

- the top pane is the live-memory step curve
- clicking any point requests an exact snapshot for that timestamp
- the lower pane renders a flamegraph for allocations still live at that moment
- the flamegraph root is grouped by logical memtrace tags first
- the right pane shows top live tags first, then top live stack sites

If you already have a raw trace and only want the viewer:

```bash
python3 tools/memtrace/analyze.py serve /tmp/lua-mem.jsonl
```

Find the biggest spike timestamps:

```bash
python3 tools/memtrace/analyze.py spikes /tmp/lua-mem.jsonl --top 10
```

Emit bucketed timeline data for a step-curve viewer:

```bash
python3 tools/memtrace/analyze.py timeline /tmp/lua-mem.jsonl \
  --bucket-us 1000 \
  --json /tmp/lua-timeline.json
```

Write a standalone HTML curve viewer:

```bash
python3 tools/memtrace/analyze.py curve-html /tmp/lua-mem.jsonl \
  --bucket-us 1000 \
  --out /tmp/lua-curve.html
```

Reconstruct live memory at one selected timestamp:

```bash
python3 tools/memtrace/analyze.py snapshot /tmp/lua-mem.jsonl \
  --time-us 3095866 \
  --top 20
```

Tagged snapshot output now includes:

- `top_live_tags`
- `top_live_stacks`

The main tags are aligned with the native compiler pipeline:

- `native.compile.decode`
- `native.compile.inline`
- `native.compile.prepare`
- `native.compile.lower_inputs`
- `native.compile.lower`
- `native.compile.optimize`
- `native.compile.runtime_module`
- `native.compile.backend_emit`
- `native.compile.publish`

Backend emission also has narrower tags such as:

- `native.compile.backend.blocks`
- `native.compile.backend.edges`
- `native.compile.backend.literal_pool`
- `native.compile.backend.tail`
- `native.compile.backend.patch_fixups`

Interpretation:

- `top_live_tags` tells you which compiler phase still owns memory at the chosen time
- `top_live_stacks` tells you which allocation site inside that phase contributed the bytes
- a row like `tag=native.compile.prepare ...` is usually much more actionable than a raw caller stack alone

Generate collapsed stacks for flamegraph tools from a selected snapshot:

```bash
python3 tools/memtrace/analyze.py snapshot /tmp/lua-mem.jsonl \
  --time-us 3095866 \
  --collapsed-out /tmp/lua-peak.folded
```

Symbolize snapshot frames with `atos` when the trace contains `image` records:

```bash
python3 tools/memtrace/analyze.py snapshot /tmp/lua-mem.jsonl \
  --time-us 3095866 \
  --symbolize \
  --top 20
```

Practical workflow:

1. Record one raw trace with `cargo memprof`.
2. Prefer `record-view` when you want the curve and click-to-flamegraph UI immediately.
3. Use `serve` when you already have a raw trace file.
4. Use `spikes` when you want the peak times in plain text or JSON.
5. Use `timeline` when you want compact curve data for a future custom UI.
6. Use `snapshot --time-us <peak>` when you want a one-off exact dump at a chosen time.
7. Add `--collapsed-out` when you want a flamegraph input file for external tools.
8. Add `--symbolize` when you want function names instead of raw PCs.

If `--symbolize` is too sparse, rebuild with debug info:

```bash
CARGO_PROFILE_RELEASE_DEBUG=1 \
cargo build --release -p sf-nano-cli --features memtrace --bin sf-nano-cli
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
code regions to symbols. It's a dev-tool feature controlled by the `jitdump`
Cargo feature; the module is compiled out of release builds by default.

Build with the feature:

```bash
cargo build --release -p sf-nano-cli --features jitdump
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

For backend-vs-backend trace comparison, use the dedicated function trace
workflow in [FUNCTION_TRACE_DEBUGGING.md](./FUNCTION_TRACE_DEBUGGING.md).

The feature is called `call-trace` on `sf-nano-core` and routed via
`sf-nano-cli`'s own `call-trace` feature.

Build:

```bash
cargo build --release -p sf-nano-cli --features call-trace
```

Record:

```bash
SF_FUNCTION_TRACE=/tmp/arm64.trace \
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm

SF_FUNCTION_TRACE=/tmp/interp.trace \
./target/release/sf-nano-cli --engine interp benchmarks/wasi/coremark/coremark.wasm
```

Compare:

```bash
diff -u /tmp/arm64.trace /tmp/interp.trace
```

Extra knob:

- `SF_FUNCTION_TRACE_MEMORY=1` also hashes memory in each event; use only when needed because it is more expensive

## Common Debug Loops

For a disciplined performance-improvement process, including measurement rules,
IR/assembly proof requirements, and landing criteria, see
[NATIVE_OPTIMIZATION_WORKFLOW.md](./NATIVE_OPTIMIZATION_WORKFLOW.md).

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
cargo memprof --memtrace-output /tmp/coremark-mem.jsonl -- \
  --backend native benchmarks/wasi/coremark/coremark.wasm
```

Then inspect:

- the raw event log in `/tmp/coremark-mem.jsonl`
- the interned stack records in the same file
- an offline post-processing tool or UI for curves, peak reconstruction, and
  flamegraphs at selected times

## Useful Environment Variables

| Variable | Purpose |
|---|---|
| `TESTSUITE_DIR` | Override the WABT/spec testsuite location for `sf-nano-spectest` |
| `RUST_BACKTRACE=1` | Show backtraces on unexpected panics |
| `SF_NATIVE_DUMP_DIR` | Write `native_index.txt` and `native_code.bin` (requires `ir-dump` feature, auto-on in dev builds) |
| `SF_JITDUMP=1` | Emit jitdump records for profiling tools (requires `jitdump` feature) |
| `SF_JITDUMP_DIR` | Override jitdump output directory |
| `SF_FUNCTION_TRACE` | Record sparse function-boundary traces (requires `call-trace` feature) |
| `SF_FUNCTION_TRACE_MEMORY=1` | Add memory hashing to function traces |

## Cross-Architecture Testing

The native backend targets RV64GC, RV32GC, and ARMv7-A in addition to ARM64
and x86_64.
For normal validation, use the unified runner:

```bash
python3 scripts/check.py
python3 scripts/check.py --release-only
```

The runner uses Colima plus `qemu-*-static` on macOS, and local
`qemu-*-static` on Linux/WSL. The manual commands below are only for
debugging the cross-architecture environment directly.

### RV64 prerequisites

Linux/WSL:

```bash
sudo apt-get install -y qemu-user-static
rustup target add riscv64gc-unknown-linux-musl
```

macOS:

```bash
brew install colima docker qemu
colima start
colima ssh -- sudo apt-get update -qq && colima ssh -- sudo apt-get install -y -qq qemu-user-static
rustup target add riscv64gc-unknown-linux-musl
```

### RV64 smoke tests

Use the checked-in helper scripts when debugging RV64 directly:

```bash
./scripts/riscv64-spectest.sh -- if
./scripts/riscv64-run-tests.sh
```

The scripts cross-compile with the static musl RV64GC target and run the CLI
or spectest binary under `qemu-riscv64-static -cpu rv64`.

### RV32 prerequisites

Linux/WSL:

```bash
sudo apt-get install -y qemu-user-static zig
rustup toolchain install nightly
```

macOS:

```bash
brew install colima docker qemu zig
colima start
colima ssh -- sudo apt-get update -qq && colima ssh -- sudo apt-get install -y -qq qemu-user-static
rustup toolchain install nightly
```

RV32 uses `riscv32gc-unknown-linux-musl` with `cargo +nightly -Z build-std`
and `scripts/zig-riscv32-linux-musl-cc.sh`; rustup does not ship a prebuilt
standard library for this target.

### RV32 smoke tests

Use the checked-in helper scripts when debugging RV32 directly:

```bash
./scripts/riscv32-spectest.sh -- if
./scripts/riscv32-run-tests.sh
```

The scripts cross-compile with the static musl RV32GC target and run the CLI
or spectest binary under `qemu-riscv32-static -cpu rv32`.

For WASI validation, the unified runner passes
`--skip-rv32-qemu-timestamp-tests` to `sf-nano-wasitest`. This skips only
`fd_filestat_set`, `path_filestat`, and `symlink_filestat`: qemu-riscv32-static
returns ENOSYS for both timestamp-setting syscall paths observed in this
runner. On macOS, the RISC-V WASI wrapper also copies preopened directories to
VM `/tmp` before invoking QEMU, because Colima's shared `/Users` mount does not
preserve the hard-link behavior the WASI `path_link` test requires.

### ARMv7 prerequisites

Linux/WSL:

```bash
sudo apt-get install -y qemu-user-static
rustup target add armv7-unknown-linux-musleabihf
```

macOS:

```bash
brew install colima docker qemu
colima start
colima ssh -- sudo apt-get update -qq && colima ssh -- sudo apt-get install -y -qq qemu-user-static
rustup target add armv7-unknown-linux-musleabihf
```

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
- Use `native_index.txt` (via `SF_NATIVE_DUMP_DIR` + `ir-dump` feature) for static meaning and `samply-for-ai` with jitdump for runtime hotness.
- If `base`, `fusion`, `micro-jit`, `function-trace`, or `SF_JIT_MAP` come up in old notes, scripts, or external docs: those are obsolete. The current names are `native`, `jit`, `call-trace`, and jitdump respectively.
