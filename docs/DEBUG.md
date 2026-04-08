# Debugging Guide

This page is the practical entry point for debugging `sf-nano` today:

- how to run the JIT backend
- how to run spec tests
- what `native` and `reference` mean
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

Run the debug-only native reference path:

```bash
cargo run --bin sf-nano-cli -- --backend native --emu64 benchmarks/wasi/coremark/coremark.wasm
```

Run native spectest:

```bash
cargo run --bin sf-nano-spectest -- --backend native
```

Run the core library regression loop used most often during bring-up:

```bash
cargo test -p sf-nano-core --lib
cargo run --bin sf-nano-spectest -- --backend native
```

## Backend Modes

The CLI accepts:

- `--backend native` (alias: `--backend jit`)
- `--backend auto`
- `--emu64` / `--emu32` for the debug-only native emulator backend

| Mode | What it does |
|---|---|
| `native` | Native JIT backend. On AArch64 release builds this is the real ARM64 backend. |
| `auto` | Resolve best available backend. Today that always means `native` because the JIT (`jit` feature) is the only compiled-in execution backend. |

Details that matter:

1. The CLI default is `native`, not `auto`.
2. `jit` is a default feature of `sf-nano-core`, so `native` is usually available without extra feature flags.
3. `--emu64` / `--emu32` are only accepted in debug builds. Release builds reject them.
4. The previous `base` (interpreter) and `fusion` backends have been removed; the interpreter will be rewritten later.

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

- validate `MachineIR` semantics
- provide a non-ISA fallback implementation
- serve as a correctness oracle while real backends come up

What it is not:

- it is not a public CLI backend mode
- there is no `--backend reference`
- it is not enabled in release builds

How to enable it:

```bash
cargo run --bin sf-nano-cli -- --backend native --emu64 path/to/module.wasm
cargo run --bin sf-nano-spectest -- --backend native --emu64
```

Use `--emu32` instead to exercise the 32-bit GP target profile.

If you need to reason about reference behavior, treat it as a debug-only native-validation backend.

## Runtime Line

Both CLI and spectest print one runtime line before execution:

```text
[runtime] jit backend=arm64
[runtime] jit backend=x86_64
[runtime] jit backend=emulator
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
cargo run --bin sf-nano-spectest -- --backend native --emu64 if
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

SF_FUNCTION_TRACE=/tmp/emu64.trace \
./target/release/sf-nano-cli --backend native --emu64 benchmarks/wasi/coremark/coremark.wasm
```

Compare:

```bash
diff -u /tmp/arm64.trace /tmp/emu64.trace
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

### Compare native vs reference emulator on one workload

```bash
./target/release/sf-nano-cli --backend native benchmarks/wasi/coremark/coremark.wasm
cargo run --bin sf-nano-cli -- --backend native --emu64 benchmarks/wasi/coremark/coremark.wasm
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
| `SF_NATIVE_DUMP_DIR` | Write `native_index.txt` and `native_code.bin` (requires `ir-dump` feature, auto-on in dev builds) |
| `SF_JITDUMP=1` | Emit jitdump records for profiling tools (requires `jitdump` feature) |
| `SF_JITDUMP_DIR` | Override jitdump output directory |
| `SF_FUNCTION_TRACE` | Record sparse function-boundary traces (requires `call-trace` feature) |
| `SF_FUNCTION_TRACE_MEMORY=1` | Add memory hashing to function traces |

## Cross-Architecture Testing (ARMv7)

The native backend targets ARMv7-A (32-bit ARM) in addition to ARM64 and x86_64.
To test on an Apple Silicon Mac, cross-compile and run under QEMU user-mode
emulation inside the Colima VM.

### Prerequisites

```bash
brew install colima docker qemu
colima start
colima ssh -- sudo apt-get update -qq && sudo apt-get install -y -qq qemu-user-static
```

### Step 1: Verify the environment with --emu

The `--emu` flag uses the platform-independent emulator backend. It works on
any architecture and confirms the build, the cross toolchain, and the QEMU
setup are all functional — before introducing target-specific JIT issues.

```bash
# Cross-compile (static musl, no default features to skip guard-pages)
cargo build --target armv7-unknown-linux-musleabihf -p sf-nano-cli --no-default-features

# Run with emulator backend via QEMU
colima ssh -- qemu-arm-static \
  /Users/$USER/Dev/Silverfir-nano/target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
  --emu /Users/$USER/Dev/Silverfir-nano/benchmarks/wasi/coremark/coremark.wasm
```

If this passes, the toolchain and runtime environment are working.

### Step 2: Test the real ARMv7 JIT

The goal is to validate the ARMv7-A native codegen, not the emulator.
Drop `--emu` so the CLI uses the real JIT backend:

```bash
colima ssh -- qemu-arm-static \
  /Users/$USER/Dev/Silverfir-nano/target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
  /Users/$USER/Dev/Silverfir-nano/benchmarks/wasi/coremark/coremark.wasm
```

Run spectests the same way:

```bash
cargo build --target armv7-unknown-linux-musleabihf -p sf-nano-spectest --no-default-features

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
- Colima mounts the macOS home directory, so host paths work directly.
- QEMU user-mode translates ARMv7 instructions but uses the host kernel for
  syscalls, so mmap/mprotect behavior may differ from real hardware.

## Practical Rules

- Use `native` for normal runs; it's the only real execution backend today.
- Use `--emu64` / `--emu32` when you suspect ARM64 codegen vs generic MachineIR semantics divergence — the reference emulator runs the same MIR through a non-ISA interpreter.
- Use `native_index.txt` (via `SF_NATIVE_DUMP_DIR` + `ir-dump` feature) for static meaning and `samply-for-ai` with jitdump for runtime hotness.
- If `base`, `fusion`, `micro-jit`, `function-trace`, or `SF_JIT_MAP` come up in old notes, scripts, or external docs: those are obsolete. The current names are `native`, `jit`, `call-trace`, and jitdump respectively.
