# Silverfir-nano

A lightweight WebAssembly JIT engine that competes with production-grade optimizing compilers.
On Apple M4, Silverfir-nano **beats Wasmtime's Cranelift** on multiple benchmarks and **goes head-to-head with V8 TurboFan**, while staying ultra-compact and `no_std`-compatible.

## Performance

All benchmarks on Apple M4 (MacBook Air, macOS 26). Silverfir vs Wasmtime Cranelift (optimizing JIT) and V8 TurboFan (Node.js 25.4).

### Integer / Control Flow

![Integer benchmarks](benchmarks/wasi/benchmark_integer.svg)

### Lua Interpreter

![Lua benchmarks](benchmarks/wasi/benchmark_lua.svg)

### Floating Point

![FP benchmarks](benchmarks/wasi/benchmark_fp.svg)

### Memory Bound (STREAM)

![Memory benchmarks](benchmarks/wasi/benchmark_memory.svg)

### Summary

See the charts above and [full benchmark results](benchmarks/wasi/RESULTS.md) for numbers.

## Highlights

- **Competes with optimizing JITs** — beats Cranelift on CoreMark and Lua benchmarks, beats V8 on STREAM and floating-point workloads
- **Ultra-compact** — the `no_std` core is ~500KB stripped, with zero runtime dependencies
- **Full WebAssembly 2.0** — multi-value, reference types, bulk memory, multiple tables
- **`no_std`** — core library requires only `alloc`; runs on embedded and bare-metal

## Architecture

Silverfir-nano uses a micro-JIT that translates WebAssembly to native ARM64 machine code at function granularity. Key techniques:

- **Native code generation** — direct ARM64 emission, no interpreter dispatch overhead
- **Register allocation** — maps WebAssembly locals and stack to hardware registers
- **Inline operations** — arithmetic, comparisons, and memory access compiled to native instructions
- **Fallback to interpreter** — unsupported patterns fall back gracefully to the fast interpreter

The engine also includes an advanced interpreter with profile-guided instruction fusion, hot-local register caching, and `preserve_none` tail-call dispatch.

## Interpreter-Only Mode

If you need a pure interpreter without JIT (e.g., for platforms without executable memory), check out the `interp` branch:

```bash
git checkout interp
```

This branch includes the fast interpreter with instruction fusion and register caching, but no native code generation.

## Building

```bash
# Default build (micro-JIT)
cargo build --release

# Run a WASI program
./target/release/sf-nano-cli program.wasm [args...]

# Run benchmarks
python3 benchmarks/wasi/run_tests.py

# Run benchmarks in V8 (Node.js)
node benchmarks/wasi/run_v8.mjs

```

## WebAssembly 2.0 Compatibility

Tested against the official [WebAssembly spec testsuite](https://github.com/WebAssembly/spec/tree/main/test):

- Multi-value returns
- Reference types (`funcref`, `externref`)
- Bulk memory operations
- Multiple tables
- Mutable globals import/export

## License

MIT / Apache-2.0
