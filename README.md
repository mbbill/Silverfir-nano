# Silverfir-nano

A lightweight WebAssembly JIT engine that competes with production-grade optimizing compilers.
On Apple M4, Silverfir-nano **beats Wasmtime's Cranelift** on CoreMark (184%), Lua fib (471%), and two other benchmarks, while going **7-7 against V8 TurboFan**. It reaches 91-99% of Cranelift on LZ4 compress, SHA-256, and STREAM Add.

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

| Category              | vs Cranelift        | vs V8 TurboFan     |
|-----------------------|---------------------|---------------------|
| Integer / Control Flow | 75-184% (wins CoreMark) | 68-118% (wins SHA-256) |
| Lua interpreter       | 134-471% (wins all 3) | 38-121% (wins fib) |
| Floating point        | 39-73%              | 37-173% (wins mandelbrot) |
| Memory (STREAM)       | 81-99%              | 104-220% (wins all 4) |
| **Overall**           | **SF wins 4, CL wins 10** | **SF wins 7, V8 wins 7** |

See [full benchmark results](benchmarks/wasi/RESULTS.md) for details.

## Highlights

- **Competes with optimizing JITs** — beats Cranelift on 4 benchmarks, ties V8 at 7-7
- **184% of Cranelift on CoreMark** — an interpreter-heavy benchmark where dispatch quality dominates
- **99.4% of Cranelift on STREAM Add** — nearly matching a full optimizing compiler on memory throughput
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

## Binary Size

| Build | Size | Features |
|-------|------|----------|
| `sf-nano-cli` minimal | **~230 KB** | `no_std`, no WASI, no fusion |
| `sf-nano-cli` full | ~3.0 MB | WASI + micro-JIT + std |

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

# Regenerate benchmark SVGs
python3 benchmarks/wasi/gen_svg.py
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
