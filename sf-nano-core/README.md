# Silverfir-nano

A compact optimizing WebAssembly JIT and interpreter with a shared embedding
API. The core supports hosted systems and `no_std` environments with an
allocator. Available WebAssembly proposals and execution engines depend on the
target; see [engine and target support](#engine-and-target-support).

## Installation

```toml
[dependencies]
sf-nano-core = "0.6"
```

For an interpreter-only build:

```toml
[dependencies]
sf-nano-core = { version = "0.6", default-features = false, features = ["interp"] }
```

## Embedding

Create an engine, instantiate a binary WebAssembly module, then obtain and call
an exported function:

```rust
use sf_nano_core::{Config, Engine, Instance, Value};

// (module (func (export "answer") (result i32) i32.const 42))
let wasm = b"\0asm\x01\0\0\0\x01\x05\x01\x60\0\x01\x7f\x03\x02\x01\0\x07\x0a\x01\x06answer\0\0\x0a\x06\x01\x04\0\x41\x2a\x0b";

let engine = Engine::new(Config::new()).expect("valid configuration");
let mut instance = Instance::new(&engine, wasm, &[]).expect("valid module");
let answer = instance.get_func("answer").expect("exported function");
let mut results = [Value::I32(0)];
instance.call(&answer, &[], &mut results).expect("successful call");
assert_eq!(results, [Value::I32(42)]);
```

`Instance::new` and `Module::new` validate input. `Module::new_unchecked` is
unsafe and requires the caller to establish full WebAssembly semantic validity;
encoding WAT or merely decoding binary input is insufficient. Module input is
owned, so the source byte slice may be released after construction.

Use `Import::func_typed` for host functions and `Instance::get_export` with
`Import::new` for module exports. `RuntimeWorld` owns linked instances. Function
handles belong to their originating instance; passing one to another instance
returns an error.

Use `Import::alias` to bind the same import under another name without changing
its identity, type or captured state. Host-created exception tags accept numeric
and abstract-reference parameters; tags with concrete module types are linked
through `Instance::get_export` and `Import::new` so their type context is retained.

`Func::to_value` produces an opaque function reference. Engine references are
bound to their `RuntimeWorld`; passing them to another world or using them after
their owner is freed returns an error. Copying a reference does not keep its
owner alive. Null references and labels created with `RefValue::hostref` or
`externref` are portable between worlds. Labels are integers below `2^27`, not
owned Rust objects. Raw engine reference encodings are not public API.

Host callbacks must fill every result with a value of the declared type.
Incorrect types, missing results and foreign references trap before execution
continues. The interpreter currently supports at most eight results from a host
function. `Caller::throw` supports host exceptions; the interpreter can catch
them in Wasm, while the current JIT propagates them to the embedding caller even
when Wasm declares a matching handler.

`Limits::new` and `Limits::new_64` construct immutable memory/table limits and
return `WasmError` for inconsistent bounds. `WASM_PAGE_SIZE` is available at the
crate root. Runtime configuration and errors are also imported from the root
(`Config`, `ConfigError`, `WasmError`).

## Linear memory

`Instance::memory` and `Instance::memory_mut` return guarded views that
dereference to byte slices. Drop the view before calling or instantiating in the
same runtime world. Multiple read views may coexist; a mutable view is exclusive
within its world. Conflicting access returns an error. Views keep the memory
backing alive even if its original instance is freed.

```rust
use sf_nano_core::{Config, Engine, Instance};

// (module (memory 1))
let wasm = b"\0asm\x01\0\0\0\x05\x03\x01\0\x01";
let engine = Engine::new(Config::new()).expect("configuration");
let mut instance = Instance::new(&engine, wasm, &[]).expect("module");
let mut memory = instance.memory_mut().expect("exclusive view");
memory[0] = 42;
drop(memory);
assert_eq!(instance.memory().expect("read view")[0], 42);
```

During a host callback, access guest memory through `Caller::memory` or
`Caller::memory_mut`. An external instance view cannot be acquired while its
world is executing. A callback's memory borrow prevents reentrant guest access
to that memory until the callback returns.

## Features

The supported minimum Rust version is 1.94. CI checks that compiler separately
from current stable. Raising the minimum requires API review.

| Feature | Purpose |
| --- | --- |
| `jit` | Compile WebAssembly to native machine code. |
| `interp` | Run the interpreter generated for the target at build time. |
| `guard-pages` | Enable JIT memory guards where the target supports them; implies `jit`. |
| `wasi` | Include WASI preview1 host imports on supported hosted targets. |
| `memprof` | Enable internal diagnostic hooks; embedding types remain unchanged. |
| `interp-count`, `call-trace`, `jit-debug` | Optional execution and compilation diagnostics. |

The defaults are `jit`, `interp` and `guard-pages`. Disable defaults to select
one engine. At least one engine must be enabled. `Config::tier` selects among
the engines compiled into the package; the default prefers the JIT. Bare-metal
applications must provide an allocator and configure their memory budgets;
hosted default budgets do not apply there. Disable default features for bare
metal: `guard-pages`, `wasi`, `call-trace`, `jit-debug`, and the allocation
profiler require `std`. There is no separate `std` feature.

A bare-metal JIT embedder must also supply executable-memory allocation,
release, write-permission and instruction-cache synchronization hooks; see
[the bare-metal hook contract][bare-metal]. Interpreter-only builds do not need
executable-memory hooks.

WASI imports consume a context built with `WasiContextBuilder`. Separate import
sets isolate arguments, environment and file descriptors. Reusing an import set
intentionally shares that context. The repository's `invoke_export` example
shows file loading, WASI setup and execution. From a repository checkout:

```sh
cargo run --release -p sf-nano-core --features wasi --example invoke_export -- module.wasm export_name
```

That example inherits the host environment and preopens the current directory;
choose the context explicitly when embedding untrusted guests.

The `sf-nano-tracked-alloc` dependency supplies internal allocation helpers.
Embedders ordinarily depend only on `sf-nano-core`; the helper's diagnostic
interface is not part of this crate's embedding API.

## Engine and target support

Both engines have backends for x86_64, ARM64, RV64, RV32, ARMv7-A and Thumb-2.
The interpreter does not execute SIMD or GC instructions. JIT SIMD is available
on ARM64 with NEON and on x86_64 with SSSE3 and SSE4.1; the remaining backends
reject SIMD modules. On x86_64, the JIT checks CPU support at runtime.

For the supported Wasm feature groups, see [the compatibility table][targets].
The host exception and result-count limitations described above apply even
where the guest instruction set supports exception handling and multi-value.

## Diagnostics

`Instance::interpreter_stats` collects an owned `InterpreterStats` snapshot;
`Instance::function_has_native_code` answers a JIT compilation query. Each
method is available with its engine feature and returns `None` for an instance
using the other engine. Snapshots retain no runtime borrow. Handler labels are
diagnostic display text and can change as the interpreter evolves.

Executable buffers manage their own trap registrations when freed. Embedders
do not need a process-wide runtime reset between instances.

## License

Licensed under either MIT or Apache-2.0, at your option. See the included
`LICENSE-MIT` and `LICENSE-APACHE` files.

[targets]: https://github.com/mbbill/Silverfir-nano#webassembly-compatibility
[bare-metal]: https://github.com/mbbill/Silverfir-nano/blob/main/sf-nano-core/src/vm/jit/runtime/os/none.rs
