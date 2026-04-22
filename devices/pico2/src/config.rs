//! sf-nano-core runtime configuration for this board.
//!
//! The bare-metal build of sf-nano-core ships with a zero default
//! (see `sf_nano_core::config::RuntimeConfig::DEFAULT` under
//! `sf_os_none`). The embedder MUST call [`init`] exactly once at
//! startup before any JIT / module instantiation, otherwise the
//! first `CodeBuffer::new()` / `MemInst::new()` call will fail with
//! `"runtime not configured"`.
//!
//! Numbers here must stay in sync with the static arena in
//! `os_shim.rs` (executable arena) and `RUNTIME_HEAP_BYTES` in
//! `heap.rs` (bookkeeping + Wasm linear memory). Design: see
//! `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md` §9.

/// Bytes reserved for the JIT code arena. Backs
/// `sf_os_alloc_executable` in `os_shim.rs`. Sized to hold the
/// compiled native code of a ~30–40 KiB Wasm binary comfortably.
pub const CODE_ARENA_BYTES: usize = 128 * 1024;

/// Maximum 64-KiB Wasm pages a single linear memory may reach.
/// Three pages (192 KiB) fits the aggressive heap budget while leaving
/// room for module metadata and a per-invoke operand stack.
pub const WASM_MEMORY_MAX_PAGES: u32 = 3;

/// Bytes allocated from the heap for the Wasm operand/call stack on
/// every `invoke`. 32 KiB = 4096 u64 slots covers deep recursion and
/// functions with many locals; hosted default is 2 MiB.
pub const WASM_STACK_BYTES: usize = 32 * 1024;

/// Install the runtime configuration. Panics if called more than once
/// — `sf_nano_core::set_runtime_config` is write-once by design.
pub fn init() {
    sf_nano_core::set_runtime_config(sf_nano_core::RuntimeConfig {
        code_arena_bytes: CODE_ARENA_BYTES,
        wasm_memory_max_pages: WASM_MEMORY_MAX_PAGES,
        wasm_stack_bytes: WASM_STACK_BYTES,
    })
    .expect("sf-nano-pico2 config already initialized");
}
