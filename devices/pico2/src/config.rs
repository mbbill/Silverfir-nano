//! sf-nano-core runtime configuration for this board.
//!
//! The bare-metal build of sf-nano-core ships with a zero default
//! (see `sf_nano_core::config::RuntimeConfig::DEFAULT` under
//! `sf_os_none`). The embedder MUST call [`init`] exactly once at
//! startup before any JIT / module instantiation, otherwise the
//! first `Instance::new()` / `CodeBuffer::new()` path will fail with
//! `"runtime not configured"`.
//!
//! Numbers here must stay in sync with the static arena in
//! `os_shim.rs` (executable arena) and the heap budget in `heap.rs`
//! (bookkeeping + materialized Wasm linear memory). Design: see
//! `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md`.

/// Bytes reserved for the JIT code arena. Backs
/// `sf_os_alloc_executable` in `os_shim.rs`. The current Pico2 demos
/// fit comfortably here; the saved SRAM goes to the allocator heap.
pub const CODE_ARENA_BYTES: usize = 32 * 1024;

/// Maximum 64-KiB Wasm pages a single linear memory may reach.
/// Three pages (192 KiB) fits the aggressive heap budget while leaving
/// room for module metadata and a per-invoke operand stack.
pub const WASM_MEMORY_MAX_PAGES: u32 = 3;

/// Bytes allocated from the heap for the Wasm operand/call stack on
/// every `invoke`. 16 KiB = 2048 u64 slots covers the current integer
/// demos while leaving more heap headroom; hosted default is 2 MiB.
pub const WASM_STACK_BYTES: usize = 16 * 1024;

/// Per-function compiler-time RAM budget. Functions estimated above this
/// budget use the template JIT path instead of the full compiler pipeline.
pub const COMPILER_RAM_BUDGET_BYTES: u32 = 500 * 1024;

/// Install the runtime configuration. Panics if called more than once
/// — `sf_nano_core::set_runtime_config` is write-once by design.
pub fn init() {
    sf_nano_core::set_runtime_config(sf_nano_core::RuntimeConfig {
        code_arena_bytes: CODE_ARENA_BYTES,
        wasm_memory_max_pages: WASM_MEMORY_MAX_PAGES,
        wasm_stack_bytes: WASM_STACK_BYTES,
        compiler_ram_budget_bytes: COMPILER_RAM_BUDGET_BYTES,
        parallel_compilation: false,
    })
    .expect("sf-nano-pico2 config already initialized");
}
