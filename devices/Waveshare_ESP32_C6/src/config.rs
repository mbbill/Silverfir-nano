//! sf-nano-core runtime configuration for the ESP32-C6 board.

pub const CODE_ARENA_BYTES: usize = 32 * 1024;
pub const WASM_MEMORY_MAX_PAGES: u32 = 3;
pub const WASM_STACK_BYTES: usize = 16 * 1024;

/// This board's engine. Every budget is explicit: the bare-metal target
/// ships no defaults, so `Engine::new` names a missing one instead of
/// inventing it.
pub fn engine() -> sf_nano_core::Engine {
    sf_nano_core::Engine::new(
        sf_nano_core::Config::new()
            .code_arena_bytes(CODE_ARENA_BYTES)
            .wasm_memory_max_pages(WASM_MEMORY_MAX_PAGES)
            .wasm_stack_bytes(WASM_STACK_BYTES)
            .compiler_ram_budget_bytes(u32::MAX)
            .parallel_compilation(false),
    )
    .expect("waveshare-esp32-c6 engine configuration")
}
