//! Global allocator for the firmware.
//!
//! sf-nano-core pulls in the `alloc` crate (`Box`, `Vec`, `Rc` from
//! `tracked_alloc`). A `no_std` program using `alloc` must provide one
//! `#[global_allocator]` — this module registers a static-buffer
//! linked-list allocator from `embedded-alloc`.
//!
//! Sized to fit: module metadata, per-invoke Wasm operand stack
//! (`WASM_STACK_BYTES` bytes each), and up to
//! `WASM_MEMORY_MAX_PAGES × 64 KiB` of linear memory. The JIT code
//! arena lives in its own static region in `os_shim.rs`, not here.
//! Keep this value paired with the `config.rs` constants — see the
//! memory-budget table in `HACKING.md` §5 when tuning.

use core::mem::MaybeUninit;
use embedded_alloc::LlffHeap;

#[global_allocator]
static HEAP: LlffHeap = LlffHeap::empty();

const HEAP_SIZE: usize = 320 * 1024;
static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

/// Install the heap. Must be called exactly once, before any `Box`,
/// `Vec`, etc. are constructed anywhere in the program.
pub fn init() {
    // SAFETY: single-threaded startup path, called exactly once before
    // any allocation. `HEAP_MEM` outlives the allocator (both are 'static).
    unsafe {
        let ptr = core::ptr::addr_of_mut!(HEAP_MEM) as *mut u8;
        HEAP.init(ptr as usize, HEAP_SIZE);
    }
}
