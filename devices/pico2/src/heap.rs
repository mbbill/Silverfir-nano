//! Global allocator for the firmware.
//!
//! sf-nano-core pulls in the `alloc` crate (`Box`, `Vec`, `Rc` from
//! `tracked_alloc`). A `no_std` program using `alloc` must provide one
//! `#[global_allocator]` — this module registers a 64 KiB static-buffer
//! linked-list allocator from `embedded-alloc`.
//!
//! 64 KiB is conservative: the Pico 2 W has 520 KiB of SRAM, so this
//! leaves room for the stack, BSS, RTT buffer, and (eventually) the
//! JIT code arena sitting in its own static `[u8; N]` region.
//! Re-tune when the real workload needs it.

use core::mem::MaybeUninit;
use embedded_alloc::LlffHeap;

#[global_allocator]
static HEAP: LlffHeap = LlffHeap::empty();

const HEAP_SIZE: usize = 64 * 1024;
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
