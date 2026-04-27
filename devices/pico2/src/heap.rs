//! Global allocator for the firmware.
//!
//! sf-nano-core pulls in the `alloc` crate (`Box`, `Vec`, `Rc` from
//! `tracked_alloc`). A `no_std` program using `alloc` must provide one
//! `#[global_allocator]` — this module registers a static-buffer
//! linked-list allocator from `embedded-alloc`.
//!
//! Sized to fit: module metadata, materialized Wasm linear memory, and
//! the per-invoke Wasm operand stack (`WASM_STACK_BYTES` bytes each).
//! JIT builds compile before materializing local linear memory to reduce
//! peak heap pressure. The JIT code arena lives in its own static region
//! in `os_shim.rs`, not here.
//! Keep this value paired with the `config.rs` constants — see
//! `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md` when tuning.

use core::alloc::{GlobalAlloc, Layout};
use core::mem::MaybeUninit;
use embedded_alloc::LlffHeap;

/// Global heap wrapper that reports allocator-boundary failures before
/// `alloc` falls back to its generic `handle_alloc_error` panic.
struct DiagnosticHeap(LlffHeap);

impl DiagnosticHeap {
    fn log_failure(&self, op: &str, size: usize, align: usize) {
        defmt::error!(
            "rust heap allocation failed: op={=str} size={} align={} heap_used={} heap_free={} heap_total={}",
            op,
            size,
            align,
            self.0.used(),
            self.0.free(),
            HEAP_SIZE
        );
    }
}

unsafe impl GlobalAlloc for DiagnosticHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.0.alloc(layout) };
        if ptr.is_null() {
            self.log_failure("alloc", layout.size(), layout.align());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.0.alloc_zeroed(layout) };
        if ptr.is_null() {
            self.log_failure("alloc_zeroed", layout.size(), layout.align());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { self.0.realloc(ptr, layout, new_size) };
        if ptr.is_null() {
            self.log_failure("realloc", new_size, layout.align());
        }
        ptr
    }
}

#[global_allocator]
static HEAP: DiagnosticHeap = DiagnosticHeap(LlffHeap::empty());

const HEAP_SIZE: usize = 448 * 1024;
static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

/// Install the heap. Must be called exactly once, before any `Box`,
/// `Vec`, etc. are constructed anywhere in the program.
pub fn init() {
    // SAFETY: single-threaded startup path, called exactly once before
    // any allocation. `HEAP_MEM` outlives the allocator (both are 'static).
    unsafe {
        let ptr = core::ptr::addr_of_mut!(HEAP_MEM) as *mut u8;
        HEAP.0.init(ptr as usize, HEAP_SIZE);
    }
}
