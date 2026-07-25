//! Pico2-style fixed global heap for `alloc`.
//!
//! The Wasm/JIT path uses this heap for module metadata, materialized linear
//! memory, and the per-instance operand stack. JIT compilation runs before local linear
//! memories are materialized so the framebuffer-sized memory is not live
//! during peak compile-time allocation pressure.

use core::{
    alloc::{GlobalAlloc, Layout},
    mem::MaybeUninit,
};

use embedded_alloc::LlffHeap;
use esp_println::println;

struct DiagnosticHeap(LlffHeap);

impl DiagnosticHeap {
    fn log_failure(&self, op: &str, size: usize, align: usize) {
        println!(
            "rust heap allocation failed: op={} size={} align={} heap_used={} heap_free={} heap_total={}",
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

pub const HEAP_SIZE: usize = 384 * 1024;
static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

pub fn init() {
    unsafe {
        let ptr = core::ptr::addr_of_mut!(HEAP_MEM) as *mut u8;
        HEAP.0.init(ptr as usize, HEAP_SIZE);
    }
}
