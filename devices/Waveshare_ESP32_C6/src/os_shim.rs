//! Bare-metal executable-memory shims for sf-nano-core.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::config::CODE_ARENA_BYTES;

#[repr(align(16))]
struct CodeArena {
    #[allow(
        dead_code,
        reason = "reserves the arena's aligned storage; reached only by raw-pointer cast in sf_os_alloc_executable"
    )]
    bytes: [u8; CODE_ARENA_BYTES],
}

// ESP32-C6 has a 64-KiB `dram2_seg` that is only free after the
// second-stage bootloader exits. The JIT arena is claimed from `main`, so it
// can live there and leave the normal RAM region to the Rust heap.
#[unsafe(link_section = ".dram2_uninit")]
static mut CODE_ARENA: CodeArena = CodeArena {
    bytes: [0; CODE_ARENA_BYTES],
};

static CODE_ARENA_TAKEN: AtomicBool = AtomicBool::new(false);

#[unsafe(no_mangle)]
pub extern "C" fn sf_os_alloc_executable(capacity: usize) -> *mut u8 {
    if capacity > CODE_ARENA_BYTES {
        return core::ptr::null_mut();
    }
    if CODE_ARENA_TAKEN
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return core::ptr::null_mut();
    }

    let ptr = core::ptr::addr_of_mut!(CODE_ARENA) as *mut u8;
    debug_assert!(ptr as usize % 16 == 0);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn sf_os_free_executable(_base: *mut u8, _capacity: usize) {
    CODE_ARENA_TAKEN.store(false, Ordering::Release);
}

#[unsafe(no_mangle)]
pub extern "C" fn sf_os_begin_write_executable(_base: *mut u8, _capacity: usize) {}

#[unsafe(no_mangle)]
pub extern "C" fn sf_os_finish_write_executable(
    _base: *mut u8,
    _capacity: usize,
    _written_start: usize,
    _written_len: usize,
) {
    crate::arch::dsb();
    crate::arch::isb();
}
