//! Windows OS backend for the native JIT runtime.
//!
//! Executable memory is committed RW via `VirtualAlloc` and flipped to RX
//! on finalize via `VirtualProtect`, with `FlushInstructionCache` for
//! icache invalidation.
//!
//! Guarded memory is not yet implemented. The reserve/commit/release
//! primitives are only compiled when `sf_has_guard_pages` is set, which
//! `build.rs` currently gates off for Windows. The stub shapes below are
//! wired up so that a future implementation can plug in without touching
//! callers.

use core::ptr;

unsafe extern "system" {
    fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
    fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(addr: *mut u8, size: usize, new_protect: u32, old_protect: *mut u32) -> i32;
    fn FlushInstructionCache(process: *mut u8, base: *const u8, size: usize) -> i32;
    fn GetCurrentProcess() -> *mut u8;
}

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;

// ── Executable memory ───────────────────────────────────────────────────────

pub(crate) fn alloc_executable(capacity: usize) -> Result<*mut u8, &'static str> {
    let base = unsafe {
        VirtualAlloc(
            ptr::null_mut(),
            capacity,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    if base.is_null() {
        return Err("VirtualAlloc failed for native code buffer");
    }
    Ok(base)
}

pub(crate) fn free_executable(base: *mut u8, _capacity: usize) {
    if base.is_null() {
        return;
    }
    unsafe { VirtualFree(base, 0, MEM_RELEASE) };
}

#[inline]
pub(crate) unsafe fn begin_write_executable(base: *mut u8, capacity: usize) {
    unsafe {
        let mut old: u32 = 0;
        let rc = VirtualProtect(base, capacity, PAGE_READWRITE, &mut old);
        assert_ne!(rc, 0, "VirtualProtect RW failed");
    }
}

#[inline]
pub(crate) unsafe fn finish_write_executable(
    base: *mut u8,
    capacity: usize,
    written_start: usize,
    written_len: usize,
) {
    unsafe {
        let mut old: u32 = 0;
        let rc = VirtualProtect(base, capacity, PAGE_EXECUTE_READ, &mut old);
        assert_ne!(rc, 0, "VirtualProtect RX failed");
        FlushInstructionCache(GetCurrentProcess(), base.add(written_start), written_len);
    }
}

// ── Guarded linear memory ───────────────────────────────────────────────────
//
// Windows can in principle support guarded memory via
// `VirtualAlloc(MEM_RESERVE|PAGE_NOACCESS)` for the reservation,
// `VirtualAlloc(MEM_COMMIT|PAGE_READWRITE)` for commit, and VEH for the
// trap handler. Today `build.rs` never emits `sf_has_guard_pages` for
// Windows, so the functions below are compiled out entirely.
