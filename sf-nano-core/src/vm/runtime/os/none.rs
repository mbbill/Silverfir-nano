//! Bare-metal OS backend (`target_os = "none"`).
//!
//! This backend has no hosted OS to call — the runtime dispatches every
//! page-management primitive to symbols that the *embedder* must provide
//! in its own crate (typically with `#[unsafe(no_mangle)] pub extern "C"
//! fn sf_os_...`). The linker resolves them at final link time.
//!
//! The contract for an embedder targeting `target_os = "none"` is:
//!
//! ```ignore
//! #[unsafe(no_mangle)]
//! pub extern "C" fn sf_os_alloc_executable(capacity: usize) -> *mut u8 { /* ... */ }
//! #[unsafe(no_mangle)]
//! pub extern "C" fn sf_os_free_executable(base: *mut u8, capacity: usize) { /* ... */ }
//! #[unsafe(no_mangle)]
//! pub extern "C" fn sf_os_begin_write_executable(base: *mut u8, capacity: usize) { /* ... */ }
//! #[unsafe(no_mangle)]
//! pub extern "C" fn sf_os_finish_write_executable(
//!     base: *mut u8, capacity: usize, written_start: usize, written_len: usize,
//! ) { /* ... */ }
//! ```
//!
//! Guarded linear memory is not wired up on bare-metal: `build.rs` does
//! not emit `sf_has_guard_pages` for `target_os = "none"`, so the
//! reserve/commit/release primitives are compiled out entirely. A future
//! bare-metal target that wants guarded memory can extend this module
//! with three more extern symbols and extend `build.rs` to enable the cfg.

unsafe extern "C" {
    fn sf_os_alloc_executable(capacity: usize) -> *mut u8;
    fn sf_os_free_executable(base: *mut u8, capacity: usize);
    fn sf_os_begin_write_executable(base: *mut u8, capacity: usize);
    fn sf_os_finish_write_executable(
        base: *mut u8,
        capacity: usize,
        written_start: usize,
        written_len: usize,
    );
}

pub(crate) fn alloc_executable(capacity: usize) -> Result<*mut u8, &'static str> {
    let base = unsafe { sf_os_alloc_executable(capacity) };
    if base.is_null() {
        return Err("sf_os_alloc_executable returned null");
    }
    Ok(base)
}

pub(crate) fn free_executable(base: *mut u8, capacity: usize) {
    if base.is_null() {
        return;
    }
    unsafe { sf_os_free_executable(base, capacity) };
}

#[inline]
pub(crate) unsafe fn begin_write_executable(base: *mut u8, capacity: usize) {
    unsafe { sf_os_begin_write_executable(base, capacity) };
}

#[inline]
pub(crate) unsafe fn finish_write_executable(
    base: *mut u8,
    capacity: usize,
    written_start: usize,
    written_len: usize,
) {
    unsafe { sf_os_finish_write_executable(base, capacity, written_start, written_len) };
}
