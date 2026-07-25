//! POSIX primitives shared between the `linux` and `macos` OS backends.
//!
//! Only the two types of constants and extern declarations that both
//! backends need live here. Anything OS-specific (e.g. `MAP_JIT`,
//! `pthread_jit_write_protect_np`, `__clear_cache`) stays in the
//! per-OS module.

unsafe extern "C" {
    pub(super) fn mmap(
        addr: *mut u8,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut u8;
    pub(super) fn munmap(addr: *mut u8, len: usize) -> i32;
}

// `mprotect` is used by the Linux executable-memory code unconditionally and
// by the guard-page implementation on both Linux and macOS. It is never used
// on macOS when guard pages are compiled out, so the extern is gated.
#[cfg(any(sf_os_linux, sf_has_guard_pages))]
unsafe extern "C" {
    pub(super) fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
}

/// `PROT_NONE` is only referenced by the guard-page reservation path.
#[cfg(sf_has_guard_pages)]
pub(super) const PROT_NONE: i32 = 0x00;
pub(super) const PROT_READ: i32 = 0x01;
pub(super) const PROT_WRITE: i32 = 0x02;
#[cfg(sf_os_macos)]
pub(super) const PROT_EXEC: i32 = 0x04;
pub(super) const MAP_PRIVATE: i32 = 0x02;
pub(super) const MAP_FAILED: *mut u8 = !0usize as *mut u8;
