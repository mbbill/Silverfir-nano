//! macOS OS backend for the native JIT runtime.
//!
//! Executable memory is allocated RWX via `mmap` with `MAP_JIT`, which is
//! the only shape Apple allows for JIT regions on Apple-silicon Macs. W^X
//! is enforced per-thread with `pthread_jit_write_protect_np(0|1)` rather
//! than via `mprotect`. The instruction cache is flushed with
//! `sys_icache_invalidate`.
//!
//! Guarded memory uses the same `mmap PROT_NONE` + per-page `mprotect`
//! pattern as Linux.

use core::ptr;

use super::posix::{mmap, munmap, MAP_FAILED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};
#[cfg(sf_has_guard_pages)]
use super::posix::{mprotect, PROT_NONE};

const MAP_ANON: i32 = 0x1000;
const MAP_JIT: i32 = 0x0800;

unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: i32);
    fn sys_icache_invalidate(addr: *const u8, len: usize);
}

// ── Executable memory ───────────────────────────────────────────────────────

pub(crate) fn alloc_executable(capacity: usize) -> Result<*mut u8, &'static str> {
    let base = unsafe {
        mmap(
            ptr::null_mut(),
            capacity,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_PRIVATE | MAP_ANON | MAP_JIT,
            -1,
            0,
        )
    };
    if base == MAP_FAILED {
        return Err("mmap failed for native code buffer");
    }
    Ok(base)
}

pub(crate) fn free_executable(base: *mut u8, capacity: usize) {
    if base.is_null() || base == MAP_FAILED {
        return;
    }
    unsafe { munmap(base, capacity) };
}

#[inline]
pub(crate) unsafe fn begin_write_executable(_base: *mut u8, _capacity: usize) {
    unsafe { pthread_jit_write_protect_np(0) };
}

#[inline]
pub(crate) unsafe fn finish_write_executable(
    base: *mut u8,
    _capacity: usize,
    written_start: usize,
    written_len: usize,
) {
    unsafe {
        pthread_jit_write_protect_np(1);
        sys_icache_invalidate(base.add(written_start), written_len);
    }
}

// ── Guarded linear memory ───────────────────────────────────────────────────

#[cfg(sf_has_guard_pages)]
pub(crate) fn reserve_guarded(total: usize) -> Result<*mut u8, &'static str> {
    let base = unsafe {
        mmap(
            ptr::null_mut(),
            total,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0,
        )
    };
    if base == MAP_FAILED || base.is_null() {
        return Err("guard-page memory: mmap failed");
    }
    Ok(base)
}

#[cfg(sf_has_guard_pages)]
pub(crate) fn commit_guarded(
    base: *mut u8,
    offset: usize,
    len: usize,
) -> Result<(), &'static str> {
    if len == 0 {
        return Ok(());
    }
    let rc = unsafe { mprotect(base.add(offset), len, PROT_READ | PROT_WRITE) };
    if rc != 0 {
        return Err("guard-page memory: mprotect failed");
    }
    Ok(())
}

#[cfg(sf_has_guard_pages)]
pub(crate) fn release_guarded(base: *mut u8, total: usize) {
    unsafe { munmap(base, total) };
}
