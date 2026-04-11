//! Linux OS backend for the native JIT runtime.
//!
//! Executable memory is mapped RW and flipped to RX on each finalize via
//! `mprotect`. The instruction cache is flushed per-architecture: aarch64
//! and x86_64 use the compiler-builtins `__clear_cache`; ARM32 uses the
//! Linux `cacheflush` syscall directly because `rust-lld` does not
//! resolve the builtin on musl.
//!
//! Guarded memory is a `PROT_NONE` reservation plus per-page `mprotect` to
//! `PROT_READ|PROT_WRITE` on `commit_guarded`.

use core::ptr;

#[cfg(sf_has_guard_pages)]
use super::posix::PROT_NONE;
use super::posix::{mmap, mprotect, munmap, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE};

const PROT_EXEC: i32 = 0x04;
const MAP_ANONYMOUS: i32 = 0x20;

// ── Instruction cache flush ─────────────────────────────────────────────────

#[cfg(sf_arch_arm64)]
unsafe extern "C" {
    fn __clear_cache(start: *mut u8, end: *mut u8);
}

#[cfg(sf_arch_arm64)]
#[inline]
unsafe fn clear_instruction_cache(start: *mut u8, end: *mut u8) {
    unsafe { __clear_cache(start, end) };
}

#[cfg(sf_arch_armv7a)]
#[inline]
unsafe fn clear_instruction_cache(start: *mut u8, end: *mut u8) {
    // ARM Linux cacheflush syscall (__ARM_NR_cacheflush = 0x0f0002).
    // The symbol `__clear_cache` is not resolved by rust-lld on ARM32 musl,
    // so we issue the syscall directly rather than calling the builtin.
    unsafe {
        core::arch::asm!(
            "mov r0, {start}",
            "mov r1, {end}",
            "mov r2, #0",
            "mov r7, #0xf0000",
            "add r7, r7, #0x2",
            "svc #0",
            start = in(reg) start,
            end = in(reg) end,
            out("r0") _,
            out("r1") _,
            out("r2") _,
            out("r7") _,
        );
    }
}

#[cfg(not(any(sf_arch_arm64, sf_arch_armv7a)))]
unsafe extern "C" {
    fn __clear_cache(start: *mut u8, end: *mut u8);
}

#[cfg(not(any(sf_arch_arm64, sf_arch_armv7a)))]
#[inline]
unsafe fn clear_instruction_cache(start: *mut u8, end: *mut u8) {
    unsafe { __clear_cache(start, end) };
}

// ── Executable memory ───────────────────────────────────────────────────────

pub(crate) fn alloc_executable(capacity: usize) -> Result<*mut u8, &'static str> {
    let base = unsafe {
        mmap(
            ptr::null_mut(),
            capacity,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
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
pub(crate) unsafe fn begin_write_executable(base: *mut u8, capacity: usize) {
    unsafe {
        let rc = mprotect(base, capacity, PROT_READ | PROT_WRITE);
        assert_eq!(rc, 0, "mprotect RW failed for native code buffer");
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
        let rc = mprotect(base, capacity, PROT_READ | PROT_EXEC);
        assert_eq!(rc, 0, "mprotect RX failed for native code buffer");
        let start = base.add(written_start);
        let end = start.add(written_len);
        clear_instruction_cache(start, end);
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
            MAP_PRIVATE | MAP_ANONYMOUS,
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
pub(crate) fn commit_guarded(base: *mut u8, offset: usize, len: usize) -> Result<(), &'static str> {
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
