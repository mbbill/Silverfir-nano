//! Executable code buffer for JIT emission.
//!
//! Arena-based bump allocator over mmap'd executable memory.
//! Platform-specific: macOS Apple Silicon (MAP_JIT) and Linux (mmap+mprotect).

use core::ptr;

// ---------------------------------------------------------------------------
// libc FFI declarations (no std needed)
// ---------------------------------------------------------------------------

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: i32);
    fn sys_icache_invalidate(addr: *const u8, len: usize);
}

// mmap constants
const PROT_READ: i32 = 0x01;
const PROT_WRITE: i32 = 0x02;
const PROT_EXEC: i32 = 0x04;
const MAP_PRIVATE: i32 = 0x02;
#[cfg(target_os = "macos")]
const MAP_ANON: i32 = 0x1000;
#[cfg(target_os = "macos")]
const MAP_JIT: i32 = 0x0800;
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

// ---------------------------------------------------------------------------
// CodeBuffer
// ---------------------------------------------------------------------------

/// A region of executable memory for JIT code emission.
pub struct CodeBuffer {
    /// Base address of the mmap'd region.
    base: *mut u8,
    /// Total capacity in bytes.
    capacity: usize,
    /// Current write offset (bump pointer).
    offset: usize,
}

impl CodeBuffer {
    /// Default arena size: 1MB (enough for thousands of JIT groups).
    const DEFAULT_CAPACITY: usize = 1024 * 1024;

    /// Allocate a new executable code buffer.
    pub fn new() -> Result<Self, &'static str> {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Allocate with a specific capacity.
    #[cfg(target_os = "macos")]
    pub fn with_capacity(capacity: usize) -> Result<Self, &'static str> {
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
            return Err("mmap failed for JIT code buffer");
        }
        Ok(Self { base, capacity, offset: 0 })
    }

    /// Allocate with a specific capacity (Linux).
    #[cfg(target_os = "linux")]
    pub fn with_capacity(capacity: usize) -> Result<Self, &'static str> {
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
            return Err("mmap failed for JIT code buffer");
        }
        Ok(Self { base, capacity, offset: 0 })
    }

    /// Begin a write session.
    /// On macOS, this disables JIT write protection for the current thread.
    #[cfg(target_os = "macos")]
    pub fn begin_write(&mut self) {
        unsafe { pthread_jit_write_protect_np(0); }
    }

    /// Begin a write session (Linux: no-op, pages are already writable).
    #[cfg(target_os = "linux")]
    pub fn begin_write(&mut self) {}

    /// End a write session. Makes the written code executable and flushes I-cache.
    /// `written_start` and `written_len` specify the range that was written.
    #[cfg(target_os = "macos")]
    pub fn finish_write(&mut self, written_start: usize, written_len: usize) {
        unsafe {
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(self.base.add(written_start), written_len);
        }
    }

    /// End a write session (Linux: mprotect to RX and clear cache).
    #[cfg(target_os = "linux")]
    pub fn finish_write(&mut self, written_start: usize, written_len: usize) {
        extern "C" {
            fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe {
            // Note: mprotect requires page-aligned address. For simplicity,
            // protect the entire buffer. A production impl would track dirty pages.
            mprotect(self.base, self.capacity, PROT_READ | PROT_EXEC);
            let start = self.base.add(written_start);
            let end = start.add(written_len);
            __clear_cache(start, end);
        }
    }

    /// Emit a single u32 instruction. Returns the offset where it was written.
    /// Must be called between `begin_write()` and `finish_write()`.
    pub fn emit(&mut self, inst: u32) -> usize {
        let offset = self.offset;
        assert!(offset + 4 <= self.capacity, "JIT code buffer overflow");
        unsafe {
            let ptr = self.base.add(offset) as *mut u32;
            ptr.write(inst);
        }
        self.offset += 4;
        offset
    }

    /// Emit a slice of u32 instructions. Returns the start offset.
    pub fn emit_slice(&mut self, insts: &[u32]) -> usize {
        let start = self.offset;
        for &inst in insts {
            self.emit(inst);
        }
        start
    }

    /// Get a function pointer to code at the given offset.
    ///
    /// # Safety
    /// The code at `offset` must be valid machine code that conforms to the
    /// calling convention of function type `F`.
    pub unsafe fn fn_ptr<F>(&self, offset: usize) -> F
    where
        F: Copy,
    {
        let ptr = self.base.add(offset);
        core::mem::transmute_copy(&ptr)
    }

    /// Get the base pointer of the buffer.
    pub fn base_ptr(&self) -> *const u8 {
        self.base
    }

    /// Current write offset (bytes used so far).
    pub fn len(&self) -> usize {
        self.offset
    }

    /// Remaining capacity in bytes.
    pub fn remaining(&self) -> usize {
        self.capacity - self.offset
    }

    /// Reset the buffer (reuse all memory).
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        if !self.base.is_null() && self.base != MAP_FAILED {
            unsafe { munmap(self.base, self.capacity); }
        }
    }
}

// SAFETY: The code buffer manages its own memory and doesn't alias.
unsafe impl Send for CodeBuffer {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::arm64_enc;
    use super::super::reg::Reg;

    #[test]
    fn test_mmap_allocation() {
        let buf = CodeBuffer::new().expect("mmap failed");
        assert!(!buf.base_ptr().is_null());
        assert_eq!(buf.len(), 0);
        assert!(buf.remaining() > 0);
    }

    #[test]
    fn test_emit_ret() {
        // Emit a single `ret` instruction and call it.
        // If it returns without crashing, mmap+execution works.
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let offset = buf.emit(arm64_enc::ret());
        buf.finish_write(offset, 4);

        let f: extern "C" fn() = unsafe { buf.fn_ptr(offset) };
        f(); // Should return without crashing
    }

    #[test]
    fn test_emit_mov_ret() {
        // Emit `mov x0, #42; ret` and verify return value.
        // T3 = x0 in our Reg enum, which is the return value register.
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let offset = buf.emit(arm64_enc::movz_64(Reg::T3, 42, 0));
        buf.emit(arm64_enc::ret());
        buf.finish_write(offset, 8);

        let f: extern "C" fn() -> u64 = unsafe { buf.fn_ptr(offset) };
        assert_eq!(f(), 42);
    }

    #[test]
    fn test_emit_add_ret() {
        // Emit `mov x0, #10; add x0, x0, #32; ret` → returns 42
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let offset = buf.emit(arm64_enc::movz_64(Reg::T3, 10, 0));
        buf.emit(arm64_enc::add_imm_64(Reg::T3, Reg::T3, 32));
        buf.emit(arm64_enc::ret());
        buf.finish_write(offset, 12);

        let f: extern "C" fn() -> u64 = unsafe { buf.fn_ptr(offset) };
        assert_eq!(f(), 42);
    }

    #[test]
    fn test_multiple_emissions() {
        // Emit multiple functions in sequence
        let mut buf = CodeBuffer::new().expect("mmap failed");

        // Function 1: returns 10
        buf.begin_write();
        let off1 = buf.emit(arm64_enc::movz_64(Reg::T3, 10, 0));
        buf.emit(arm64_enc::ret());
        buf.finish_write(off1, 8);

        // Function 2: returns 20
        buf.begin_write();
        let off2 = buf.emit(arm64_enc::movz_64(Reg::T3, 20, 0));
        buf.emit(arm64_enc::ret());
        buf.finish_write(off2, 8);

        let f1: extern "C" fn() -> u64 = unsafe { buf.fn_ptr(off1) };
        let f2: extern "C" fn() -> u64 = unsafe { buf.fn_ptr(off2) };
        assert_eq!(f1(), 10);
        assert_eq!(f2(), 20);
    }
}
