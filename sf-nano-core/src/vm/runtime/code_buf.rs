//! Executable memory ownership for the native backend.
//!
//! This owns one mmap-backed writable/executable region. The current machine
//! backend uses it as a module-wide arena for finalized native code.

use core::ptr;

#[cfg(sf_has_posix)]
unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

#[cfg(sf_os_linux)]
unsafe extern "C" {
    fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
}

/// Flush the instruction cache for the given range.
///
/// On AArch64, `__clear_cache` is provided by compiler-builtins. On ARM32
/// musl with rust-lld the symbol is absent, so we call the kernel's
/// `cacheflush` syscall directly.
#[cfg(all(sf_os_linux, sf_arch_arm64))]
unsafe extern "C" {
    fn __clear_cache(start: *mut u8, end: *mut u8);
}

#[cfg(all(sf_os_linux, sf_arch_arm64))]
#[inline]
unsafe fn clear_instruction_cache(start: *mut u8, end: *mut u8) {
    unsafe { __clear_cache(start, end) };
}

#[cfg(all(sf_os_linux, sf_arch_armv7a))]
#[inline]
unsafe fn clear_instruction_cache(start: *mut u8, end: *mut u8) {
    // ARM Linux cacheflush syscall (__ARM_NR_cacheflush = 0x0f0002)
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

#[cfg(all(sf_os_linux, not(any(sf_arch_arm64, sf_arch_armv7a))))]
unsafe extern "C" {
    fn __clear_cache(start: *mut u8, end: *mut u8);
}

#[cfg(all(sf_os_linux, not(any(sf_arch_arm64, sf_arch_armv7a))))]
#[inline]
unsafe fn clear_instruction_cache(start: *mut u8, end: *mut u8) {
    unsafe { __clear_cache(start, end) };
}

#[cfg(sf_os_macos)]
unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: i32);
    fn sys_icache_invalidate(addr: *const u8, len: usize);
}

#[cfg(sf_os_windows)]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
    fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(addr: *mut u8, size: usize, new_protect: u32, old_protect: *mut u32) -> i32;
    fn FlushInstructionCache(process: *mut u8, base: *const u8, size: usize) -> i32;
    fn GetCurrentProcess() -> *mut u8;
}
#[cfg(sf_os_windows)]
const MEM_COMMIT: u32 = 0x1000;
#[cfg(sf_os_windows)]
const MEM_RESERVE: u32 = 0x2000;
#[cfg(sf_os_windows)]
const MEM_RELEASE: u32 = 0x8000;
#[cfg(sf_os_windows)]
const PAGE_READWRITE: u32 = 0x04;
#[cfg(sf_os_windows)]
const PAGE_EXECUTE_READ: u32 = 0x20;

#[cfg(sf_has_posix)]
const PROT_READ: i32 = 0x01;
#[cfg(sf_has_posix)]
const PROT_WRITE: i32 = 0x02;
#[cfg(sf_has_posix)]
const PROT_EXEC: i32 = 0x04;
#[cfg(sf_has_posix)]
const MAP_PRIVATE: i32 = 0x02;
#[cfg(sf_os_macos)]
const MAP_ANON: i32 = 0x1000;
#[cfg(sf_os_macos)]
const MAP_JIT: i32 = 0x0800;
#[cfg(sf_os_linux)]
const MAP_ANONYMOUS: i32 = 0x20;
#[cfg(sf_has_posix)]
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

pub struct CodeBuffer {
    base: *mut u8,
    capacity: usize,
    offset: usize,
}

impl core::fmt::Debug for CodeBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CodeBuffer")
            .field("base", &self.base)
            .field("capacity", &self.capacity)
            .field("offset", &self.offset)
            .finish()
    }
}

impl CodeBuffer {
    #[cfg(target_pointer_width = "32")]
    const DEFAULT_CAPACITY: usize = 12 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "32"))]
    const DEFAULT_CAPACITY: usize = 16 * 1024 * 1024;

    #[inline]
    pub fn new() -> Result<Self, &'static str> {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    #[cfg(sf_os_macos)]
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
            return Err("mmap failed for native code buffer");
        }
        Ok(Self {
            base,
            capacity,
            offset: 0,
        })
    }

    #[cfg(sf_os_linux)]
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
            return Err("mmap failed for native code buffer");
        }
        Ok(Self {
            base,
            capacity,
            offset: 0,
        })
    }

    #[cfg(sf_os_windows)]
    pub fn with_capacity(capacity: usize) -> Result<Self, &'static str> {
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
        Ok(Self {
            base,
            capacity,
            offset: 0,
        })
    }

    #[cfg(sf_os_macos)]
    #[inline]
    pub fn begin_write(&mut self) {
        unsafe { pthread_jit_write_protect_np(0) };
    }

    #[cfg(sf_os_linux)]
    #[inline]
    pub fn begin_write(&mut self) {
        unsafe {
            let rc = mprotect(self.base, self.capacity, PROT_READ | PROT_WRITE);
            assert_eq!(rc, 0, "mprotect RW failed for native code buffer");
        }
    }

    #[cfg(sf_os_windows)]
    #[inline]
    pub fn begin_write(&mut self) {
        unsafe {
            let mut old: u32 = 0;
            let rc = VirtualProtect(self.base, self.capacity, PAGE_READWRITE, &mut old);
            assert_ne!(rc, 0, "VirtualProtect RW failed");
        }
    }

    #[cfg(sf_os_macos)]
    #[inline]
    pub fn finish_write(&mut self, written_start: usize, written_len: usize) {
        unsafe {
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(self.base.add(written_start), written_len);
        }
    }

    #[cfg(sf_os_linux)]
    #[inline]
    pub fn finish_write(&mut self, written_start: usize, written_len: usize) {
        unsafe {
            let rc = mprotect(self.base, self.capacity, PROT_READ | PROT_EXEC);
            assert_eq!(rc, 0, "mprotect RX failed for native code buffer");
            let start = self.base.add(written_start);
            let end = start.add(written_len);
            clear_instruction_cache(start, end);
        }
    }

    #[cfg(sf_os_windows)]
    #[inline]
    pub fn finish_write(&mut self, written_start: usize, written_len: usize) {
        unsafe {
            let mut old: u32 = 0;
            let rc = VirtualProtect(self.base, self.capacity, PAGE_EXECUTE_READ, &mut old);
            assert_ne!(rc, 0, "VirtualProtect RX failed");
            FlushInstructionCache(
                GetCurrentProcess(),
                self.base.add(written_start),
                written_len,
            );
        }
    }

    #[inline]
    pub fn emit_u32(&mut self, inst: u32) -> usize {
        let offset = self.offset;
        assert!(offset + 4 <= self.capacity, "native code buffer overflow");
        unsafe {
            (self.base.add(offset) as *mut u32).write(inst);
        }
        self.offset += 4;
        offset
    }

    #[inline]
    pub fn emit_u64(&mut self, value: u64) -> usize {
        let offset = self.offset;
        assert!(offset + 8 <= self.capacity, "native code buffer overflow");
        unsafe {
            (self.base.add(offset) as *mut u64).write(value);
        }
        self.offset += 8;
        offset
    }

    #[inline]
    pub fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
        let offset = self.offset;
        assert!(
            offset + bytes.len() <= self.capacity,
            "native code buffer overflow"
        );
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(offset), bytes.len());
        }
        self.offset += bytes.len();
        offset
    }

    #[inline]
    pub fn patch_u32(&mut self, offset: usize, inst: u32) {
        assert!(offset + 4 <= self.offset, "patch beyond written region");
        unsafe {
            (self.base.add(offset) as *mut u32).write(inst);
        }
    }

    #[inline]
    pub fn patch_u64(&mut self, offset: usize, value: u64) {
        assert!(offset + 8 <= self.offset, "patch beyond written region");
        unsafe {
            (self.base.add(offset) as *mut u64).write(value);
        }
    }

    #[inline]
    pub unsafe fn fn_ptr<F>(&self, offset: usize) -> F
    where
        F: Copy,
    {
        let ptr = unsafe { self.base.add(offset) };
        core::mem::transmute_copy(&ptr)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.base.cast::<u8>()
    }

    #[inline]
    pub unsafe fn ptr(&self, offset: usize) -> *const u8 {
        unsafe { self.base.add(offset) }.cast::<u8>()
    }

    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        if self.base.is_null() {
            return;
        }
        #[cfg(sf_has_posix)]
        {
            if self.base != MAP_FAILED {
                unsafe {
                    munmap(self.base, self.capacity);
                }
            }
        }
        #[cfg(sf_os_windows)]
        {
            unsafe {
                VirtualFree(self.base, 0, MEM_RELEASE);
            }
        }
    }
}

unsafe impl Send for CodeBuffer {}

#[cfg(test)]
mod tests {
    use super::CodeBuffer;

    #[test]
    fn allocates_and_writes_bytes() {
        let mut buf = CodeBuffer::with_capacity(64).expect("mmap failed");
        buf.begin_write();
        let start = buf.emit_bytes(&[1, 2, 3, 4]);
        buf.finish_write(start, 4);
        assert_eq!(start, 0);
        assert_eq!(buf.len(), 4);
    }
}
