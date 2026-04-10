//! Executable memory ownership for the native backend.
//!
//! This owns one OS-backed writable/executable region. The current machine
//! backend uses it as a module-wide arena for finalized native code.
//!
//! All OS coupling — page allocation, W^X toggling, instruction-cache
//! invalidation — is delegated to [`crate::vm::runtime::os`]. This module
//! holds only the per-buffer state (`base`, `capacity`, `offset`) and the
//! offset-bumping emit helpers.

use core::ptr;

use super::os;

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

    pub fn with_capacity(capacity: usize) -> Result<Self, &'static str> {
        let base = os::alloc_executable(capacity)?;
        let buffer = Self {
            base,
            capacity,
            offset: 0,
        };
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_state(base as usize, capacity, 0);
        Ok(buffer)
    }

    #[inline]
    pub fn begin_write(&mut self) {
        unsafe { os::begin_write_executable(self.base, self.capacity) };
    }

    #[inline]
    pub fn finish_write(&mut self, written_start: usize, written_len: usize) {
        unsafe {
            os::finish_write_executable(self.base, self.capacity, written_start, written_len);
        }
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_state(self.base as usize, self.capacity, self.offset);
    }

    #[inline]
    pub fn emit_u32(&mut self, inst: u32) -> usize {
        let offset = self.offset;
        assert!(offset + 4 <= self.capacity, "native code buffer overflow");
        unsafe {
            (self.base.add(offset) as *mut u32).write(inst);
        }
        self.offset += 4;
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_state(self.base as usize, self.capacity, self.offset);
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
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_state(self.base as usize, self.capacity, self.offset);
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
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_state(self.base as usize, self.capacity, self.offset);
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
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_state(self.base as usize, self.capacity, 0);
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "memtrace")]
        sf_nano_memtrace::record_exec_buffer_drop(self.base as usize);
        os::free_executable(self.base, self.capacity);
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
