//! Executable memory ownership for the native backend.
//!
//! This owns one OS-backed writable/executable region. The current machine
//! backend uses it as a module-wide arena for finalized native code.
//!
//! All OS coupling — page allocation, W^X toggling, instruction-cache
//! invalidation — is delegated to [`crate::vm::jit::runtime::os`]. This module
//! holds only the per-buffer state (`base`, `capacity`, `offset`) and the
//! offset-bumping emit helpers.

use crate::config::Config;
use core::ptr;

use super::os;
#[cfg(sf_has_guard_pages)]
use super::trap_signal;
#[cfg(feature = "memprof")]
use tracked_alloc::{
    AllocationDescriptor, AllocationHandle, AllocationState, RUNTIME_MEMORY_OWNER,
};

pub(crate) struct CodeBuffer {
    base: *mut u8,
    capacity: usize,
    offset: usize,
    #[cfg(feature = "memprof")]
    trace: AllocationHandle,
}

impl CodeBuffer {
    #[cfg(sf_has_guard_pages)]
    #[inline]
    fn unregister_trap_ranges(&self) {
        if self.base.is_null() || self.capacity == 0 {
            return;
        }
        let start = self.base as usize;
        let end = start.saturating_add(self.capacity);
        trap_signal::unregister_jit_ranges_in(start, end);
    }
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
    #[inline]
    pub(crate) fn new(config: &Config) -> Result<Self, &'static str> {
        Self::with_capacity(config.get_code_arena_bytes())
    }

    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, &'static str> {
        let base = os::alloc_executable(capacity)?;
        let buffer = Self {
            base,
            capacity,
            offset: 0,
            #[cfg(feature = "memprof")]
            trace: AllocationHandle::new(
                AllocationDescriptor::new(RUNTIME_MEMORY_OWNER, "CodeBuffer"),
                AllocationState::new(capacity)
                    .with_len(0)
                    .with_capacity(capacity)
                    .with_ptr(base as usize),
            ),
        };
        Ok(buffer)
    }

    #[cfg(feature = "memprof")]
    #[inline]
    fn trace_state(&self) -> AllocationState {
        AllocationState::new(self.capacity)
            .with_len(self.offset)
            .with_capacity(self.capacity)
            .with_ptr(self.base as usize)
    }

    #[inline]
    pub(crate) fn begin_write(&mut self) {
        unsafe { os::begin_write_executable(self.base, self.capacity) };
    }

    #[inline]
    pub(crate) fn finish_write(&mut self, written_start: usize, written_len: usize) {
        unsafe {
            os::finish_write_executable(self.base, self.capacity, written_start, written_len);
        }
        #[cfg(feature = "memprof")]
        self.trace.update(self.trace_state());
    }

    #[cfg(any(
        sf_backend_x64,
        sf_arm32_isa_thumb,
        sf_backend_riscv32,
        sf_backend_riscv64
    ))]
    #[inline]
    pub(crate) fn emit_u8(&mut self, byte: u8) -> usize {
        let offset = self.offset;
        assert!(offset < self.capacity, "native code buffer overflow");
        unsafe {
            self.base.add(offset).write(byte);
        }
        self.offset += 1;
        offset
    }

    #[inline]
    pub(crate) fn emit_u32(&mut self, inst: u32) -> usize {
        let offset = self.offset;
        assert!(offset + 4 <= self.capacity, "native code buffer overflow");
        unsafe {
            (self.base.add(offset) as *mut u32).write(inst);
        }
        self.offset += 4;
        offset
    }

    #[cfg(any(sf_backend_arm64, sf_backend_x64, sf_backend_riscv64))]
    #[inline]
    pub(crate) fn emit_u64(&mut self, value: u64) -> usize {
        let offset = self.offset;
        assert!(offset + 8 <= self.capacity, "native code buffer overflow");
        unsafe {
            (self.base.add(offset) as *mut u64).write(value);
        }
        self.offset += 8;
        offset
    }

    #[inline]
    pub(crate) fn emit_bytes(&mut self, bytes: &[u8]) -> usize {
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
    pub(crate) fn patch_u32(&mut self, offset: usize, inst: u32) {
        assert!(offset + 4 <= self.offset, "patch beyond written region");
        unsafe {
            (self.base.add(offset) as *mut u32).write(inst);
        }
    }

    #[cfg(any(sf_backend_arm64, sf_backend_x64, sf_backend_riscv64))]
    #[inline]
    pub(crate) fn patch_u64(&mut self, offset: usize, value: u64) {
        assert!(offset + 8 <= self.offset, "patch beyond written region");
        unsafe {
            (self.base.add(offset) as *mut u64).write(value);
        }
    }

    #[inline]
    pub(crate) fn byte(&self, offset: usize) -> u8 {
        assert!(offset < self.offset, "read beyond written region");
        unsafe { *self.base.add(offset) }
    }

    #[inline]
    pub(crate) unsafe fn fn_ptr<F>(&self, offset: usize) -> F
    where
        F: Copy,
    {
        let ptr = unsafe { self.base.add(offset) };
        core::mem::transmute_copy(&ptr)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.offset
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.base.cast::<u8>()
    }

    #[cfg(sf_has_guard_pages)]
    #[inline]
    pub(crate) unsafe fn ptr(&self, offset: usize) -> *const u8 {
        unsafe { self.base.add(offset) }.cast::<u8>()
    }

    #[inline]
    pub(crate) fn reset(&mut self) {
        #[cfg(sf_has_guard_pages)]
        self.unregister_trap_ranges();
        self.offset = 0;
        #[cfg(feature = "memprof")]
        self.trace.update(self.trace_state());
    }

    #[inline]
    pub(crate) fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity, "native code buffer overflow");
        self.offset = len;
        #[cfg(feature = "memprof")]
        self.trace.update(self.trace_state());
    }

    #[inline]
    pub(crate) fn move_region(&mut self, src_offset: usize, dst_offset: usize, len: usize) {
        assert!(
            src_offset + len <= self.offset,
            "move source beyond written region"
        );
        assert!(
            dst_offset + len <= self.capacity,
            "move destination beyond capacity"
        );
        unsafe {
            ptr::copy(self.base.add(src_offset), self.base.add(dst_offset), len);
        }
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        #[cfg(sf_has_guard_pages)]
        self.unregister_trap_ranges();
        #[cfg(feature = "memprof")]
        self.trace.remove();
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
