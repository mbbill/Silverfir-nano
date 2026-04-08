//! Guard-page memory allocator for the native JIT backend.
//!
//! Reserves a large virtual address region (8 GB + 64 KB) per wasm linear
//! memory. The committed (RW) region tracks the current memory size; the
//! remainder is mapped PROT_NONE so that any out-of-bounds access faults
//! into the guard region. A signal handler converts the fault to a wasm trap
//! without explicit per-access bounds checks in JIT code.
//!
//! This module is gated on `#[cfg(sf_has_guard_pages)]`.

use core::ptr;

use crate::error::WasmError;

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
}

const PROT_NONE: i32 = 0x00;
const PROT_READ: i32 = 0x01;
const PROT_WRITE: i32 = 0x02;
const MAP_PRIVATE: i32 = 0x02;
#[cfg(sf_os_macos)]
const MAP_ANON: i32 = 0x1000;
#[cfg(sf_os_linux)]
const MAP_ANON: i32 = 0x20;
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

const WASM_PAGE_SIZE: usize = crate::constants::WASM_PAGE_SIZE;

/// Total virtual reservation per memory: 8 GB + 64 KB.
///
/// wasm32 max address (2^32 - 1) + max offset (2^32 - 1) + max access (16)
/// fits within this range. No OOB access can escape the guard region.
const GUARD_RESERVATION: usize = 8 * 1024 * 1024 * 1024 + 64 * 1024;

/// A wasm linear memory backed by mmap with guard pages.
///
/// The base pointer is stable for the lifetime of the allocation (no
/// reallocation on grow). Only the committed size changes via `mprotect`.
pub struct GuardPageMemory {
    base: *mut u8,
    committed: usize,
}

// SAFETY: The mmap'd region is process-private and not aliased.
unsafe impl Send for GuardPageMemory {}

impl GuardPageMemory {
    /// Allocate a guarded memory with `initial_pages` committed.
    pub fn new(initial_pages: usize) -> Result<Self, WasmError> {
        let initial_bytes = initial_pages * WASM_PAGE_SIZE;
        if initial_bytes > GUARD_RESERVATION {
            return Err(WasmError::internal(
                "guard-page memory: initial size exceeds reservation".into(),
            ));
        }

        let base = unsafe {
            mmap(
                ptr::null_mut(),
                GUARD_RESERVATION,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if base == MAP_FAILED || base.is_null() {
            return Err(WasmError::internal("guard-page memory: mmap failed".into()));
        }

        if initial_bytes > 0 {
            let rc = unsafe { mprotect(base, initial_bytes, PROT_READ | PROT_WRITE) };
            if rc != 0 {
                unsafe { munmap(base, GUARD_RESERVATION) };
                return Err(WasmError::internal(
                    "guard-page memory: mprotect failed for initial pages".into(),
                ));
            }
        }

        Ok(Self {
            base,
            committed: initial_bytes,
        })
    }

    /// Grow by `delta_pages`. Returns the old size in pages.
    pub fn grow(&mut self, delta_pages: usize) -> Result<usize, WasmError> {
        let old_pages = self.committed / WASM_PAGE_SIZE;
        let new_bytes = self
            .committed
            .checked_add(delta_pages * WASM_PAGE_SIZE)
            .ok_or_else(|| WasmError::internal("guard-page memory: grow overflow".into()))?;
        if new_bytes > GUARD_RESERVATION {
            return Err(WasmError::internal(
                "guard-page memory: grow exceeds reservation".into(),
            ));
        }

        if new_bytes > self.committed {
            let rc = unsafe {
                mprotect(
                    self.base.add(self.committed),
                    new_bytes - self.committed,
                    PROT_READ | PROT_WRITE,
                )
            };
            if rc != 0 {
                return Err(WasmError::internal(
                    "guard-page memory: mprotect failed for grow".into(),
                ));
            }
        }
        self.committed = new_bytes;
        Ok(old_pages)
    }

    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.committed
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.base, self.committed) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.base, self.committed) }
    }
}

impl Drop for GuardPageMemory {
    fn drop(&mut self) {
        unsafe {
            munmap(self.base, GUARD_RESERVATION);
        }
    }
}

impl core::fmt::Debug for GuardPageMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GuardPageMemory")
            .field("base", &self.base)
            .field("committed", &self.committed)
            .finish()
    }
}
