//! Guard-page memory allocator for the native JIT backend.
//!
//! Reserves a large virtual address region (8 GB + 64 KB) per wasm linear
//! memory. The committed (RW) region tracks the current memory size; the
//! remainder is mapped PROT_NONE so that any out-of-bounds access faults
//! into the guard region. A signal handler converts the fault to a wasm trap
//! without explicit per-access bounds checks in JIT code.
//!
//! This module is gated on `#[cfg(sf_has_guard_pages)]`. All OS coupling
//! — reservation, per-page commit, release — is delegated to
//! [`crate::vm::runtime::os`].

use crate::error::WasmError;

use super::os;

const WASM_PAGE_SIZE: usize = crate::constants::WASM_PAGE_SIZE;

/// Total virtual reservation per memory: 8 GB + 64 KB.
///
/// wasm32 max address (2^32 - 1) + max offset (2^32 - 1) + max access (16)
/// fits within this range. No OOB access can escape the guard region.
const GUARD_RESERVATION: usize = 8 * 1024 * 1024 * 1024 + 64 * 1024;

/// A wasm linear memory backed by an OS reservation with guard pages.
///
/// The base pointer is stable for the lifetime of the allocation (no
/// reallocation on grow). Only the committed size changes.
pub struct GuardPageMemory {
    base: *mut u8,
    committed: usize,
}

// SAFETY: The reserved region is process-private and not aliased.
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

        let base = os::reserve_guarded(GUARD_RESERVATION)
            .map_err(|msg| WasmError::internal(msg.into()))?;

        if initial_bytes > 0 {
            if let Err(msg) = os::commit_guarded(base, 0, initial_bytes) {
                os::release_guarded(base, GUARD_RESERVATION);
                return Err(WasmError::internal(msg.into()));
            }
        }

        let memory = Self {
            base,
            committed: initial_bytes,
        };
        #[cfg(sf_memtrace)]
        sf_nano_memtrace::record_guard_region_new(base as usize, GUARD_RESERVATION, initial_bytes);
        Ok(memory)
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
            os::commit_guarded(self.base, self.committed, new_bytes - self.committed)
                .map_err(|msg| WasmError::internal(msg.into()))?;
        }
        self.committed = new_bytes;
        #[cfg(sf_memtrace)]
        sf_nano_memtrace::record_guard_region_grow(self.base as usize, new_bytes);
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
        #[cfg(sf_memtrace)]
        sf_nano_memtrace::record_guard_region_drop(self.base as usize);
        os::release_guarded(self.base, GUARD_RESERVATION);
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
