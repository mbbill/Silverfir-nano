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
//! [`crate::vm::jit::runtime::os`].

use crate::error::WasmError;

use super::os;
#[cfg(feature = "memprof")]
use tracked_alloc::{
    AllocationDescriptor, AllocationHandle, AllocationState, RUNTIME_MEMORY_OWNER,
};

const WASM_PAGE_SIZE: usize = crate::constants::WASM_PAGE_SIZE;

/// Total virtual reservation per memory: 8 GB + 64 KB.
///
/// wasm32 max address (2^32 - 1) + max offset (2^32 - 1) + max access (16)
/// fits within this range. No OOB access can escape the guard region.
const GUARD_RESERVATION: usize = 8 * 1024 * 1024 * 1024 + 64 * 1024;

/// Minimum guard range after the usable wasm stack.
///
/// The stack guard is also expanded by the module's largest frame footprint
/// so a single overflowing callee-frame probe cannot jump beyond the protected
/// reservation.
const STACK_GUARD_MIN_BYTES: usize = WASM_PAGE_SIZE;

#[inline]
const fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

/// A wasm linear memory backed by an OS reservation with guard pages.
///
/// The base pointer is stable for the lifetime of the allocation (no
/// reallocation on grow). Only the committed size changes.
pub(crate) struct GuardPageMemory {
    base: *mut u8,
    committed: usize,
    #[cfg(feature = "memprof")]
    trace: AllocationHandle,
}

// SAFETY: The reserved region is process-private and not aliased.
unsafe impl Send for GuardPageMemory {}

impl GuardPageMemory {
    /// Allocate a guarded memory with `initial_pages` committed.
    pub(crate) fn new(initial_pages: usize) -> Result<Self, WasmError> {
        let initial_bytes = initial_pages * WASM_PAGE_SIZE;
        if initial_bytes > GUARD_RESERVATION {
            return Err(WasmError::internal(
                "guard-page memory: initial size exceeds reservation".into(),
            ));
        }

        let base =
            os::reserve_guarded(GUARD_RESERVATION).map_err(|msg| WasmError::internal(msg))?;

        if initial_bytes > 0 {
            if let Err(msg) = os::commit_guarded(base, 0, initial_bytes) {
                os::release_guarded(base, GUARD_RESERVATION);
                return Err(WasmError::internal(msg));
            }
        }

        let memory = Self {
            base,
            committed: initial_bytes,
            #[cfg(feature = "memprof")]
            trace: AllocationHandle::new(
                AllocationDescriptor::new(RUNTIME_MEMORY_OWNER, "GuardPageMemory"),
                AllocationState::new(initial_bytes)
                    .with_len(initial_bytes)
                    .with_capacity(GUARD_RESERVATION)
                    .with_ptr(base as usize),
            ),
        };
        Ok(memory)
    }

    #[cfg(feature = "memprof")]
    #[inline]
    fn trace_state(&self) -> AllocationState {
        AllocationState::new(self.committed)
            .with_len(self.committed)
            .with_capacity(GUARD_RESERVATION)
            .with_ptr(self.base as usize)
    }

    /// Grow by `delta_pages`. Returns the old size in pages.
    pub(crate) fn grow(&mut self, delta_pages: usize) -> Result<usize, WasmError> {
        let old_pages = self.committed / WASM_PAGE_SIZE;
        let new_bytes = self
            .committed
            .checked_add(delta_pages * WASM_PAGE_SIZE)
            .ok_or_else(|| WasmError::internal("guard-page memory: grow overflow"))?;
        if new_bytes > GUARD_RESERVATION {
            return Err(WasmError::internal(
                "guard-page memory: grow exceeds reservation".into(),
            ));
        }

        if new_bytes > self.committed {
            os::commit_guarded(self.base, self.committed, new_bytes - self.committed)
                .map_err(|msg| WasmError::internal(msg))?;
        }
        self.committed = new_bytes;
        #[cfg(feature = "memprof")]
        self.trace.update(self.trace_state());
        Ok(old_pages)
    }

    #[inline]
    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.committed
    }
}

impl Drop for GuardPageMemory {
    fn drop(&mut self) {
        #[cfg(feature = "memprof")]
        self.trace.remove();
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

/// Native wasm-value stack backed by an OS reservation with a guard range
/// immediately above the usable stack bytes.
pub(crate) struct GuardPageStack {
    base: *mut u8,
    usable: usize,
    guard: usize,
    #[cfg(feature = "memprof")]
    trace: AllocationHandle,
}

// SAFETY: The reserved region is process-private and not aliased.
unsafe impl Send for GuardPageStack {}

impl GuardPageStack {
    pub(crate) fn new(requested_bytes: usize, max_frame_bytes: usize) -> Result<Self, WasmError> {
        let usable = align_up(
            requested_bytes.max(core::mem::size_of::<u64>()),
            WASM_PAGE_SIZE,
        );
        let guard = align_up(max_frame_bytes.max(STACK_GUARD_MIN_BYTES), WASM_PAGE_SIZE);
        let total = usable
            .checked_add(guard)
            .ok_or_else(|| WasmError::internal("guard-page stack: reservation overflow"))?;

        let base = os::reserve_guarded(total).map_err(|msg| WasmError::internal(msg))?;
        if let Err(msg) = os::commit_guarded(base, 0, usable) {
            os::release_guarded(base, total);
            return Err(WasmError::internal(msg));
        }

        Ok(Self {
            base,
            usable,
            guard,
            #[cfg(feature = "memprof")]
            trace: AllocationHandle::new(
                AllocationDescriptor::new(RUNTIME_MEMORY_OWNER, "GuardPageStack"),
                AllocationState::new(usable)
                    .with_len(usable)
                    .with_capacity(total)
                    .with_ptr(base as usize),
            ),
        })
    }

    #[inline]
    pub(crate) fn base(&self) -> *mut u64 {
        self.base.cast()
    }

    #[inline]
    pub(crate) fn end(&self) -> *mut u64 {
        unsafe { self.base.add(self.usable).cast() }
    }

    #[inline]
    pub(crate) fn guard_end(&self) -> *mut u8 {
        unsafe { self.base.add(self.usable + self.guard) }
    }

    #[inline]
    pub(crate) fn supports(&self, requested_bytes: usize, max_frame_bytes: usize) -> bool {
        self.usable >= requested_bytes.max(core::mem::size_of::<u64>())
            && self.guard >= max_frame_bytes.max(STACK_GUARD_MIN_BYTES)
    }
}

impl Drop for GuardPageStack {
    fn drop(&mut self) {
        #[cfg(feature = "memprof")]
        self.trace.remove();
        os::release_guarded(self.base, self.usable + self.guard);
    }
}

impl core::fmt::Debug for GuardPageStack {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GuardPageStack")
            .field("base", &self.base)
            .field("usable", &self.usable)
            .field("guard", &self.guard)
            .finish()
    }
}
