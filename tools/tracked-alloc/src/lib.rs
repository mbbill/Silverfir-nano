#![no_std]
//! Internal allocation diagnostics. Containers are always the standard `alloc`
//! types, regardless of features. With `memprof`, a CLI-installed allocator
//! records process-wide heap bytes/counts; existing explicit runtime-buffer and
//! compilation-phase hooks provide separate statistics. No container wrappers,
//! Rust type inference, or allocation backtraces are involved.
extern crate alloc;
#[cfg(feature = "memprof")]
extern crate std;

pub use alloc::{boxed, collections, format, rc, string, vec};
pub use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    string::String,
    vec::Vec,
};
use core::alloc::{GlobalAlloc, Layout};

#[inline]
pub fn from_alloc_vec<T>(value: Vec<T>) -> Vec<T> {
    value
}
#[inline]
pub fn into_alloc_vec<T>(value: Vec<T>) -> Vec<T> {
    value
}
#[inline]
pub fn box_from_alloc<T: ?Sized>(value: Box<T>) -> Box<T> {
    value
}

/// Transfer a vector allocation to an equal-layout element type.
///
/// # Safety
/// Every initialized element must be valid as `U`, and no alias may access
/// the allocation through the old type afterwards. Layout and absence of drop
/// glue are checked before disarming the original owner.
pub unsafe fn retype_vec<T, U>(value: Vec<T>) -> Vec<U> {
    assert_eq!(core::mem::size_of::<T>(), core::mem::size_of::<U>());
    assert_eq!(core::mem::align_of::<T>(), core::mem::align_of::<U>());
    assert!(!core::mem::needs_drop::<T>() && !core::mem::needs_drop::<U>());
    let mut value = core::mem::ManuallyDrop::new(value);
    let (len, capacity) = (value.len(), value.capacity());
    let ptr = value.as_mut_ptr().cast::<U>();
    // SAFETY: checked allocation layout, caller guarantees validity and aliases.
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

pub const RUNTIME_MEMORY_OWNER: &str = "RuntimeMemory";
pub const RUNTIME_TYPE_CODE_BUFFER: &str = "CodeBuffer";
pub const RUNTIME_TYPE_GUARD_PAGE: &str = "GuardPageMemory";

/// Tracks allocations made while enabled. Bytes allocated before a reset or
/// while disabled are excluded. Profiler bookkeeping is excluded. Deallocation
/// and resize of previously recorded blocks remain accounted for while paused.
pub struct TrackingAllocator<A> {
    inner: A,
}
impl<A> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        #[cfg(feature = "memprof")]
        if !ptr.is_null() && tracking_enabled() {
            enabled::with_state(|s| s.allocate(ptr as usize, layout.size()));
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        #[cfg(feature = "memprof")]
        if !ptr.is_null() && tracking_enabled() {
            enabled::with_state(|s| s.allocate(ptr as usize, layout.size()));
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Remove before freeing: another thread may immediately reuse the address.
        #[cfg(feature = "memprof")]
        if enabled::initialized() {
            enabled::with_state(|s| s.deallocate(ptr as usize));
        }
        unsafe { self.inner.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        #[cfg(feature = "memprof")]
        if enabled::initialized() || tracking_enabled() {
            // Hold the registry lock across realloc so a moved block's old
            // address cannot be reused before its record is removed.
            if let Some(result) = enabled::with_state(|s| {
                let result = unsafe { self.inner.realloc(ptr, layout, new_size) };
                if !result.is_null() {
                    s.reallocate(ptr as usize, result as usize, new_size);
                }
                result
            }) {
                return result;
            }
        }
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocationState {
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: usize,
    pub ptr: usize,
}

impl AllocationState {
    #[inline]
    pub const fn new(size_bytes: usize) -> Self {
        Self {
            len: None,
            capacity: None,
            size_bytes,
            ptr: 0,
        }
    }

    #[inline]
    pub const fn with_len(mut self, len: usize) -> Self {
        self.len = Some(len);
        self
    }

    #[inline]
    pub const fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    #[inline]
    pub const fn with_ptr(mut self, ptr: usize) -> Self {
        self.ptr = ptr;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationDescriptor {
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
}

impl AllocationDescriptor {
    #[inline]
    pub const fn new(owner_kind: &'static str, type_name: &'static str) -> Self {
        Self {
            owner_kind,
            type_name,
            element_type: None,
        }
    }

    #[inline]
    pub const fn with_element_type(mut self, element_type: &'static str) -> Self {
        self.element_type = Some(element_type);
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct RegistrySnapshot {
    pub total_bytes: usize,
    pub peak_bytes: usize,
    pub allocated_bytes: u64,
    pub allocations: u64,
    pub reallocations: u64,
    pub deallocations: u64,
    pub live_allocations: usize,
    pub code_buffer_bytes: usize,
    pub peak_code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
    pub peak_guard_page_bytes: usize,
    pub other_runtime_bytes: usize,
    pub peak_other_runtime_bytes: usize,
    pub records: Vec<(AllocationDescriptor, AllocationState)>,
}
#[derive(Clone, Debug)]
pub struct ProfilePhase {
    pub name: &'static str,
    pub function_index: Option<u32>,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    /// Process-wide traffic during this span, including parallel workers.
    pub allocated_bytes: u64,
}
#[derive(Clone, Debug, Default)]
pub struct AllocationProfile {
    pub snapshot: RegistrySnapshot,
    pub phases: Vec<ProfilePhase>,
    pub omitted_phases: usize,
}

#[cfg(feature = "memprof")]
mod enabled;
#[cfg(feature = "memprof")]
pub use enabled::{
    phase_span, phase_span_with_function, profile, reset_tracking, set_tracking_enabled, snapshot,
    tracking_enabled, AllocationHandle, PhaseGuard,
};

#[cfg(not(feature = "memprof"))]
mod disabled {
    use super::*;
    pub fn reset_tracking() {}
    pub fn set_tracking_enabled(_enabled: bool) {}
    pub fn tracking_enabled() -> bool {
        false
    }
    pub fn snapshot() -> RegistrySnapshot {
        RegistrySnapshot::default()
    }
    pub fn profile() -> AllocationProfile {
        AllocationProfile::default()
    }
    pub struct AllocationHandle;
    impl AllocationHandle {
        pub fn new(_descriptor: AllocationDescriptor, _state: AllocationState) -> Self {
            Self
        }
        pub fn update(&mut self, _state: AllocationState) {}
        pub fn remove(&mut self) {}
    }
    pub struct PhaseGuard;
    #[must_use]
    pub fn phase_span(_name: &'static str) -> PhaseGuard {
        PhaseGuard
    }
    #[must_use]
    pub fn phase_span_with_function(
        _name: &'static str,
        _function_index: Option<u32>,
    ) -> PhaseGuard {
        PhaseGuard
    }
}
#[cfg(not(feature = "memprof"))]
pub use disabled::*;
