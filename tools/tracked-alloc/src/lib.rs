#![no_std]
//! `tracked_alloc` mirrors `alloc` for the core crate.
//!
//! With the `memprof` feature disabled, this crate is a thin facade over the
//! normal `alloc` types and macros.
//!
//! With the `memprof` feature enabled, selected allocation owners are wrapped
//! so the CLI can record live objects, bytes, and creation sites for the HTML
//! memory profiler.
//!
//! Recording is runtime-gated. Building with `memprof` compiles in the
//! tracking-capable wrappers, but allocations remain dormant until
//! `set_tracking_enabled(true)` is called.
//!
//! Public surface map:
//! - Hooked today: [`Box`], [`Vec`], [`vec!`], [`from_alloc_vec`],
//!   [`into_alloc_vec`], [`String`], [`format!`], [`Rc`], [`BTreeMap`],
//!   [`BTreeSet`], and explicit [`AllocationHandle`] owners such as runtime
//!   buffers.
//! - Pass-through today: the non-B-tree members of [`collections`] and
//!   [`rc::Weak`].

extern crate alloc;
#[cfg(feature = "memprof")]
extern crate std;

use alloc::boxed::Box as AllocBox;
use core::alloc::GlobalAlloc;
#[cfg(feature = "memprof")]
use core::borrow::Borrow;
#[cfg(feature = "memprof")]
use core::cell::Cell;
#[cfg(feature = "memprof")]
use core::cmp::Ordering;
#[cfg(feature = "memprof")]
use core::fmt;
#[cfg(feature = "memprof")]
use core::hash::{Hash, Hasher};
#[cfg(feature = "memprof")]
use core::iter::FromIterator;
#[cfg(feature = "memprof")]
use core::marker::PhantomData;
#[cfg(feature = "memprof")]
use core::mem;
#[cfg(feature = "memprof")]
use core::ops::{Deref, DerefMut, RangeBounds};

#[cfg(feature = "memprof")]
use core::any::type_name;
#[cfg(feature = "memprof")]
use core::panic::Location;
#[cfg(feature = "memprof")]
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
#[cfg(feature = "memprof")]
use std::collections::HashMap as StdHashMap;
#[cfg(feature = "memprof")]
use std::sync::{Mutex, OnceLock};

mod inner {
    #[cfg(feature = "memprof")]
    pub use alloc::collections::btree_map::{
        Entry as BTreeMapEntry, IntoIter as BTreeMapIntoIter,
        OccupiedEntry as BTreeMapOccupiedEntry, VacantEntry as BTreeMapVacantEntry,
    };
    #[cfg(feature = "memprof")]
    pub use alloc::collections::btree_set::IntoIter as BTreeSetIntoIter;
    pub type BTreeMap<K, V> = alloc::collections::BTreeMap<K, V>;
    pub type BTreeSet<T> = alloc::collections::BTreeSet<T>;
    pub type Rc<T> = alloc::rc::Rc<T>;
    pub type String = alloc::string::String;
    pub use alloc::vec::IntoIter;
    #[cfg(feature = "memprof")]
    pub use alloc::vec::{Drain, Splice};
    pub type Vec<T> = alloc::vec::Vec<T>;
}

#[doc(hidden)]
pub mod __private {
    pub extern crate alloc as alloc_crate;
}

/// Construct this crate's [`Vec`].
///
/// This is part of the tracked surface because it goes through [`Vec`]. When
/// `memprof` is disabled it behaves like `alloc::vec!`.
#[macro_export]
macro_rules! vec {
    ($($tt:tt)*) => {
        $crate::from_alloc_vec($crate::__private::alloc_crate::vec![$($tt)*])
    };
}

/// Construct this crate's [`String`].
///
/// The formatted output is tracked as a normal [`String`] owner when
/// `memprof` is enabled.
#[macro_export]
macro_rules! format {
    ($($tt:tt)*) => {
        $crate::string::String::from($crate::__private::alloc_crate::format![$($tt)*])
    };
}

// === Tracked surface ========================================================

#[cfg(not(feature = "memprof"))]
/// A tracked vector facade over `alloc::vec::Vec<T>`.
///
/// With `memprof` disabled, this is a plain alias to `alloc::vec::Vec<T>`.
pub type Vec<T> = inner::Vec<T>;

#[cfg(not(feature = "memprof"))]
/// A tracked box facade over `alloc::boxed::Box<T>`.
///
/// With `memprof` disabled, this is a plain alias to `alloc::boxed::Box<T>`.
pub type Box<T> = AllocBox<T>;

#[cfg(not(feature = "memprof"))]
/// A tracked string facade over `alloc::string::String`.
///
/// With `memprof` disabled, this is a plain alias to `alloc::string::String`.
pub type String = inner::String;

#[cfg(not(feature = "memprof"))]
/// A tracked reference-counted facade over `alloc::rc::Rc<T>`.
///
/// With `memprof` disabled, this is a plain alias to `alloc::rc::Rc<T>`.
pub type Rc<T> = inner::Rc<T>;

#[cfg(not(feature = "memprof"))]
/// An ordered map facade over `alloc::collections::BTreeMap<K, V>`.
pub type BTreeMap<K, V> = inner::BTreeMap<K, V>;

#[cfg(not(feature = "memprof"))]
/// An ordered set facade over `alloc::collections::BTreeSet<T>`.
pub type BTreeSet<T> = inner::BTreeSet<T>;

#[cfg(not(feature = "memprof"))]
#[inline]
/// Wrap an `alloc::vec::Vec<T>` as this crate's [`Vec`].
pub fn from_alloc_vec<T>(inner: inner::Vec<T>) -> Vec<T> {
    inner
}

#[cfg(not(feature = "memprof"))]
#[inline]
/// Convert this crate's [`Vec`] back to `alloc::vec::Vec<T>`.
pub fn into_alloc_vec<T>(value: Vec<T>) -> inner::Vec<T> {
    value
}

pub struct TrackingAllocator<A> {
    inner: A,
}

impl<A> TrackingAllocator<A> {
    #[inline]
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        #[cfg(feature = "memprof")]
        if !ptr.is_null() {
            record_context_allocation(ptr, layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: core::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        #[cfg(feature = "memprof")]
        if !ptr.is_null() {
            record_context_allocation(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        unsafe { self.inner.dealloc(ptr, layout) };
        #[cfg(feature = "memprof")]
        {
            remove_unique_allocation(ptr as usize);
        }
    }

    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: core::alloc::Layout,
        new_size: usize,
    ) -> *mut u8 {
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        #[cfg(feature = "memprof")]
        if !new_ptr.is_null() {
            record_context_reallocation(ptr, new_ptr, new_size);
        }
        new_ptr
    }
}

/// Owner kind reserved for runtime-managed memory regions (e.g. the native
/// code buffer and guard-page reservations) that are registered explicitly
/// via [`AllocationHandle`] rather than going through the wrapper allocators.
///
/// Records with this owner kind are tracked separately from the normal
/// `total_bytes` so that a 16 MiB code buffer reservation does not drown out
/// the heap allocations the report is meant to highlight.
pub const RUNTIME_MEMORY_OWNER: &str = "RuntimeMemory";

/// Type-name marker for the native code buffer reservation.
pub const RUNTIME_TYPE_CODE_BUFFER: &str = "CodeBuffer";

/// Type-name marker for a guard-page backed wasm memory reservation.
pub const RUNTIME_TYPE_GUARD_PAGE: &str = "GuardPageMemory";

/// Categorization of a record outside the normal heap-tracked total.
#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeBucket {
    /// Ordinary heap allocation — counts toward `total_bytes`.
    Heap,
    /// Native code buffer (mmap'd executable region).
    CodeBuffer,
    /// Guard-page memory reservation for a wasm linear memory.
    GuardPage,
}

#[cfg(feature = "memprof")]
#[inline]
fn runtime_bucket(owner_kind: &str, type_name: &str) -> RuntimeBucket {
    if owner_kind != RUNTIME_MEMORY_OWNER {
        return RuntimeBucket::Heap;
    }
    match type_name {
        RUNTIME_TYPE_CODE_BUFFER => RuntimeBucket::CodeBuffer,
        RUNTIME_TYPE_GUARD_PAGE => RuntimeBucket::GuardPage,
        // Any future RuntimeMemory type falls into CodeBuffer by default so
        // it is at least accounted for outside the heap total; if it becomes
        // meaningful, add a dedicated bucket.
        _ => RuntimeBucket::CodeBuffer,
    }
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub records: inner::Vec<RecordSnapshot>,
    pub total_bytes: usize,
    pub code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordSnapshot {
    pub id: u64,
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: usize,
    pub ptr: usize,
    pub create_stack: AllocBox<str>,
    pub last_update_stack: Option<AllocBox<str>>,
}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn snapshot() -> RegistrySnapshot {
    RegistrySnapshot::default()
}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn reset_tracking() {}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn set_tracking_enabled(_enabled: bool) {}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn tracking_enabled() -> bool {
    false
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocationState {
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: usize,
    pub ptr: usize,
}

#[cfg(not(feature = "memprof"))]
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

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationDescriptor {
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
}

#[cfg(not(feature = "memprof"))]
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

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateEntry {
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
    pub create_stack_id: u64,
    pub count: usize,
    pub total_bytes: usize,
    pub largest_bytes: usize,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AggregateSnapshot {
    pub time_ns: u64,
    pub total_bytes: usize,
    pub code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
    pub live_records: usize,
    pub entries: inner::Vec<AggregateEntry>,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimelinePoint {
    pub time_ns: u64,
    pub total_bytes: usize,
    pub code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
    pub live_records: usize,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfilePhase {
    pub name: &'static str,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub function_index: Option<u32>,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileStack {
    pub id: u64,
    pub text: AllocBox<str>,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllocationProfile {
    pub now_ns: u64,
    pub snapshot: RegistrySnapshot,
    pub timeline: inner::Vec<TimelinePoint>,
    pub phases: inner::Vec<ProfilePhase>,
    pub stacks: inner::Vec<ProfileStack>,
    pub snapshots: inner::Vec<AggregateSnapshot>,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseGuard;

#[cfg(not(feature = "memprof"))]
#[inline]
#[must_use]
pub fn phase_span(_name: &'static str) -> PhaseGuard {
    PhaseGuard
}

#[cfg(not(feature = "memprof"))]
#[inline]
#[must_use]
pub fn phase_span_with_function(_name: &'static str, _function_index: Option<u32>) -> PhaseGuard {
    PhaseGuard
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllocationHandle;

#[cfg(not(feature = "memprof"))]
impl AllocationHandle {
    #[inline]
    pub fn new(_descriptor: AllocationDescriptor, _state: AllocationState) -> Self {
        Self
    }

    #[inline]
    pub fn update(&mut self, _state: AllocationState) {}

    #[inline]
    pub fn remove(&mut self) {}
}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn profile() -> AllocationProfile {
    AllocationProfile::default()
}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn snapshot_at(_time_ns: u64) -> RegistrySnapshot {
    RegistrySnapshot::default()
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordSnapshot {
    pub id: u64,
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: usize,
    pub ptr: usize,
    pub create_stack: AllocBox<str>,
    pub last_update_stack: Option<AllocBox<str>>,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub records: inner::Vec<RecordSnapshot>,
    pub total_bytes: usize,
    pub code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
}

#[cfg(feature = "memprof")]
#[derive(Debug)]
struct Record {
    owner_kind: &'static str,
    type_name: &'static str,
    element_type: Option<&'static str>,
    len: Option<usize>,
    capacity: Option<usize>,
    size_bytes: usize,
    ptr: usize,
    create_stack: StackId,
    last_update_stack: Option<StackId>,
}

#[cfg(feature = "memprof")]
type StackId = u64;

#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocationState {
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: usize,
    pub ptr: usize,
}

#[cfg(feature = "memprof")]
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

#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationDescriptor {
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
}

#[cfg(feature = "memprof")]
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

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateEntry {
    pub owner_kind: &'static str,
    pub type_name: &'static str,
    pub element_type: Option<&'static str>,
    pub create_stack_id: u64,
    pub count: usize,
    pub total_bytes: usize,
    pub largest_bytes: usize,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AggregateSnapshot {
    pub time_ns: u64,
    pub total_bytes: usize,
    pub code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
    pub live_records: usize,
    pub entries: inner::Vec<AggregateEntry>,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimelinePoint {
    pub time_ns: u64,
    pub total_bytes: usize,
    pub code_buffer_bytes: usize,
    pub guard_page_bytes: usize,
    pub live_records: usize,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfilePhase {
    pub name: &'static str,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub function_index: Option<u32>,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileStack {
    pub id: u64,
    pub text: AllocBox<str>,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllocationProfile {
    pub now_ns: u64,
    pub snapshot: RegistrySnapshot,
    pub timeline: inner::Vec<TimelinePoint>,
    pub phases: inner::Vec<ProfilePhase>,
    pub stacks: inner::Vec<ProfileStack>,
    pub snapshots: inner::Vec<AggregateSnapshot>,
}

#[cfg(feature = "memprof")]
#[derive(Debug)]
struct StoredStack {
    text: AllocBox<str>,
}

#[cfg(feature = "memprof")]
impl StoredStack {
    fn new(text: AllocBox<str>) -> Self {
        Self { text }
    }

    fn render(&self) -> AllocBox<str> {
        self.text.clone()
    }
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AggregateKey {
    owner_kind: &'static str,
    type_name: &'static str,
    element_type: Option<&'static str>,
    create_stack: StackId,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default)]
struct RunningAggregate {
    count: usize,
    total_bytes: usize,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationContext {
    descriptor: AllocationDescriptor,
    create_stack: Option<StackId>,
}

#[cfg(feature = "memprof")]
struct ProfilerState {
    start: std::time::Instant,
    next_stack_id: StackId,
    /// Bytes from records whose owner kind is *not* `RUNTIME_MEMORY_OWNER`
    /// (i.e. ordinary heap allocations through Vec/Box/etc.). This is the
    /// number the report headlines as "Tracked total".
    total_bytes: usize,
    /// Bytes from the native code buffer (`RUNTIME_TYPE_CODE_BUFFER`).
    /// Exposed as a separate series so the 16 MiB reservation does not
    /// dominate the tracked total.
    code_buffer_bytes: usize,
    /// Bytes from guard-page memory reservations
    /// (`RUNTIME_TYPE_GUARD_PAGE`). Exposed as a separate series.
    guard_page_bytes: usize,
    stacks: StdHashMap<StackId, StoredStack>,
    stack_ids_by_text: StdHashMap<AllocBox<str>, StackId>,
    records: StdHashMap<u64, Record>,
    timeline: inner::Vec<TimelinePoint>,
    phases: inner::Vec<ProfilePhase>,
    aggregate: StdHashMap<AggregateKey, RunningAggregate>,
    snapshots: inner::Vec<AggregateSnapshot>,
    event_count: u64,
    timeline_event_count: u64,
    last_snapshot_total_bytes: usize,
}

#[cfg(feature = "memprof")]
impl Default for ProfilerState {
    fn default() -> Self {
        Self {
            start: std::time::Instant::now(),
            next_stack_id: 1,
            total_bytes: 0,
            code_buffer_bytes: 0,
            guard_page_bytes: 0,
            stacks: StdHashMap::new(),
            stack_ids_by_text: StdHashMap::new(),
            records: StdHashMap::new(),
            timeline: inner::Vec::from([TimelinePoint::default()]),
            phases: inner::Vec::new(),
            aggregate: StdHashMap::new(),
            snapshots: inner::Vec::new(),
            event_count: 0,
            timeline_event_count: 0,
            last_snapshot_total_bytes: 0,
        }
    }
}

/// Snapshot the aggregate every this many events.
#[cfg(feature = "memprof")]
const SNAPSHOT_STRIDE: u64 = 5_000;

/// Sample a timeline point every this many events.
#[cfg(feature = "memprof")]
const TIMELINE_STRIDE: u64 = 64;

/// Force a snapshot when total_bytes changes by at least this much since the
/// last snapshot, provided at least SIGNIFICANT_MIN_EVENTS events have elapsed.
#[cfg(feature = "memprof")]
const SIGNIFICANT_DELTA_BYTES: usize = 512 * 1024;

/// Minimum events between significant-delta snapshots to avoid bursts.
#[cfg(feature = "memprof")]
const SIGNIFICANT_MIN_EVENTS: u64 = 100;

#[cfg(feature = "memprof")]
impl ProfilerState {
    fn take_aggregate_snapshot(&mut self) {
        let time_ns = elapsed_ns(self.start);
        let total_bytes = self.total_bytes;
        let code_buffer_bytes = self.code_buffer_bytes;
        let guard_page_bytes = self.guard_page_bytes;
        let live_records = self.records.len();
        // Build largest_bytes per key by scanning records.
        let mut largest: StdHashMap<AggregateKey, usize> = StdHashMap::new();
        for record in self.records.values() {
            let key = AggregateKey {
                owner_kind: record.owner_kind,
                type_name: record.type_name,
                element_type: record.element_type,
                create_stack: record.create_stack,
            };
            let entry = largest.entry(key).or_insert(0);
            if record.size_bytes > *entry {
                *entry = record.size_bytes;
            }
        }
        let mut entries: inner::Vec<AggregateEntry> = self
            .aggregate
            .iter()
            .filter(|(_, agg)| agg.count > 0)
            .map(|(key, agg)| {
                let largest_bytes = largest.get(key).copied().unwrap_or(0);
                AggregateEntry {
                    owner_kind: key.owner_kind,
                    type_name: key.type_name,
                    element_type: key.element_type,
                    create_stack_id: key.create_stack,
                    count: agg.count,
                    total_bytes: agg.total_bytes,
                    largest_bytes,
                }
            })
            .collect();
        entries.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes));
        self.snapshots.push(AggregateSnapshot {
            time_ns,
            total_bytes,
            code_buffer_bytes,
            guard_page_bytes,
            live_records,
            entries,
        });
        self.last_snapshot_total_bytes = total_bytes;
    }

    fn maybe_sample_timeline(&mut self) {
        self.timeline_event_count += 1;
        if self.timeline_event_count >= TIMELINE_STRIDE {
            self.timeline_event_count = 0;
            let time_ns = elapsed_ns(self.start);
            self.timeline.push(TimelinePoint {
                time_ns,
                total_bytes: self.total_bytes,
                code_buffer_bytes: self.code_buffer_bytes,
                guard_page_bytes: self.guard_page_bytes,
                live_records: self.records.len(),
            });
        }
    }

    fn maybe_take_snapshot(&mut self) {
        self.event_count += 1;
        if self.event_count >= SNAPSHOT_STRIDE {
            self.event_count = 0;
            self.take_aggregate_snapshot();
        } else if self.event_count >= SIGNIFICANT_MIN_EVENTS {
            let delta = self.total_bytes.abs_diff(self.last_snapshot_total_bytes);
            if delta >= SIGNIFICANT_DELTA_BYTES {
                self.event_count = 0;
                self.take_aggregate_snapshot();
            }
        }
    }

    fn aggregate_add(&mut self, record: &Record) {
        let key = AggregateKey {
            owner_kind: record.owner_kind,
            type_name: record.type_name,
            element_type: record.element_type,
            create_stack: record.create_stack,
        };
        let agg = self.aggregate.entry(key).or_default();
        agg.count += 1;
        agg.total_bytes += record.size_bytes;
    }

    fn aggregate_update(&mut self, key: &AggregateKey, old_size: usize, new_size: usize) {
        if let Some(agg) = self.aggregate.get_mut(key) {
            agg.total_bytes = agg.total_bytes.saturating_sub(old_size) + new_size;
        }
    }

    fn aggregate_remove(&mut self, record: &Record) {
        let key = AggregateKey {
            owner_kind: record.owner_kind,
            type_name: record.type_name,
            element_type: record.element_type,
            create_stack: record.create_stack,
        };
        if let Some(agg) = self.aggregate.get_mut(&key) {
            agg.count = agg.count.saturating_sub(1);
            agg.total_bytes = agg.total_bytes.saturating_sub(record.size_bytes);
        }
    }
}

#[cfg(feature = "memprof")]
static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
#[cfg(feature = "memprof")]
static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "memprof")]
static PROFILER: OnceLock<Mutex<ProfilerState>> = OnceLock::new();
#[cfg(feature = "memprof")]
static TRACKED_ALLOCATIONS: OnceLock<Mutex<StdHashMap<usize, AllocationHandle>>> = OnceLock::new();

#[cfg(feature = "memprof")]
std::thread_local! {
    static TRACKING_INTERNAL_DEPTH: Cell<u32> = const { Cell::new(0) };
    static ALLOCATION_CONTEXT: Cell<Option<AllocationContext>> = const { Cell::new(None) };
}

#[cfg(feature = "memprof")]
fn profiler() -> &'static Mutex<ProfilerState> {
    PROFILER.get_or_init(|| Mutex::new(ProfilerState::default()))
}

#[cfg(feature = "memprof")]
fn tracked_allocations() -> &'static Mutex<StdHashMap<usize, AllocationHandle>> {
    TRACKED_ALLOCATIONS.get_or_init(|| Mutex::new(StdHashMap::new()))
}

#[cfg(feature = "memprof")]
fn tracking_internal_active() -> bool {
    TRACKING_INTERNAL_DEPTH.with(|depth| depth.get() != 0)
}

#[cfg(feature = "memprof")]
fn with_tracking_internal<R>(f: impl FnOnce() -> R) -> R {
    TRACKING_INTERNAL_DEPTH.with(|depth| {
        depth.set(depth.get().saturating_add(1));
        let result = f();
        depth.set(depth.get().saturating_sub(1));
        result
    })
}

#[cfg(feature = "memprof")]
#[inline]
pub fn set_tracking_enabled(enabled: bool) {
    TRACKING_ENABLED.store(enabled, AtomicOrdering::Relaxed);
}

#[cfg(feature = "memprof")]
#[inline]
pub fn tracking_enabled() -> bool {
    TRACKING_ENABLED.load(AtomicOrdering::Relaxed)
}

#[cfg(feature = "memprof")]
#[track_caller]
fn capture_site() -> AllocBox<str> {
    let location = Location::caller();
    alloc::format!(
        "{}:{}:{}",
        location.file(),
        location.line(),
        location.column()
    )
    .into_boxed_str()
}

#[cfg(feature = "memprof")]
#[track_caller]
fn capture_stack_id() -> Option<StackId> {
    if !tracking_enabled() {
        return None;
    }
    let site_key = capture_site();
    with_tracking_internal(|| {
        let mut profiler = profiler().lock().unwrap();
        // Fast path: site already interned AND points to our code (not stdlib).
        if let Some(&id) = profiler.stack_ids_by_text.get(site_key.as_ref()) {
            return Some(id);
        }
        // Slow path: new site. Capture full backtrace for the display text
        // and for a better dedup key if the caller location is in stdlib
        // (meaning #[track_caller] broke at a closure boundary).
        let backtrace = std::backtrace::Backtrace::force_capture();
        let (bt_key, full_text) = format_backtrace(&backtrace);
        // If the caller site is in our code, use it as the dedup key.
        // If it's in stdlib (e.g. core::ops::function), use the backtrace
        // key to disambiguate different .collect() call sites.
        let key = if site_key.contains("sf-nano") || site_key.contains("sf_nano") {
            site_key
        } else {
            bt_key
        };
        if let Some(&id) = profiler.stack_ids_by_text.get(key.as_ref()) {
            return Some(id);
        }
        Some(intern_stack(&mut profiler, key, full_text))
    })
}

/// Maximum number of relevant frames to include in the dedup key.
#[cfg(feature = "memprof")]
const DEDUP_FRAMES: usize = 4;

#[cfg(feature = "memprof")]
fn is_noise_frame(frame: &str) -> bool {
    frame.contains("tracked_alloc::")
        || frame.contains("std::backtrace")
        || frame.contains("__rust_begin_short_backtrace")
        || frame.contains("__rust_end_short_backtrace")
        || frame.contains("std::rt::lang_start")
        || frame.contains("std::sys::")
        || frame.contains("start_thread")
        || frame.contains("_main")
        || frame.contains("core::ops::function")
        || frame.contains("alloc::vec::in_place_collect")
        || frame.contains("alloc::vec::Vec<T,A>::extend_desugared")
}

/// Returns (dedup_key, full_display_text).
///
/// The dedup key is built from the first few non-noise function frames so
/// that allocations from different call sites get separate groups even when
/// `#[track_caller]` collapses (e.g. inside `.collect()`).
#[cfg(feature = "memprof")]
fn format_backtrace(backtrace: &std::backtrace::Backtrace) -> (AllocBox<str>, AllocBox<str>) {
    let raw = alloc::format!("{backtrace}");
    let mut key_parts = inner::Vec::<&str>::new();
    let mut display_lines = inner::Vec::<&str>::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.parse::<usize>().is_ok() {
            continue;
        }
        if is_noise_frame(trimmed) {
            continue;
        }
        display_lines.push(trimmed);
        // Use function-name frames (not "at file:line" frames) for the key.
        if key_parts.len() < DEDUP_FRAMES && !trimmed.starts_with("at ") {
            key_parts.push(trimmed);
        }
    }
    let site_key: AllocBox<str> = key_parts.join(" <- ").into_boxed_str();
    let full_text: AllocBox<str> = display_lines.join("\n").into_boxed_str();
    (site_key, full_text)
}

#[cfg(feature = "memprof")]
#[track_caller]
fn with_allocation_context<R>(descriptor: AllocationDescriptor, f: impl FnOnce() -> R) -> R {
    if !tracking_enabled() {
        return f();
    }
    let context = AllocationContext {
        descriptor,
        create_stack: capture_stack_id(),
    };
    ALLOCATION_CONTEXT.with(|slot| {
        let previous = slot.get();
        slot.set(Some(context));
        let result = f();
        slot.set(previous);
        result
    })
}

#[cfg(feature = "memprof")]
fn with_existing_allocation_context<R>(context: AllocationContext, f: impl FnOnce() -> R) -> R {
    if !tracking_enabled() {
        return f();
    }
    ALLOCATION_CONTEXT.with(|slot| {
        let previous = slot.get();
        slot.set(Some(context));
        let result = f();
        slot.set(previous);
        result
    })
}

#[cfg(feature = "memprof")]
fn current_allocation_context() -> Option<AllocationContext> {
    if tracking_internal_active() {
        return None;
    }
    ALLOCATION_CONTEXT.with(|slot| slot.get())
}

#[cfg(feature = "memprof")]
fn intern_stack(
    profiler: &mut ProfilerState,
    site_key: AllocBox<str>,
    full_text: AllocBox<str>,
) -> StackId {
    if let Some(&id) = profiler.stack_ids_by_text.get(site_key.as_ref()) {
        return id;
    }
    let id = profiler.next_stack_id;
    profiler.next_stack_id = profiler.next_stack_id.saturating_add(1);
    profiler.stack_ids_by_text.insert(site_key, id);
    profiler.stacks.insert(id, StoredStack::new(full_text));
    id
}

#[cfg(feature = "memprof")]
fn render_stack(profiler: &ProfilerState, stack_id: StackId) -> AllocBox<str> {
    profiler
        .stacks
        .get(&stack_id)
        .expect("tracked stack exists")
        .render()
}

#[cfg(feature = "memprof")]
fn elapsed_ns(start: std::time::Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(feature = "memprof")]
fn insert_tracked_allocation(ptr: usize, mut handle: AllocationHandle) {
    if ptr == 0 {
        handle.remove();
        return;
    }
    let displaced = with_tracking_internal(|| {
        let mut allocations = tracked_allocations().lock().unwrap();
        allocations.insert(ptr, handle)
    });
    if let Some(mut displaced) = displaced {
        displaced.remove();
    }
}

#[cfg(feature = "memprof")]
fn sync_unique_allocation_with_stack(
    old_ptr: usize,
    descriptor: AllocationDescriptor,
    state: AllocationState,
    create_stack: Option<StackId>,
) {
    let mut handle = if old_ptr == 0 {
        None
    } else {
        tracked_allocations().lock().unwrap().remove(&old_ptr)
    };

    if let Some(existing) = handle.as_mut() {
        if state.size_bytes == 0 || state.ptr == 0 {
            existing.remove();
            return;
        }
        existing.retype(descriptor);
        existing.update(state);
        insert_tracked_allocation(state.ptr, handle.take().expect("tracked handle exists"));
        return;
    }

    if !tracking_enabled() || state.size_bytes == 0 || state.ptr == 0 {
        return;
    }

    let handle = AllocationHandle::new_with_stack(descriptor, state, create_stack);
    insert_tracked_allocation(state.ptr, handle);
}

#[cfg(feature = "memprof")]
#[track_caller]
fn sync_unique_allocation(
    old_ptr: usize,
    descriptor: AllocationDescriptor,
    state: AllocationState,
) {
    let create_stack = if old_ptr == 0 && state.ptr != 0 && state.size_bytes != 0 {
        capture_stack_id()
    } else {
        None
    };
    sync_unique_allocation_with_stack(old_ptr, descriptor, state, create_stack);
}

#[cfg(feature = "memprof")]
fn remove_unique_allocation(ptr: usize) {
    if ptr == 0 || tracking_internal_active() {
        return;
    }
    let mut handle = tracked_allocations().lock().unwrap().remove(&ptr);
    if let Some(handle) = handle.as_mut() {
        handle.remove();
    }
}

#[cfg(feature = "memprof")]
fn record_context_allocation(ptr: *mut u8, size_bytes: usize) {
    if ptr.is_null() || size_bytes == 0 || tracking_internal_active() {
        return;
    }
    let Some(context) = current_allocation_context() else {
        return;
    };
    let state = AllocationState::new(size_bytes).with_ptr(ptr as usize);
    let handle = AllocationHandle::new_with_stack(context.descriptor, state, context.create_stack);
    insert_tracked_allocation(ptr as usize, handle);
}

#[cfg(feature = "memprof")]
fn record_context_reallocation(old_ptr: *mut u8, new_ptr: *mut u8, new_size: usize) {
    if old_ptr.is_null() || tracking_internal_active() {
        return;
    }
    let mut handle = tracked_allocations()
        .lock()
        .unwrap()
        .remove(&(old_ptr as usize));
    if new_ptr.is_null() {
        if let Some(handle) = handle.take() {
            insert_tracked_allocation(old_ptr as usize, handle);
        }
        return;
    }

    let state = AllocationState::new(new_size).with_ptr(new_ptr as usize);
    if let Some(existing) = handle.as_mut() {
        existing.update(state);
        insert_tracked_allocation(
            new_ptr as usize,
            handle.take().expect("tracked handle exists"),
        );
        return;
    }

    let Some(context) = current_allocation_context() else {
        return;
    };
    let handle = AllocationHandle::new_with_stack(context.descriptor, state, context.create_stack);
    insert_tracked_allocation(new_ptr as usize, handle);
}

#[cfg(feature = "memprof")]
fn sort_records(records: &mut inner::Vec<RecordSnapshot>) {
    records.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| a.id.cmp(&b.id))
    });
}

#[cfg(feature = "memprof")]
fn snapshot_from_records(profiler: &ProfilerState) -> RegistrySnapshot {
    let mut records: inner::Vec<RecordSnapshot> = profiler
        .records
        .iter()
        .map(|(&id, record)| RecordSnapshot {
            id,
            owner_kind: record.owner_kind,
            type_name: record.type_name,
            element_type: record.element_type,
            len: record.len,
            capacity: record.capacity,
            size_bytes: record.size_bytes,
            ptr: record.ptr,
            create_stack: render_stack(profiler, record.create_stack),
            last_update_stack: record
                .last_update_stack
                .map(|stack_id| render_stack(profiler, stack_id)),
        })
        .collect();
    sort_records(&mut records);
    let mut total_bytes = 0usize;
    let mut code_buffer_bytes = 0usize;
    let mut guard_page_bytes = 0usize;
    for record in &records {
        match runtime_bucket(record.owner_kind, record.type_name) {
            RuntimeBucket::Heap => {
                total_bytes = total_bytes.saturating_add(record.size_bytes);
            }
            RuntimeBucket::CodeBuffer => {
                code_buffer_bytes = code_buffer_bytes.saturating_add(record.size_bytes);
            }
            RuntimeBucket::GuardPage => {
                guard_page_bytes = guard_page_bytes.saturating_add(record.size_bytes);
            }
        }
    }
    RegistrySnapshot {
        records,
        total_bytes,
        code_buffer_bytes,
        guard_page_bytes,
    }
}

#[cfg(feature = "memprof")]
#[derive(Debug)]
pub struct AllocationHandle {
    id: Option<u64>,
    descriptor: AllocationDescriptor,
    create_stack: Option<StackId>,
}

#[cfg(feature = "memprof")]
impl AllocationHandle {
    fn materialize(&mut self, state: AllocationState) {
        if self.id.is_some() || !tracking_enabled() || state.size_bytes == 0 {
            return;
        }
        with_tracking_internal(|| {
            let id = NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed) as u64;
            let create_stack = self
                .create_stack
                .or_else(capture_stack_id)
                .unwrap_or_else(|| {
                    let site_key = capture_site();
                    let backtrace = std::backtrace::Backtrace::force_capture();
                    let (bt_key, full_text) = format_backtrace(&backtrace);
                    let key = if site_key.contains("sf-nano") || site_key.contains("sf_nano") {
                        site_key
                    } else {
                        bt_key
                    };
                    let mut profiler = profiler().lock().unwrap();
                    if let Some(&id) = profiler.stack_ids_by_text.get(key.as_ref()) {
                        return id;
                    }
                    intern_stack(&mut profiler, key, full_text)
                });
            let mut profiler = profiler().lock().unwrap();
            match runtime_bucket(self.descriptor.owner_kind, self.descriptor.type_name) {
                RuntimeBucket::Heap => {
                    profiler.total_bytes = profiler.total_bytes.saturating_add(state.size_bytes);
                }
                RuntimeBucket::CodeBuffer => {
                    profiler.code_buffer_bytes =
                        profiler.code_buffer_bytes.saturating_add(state.size_bytes);
                }
                RuntimeBucket::GuardPage => {
                    profiler.guard_page_bytes =
                        profiler.guard_page_bytes.saturating_add(state.size_bytes);
                }
            }
            let record = Record {
                owner_kind: self.descriptor.owner_kind,
                type_name: self.descriptor.type_name,
                element_type: self.descriptor.element_type,
                len: state.len,
                capacity: state.capacity,
                size_bytes: state.size_bytes,
                ptr: state.ptr,
                create_stack,
                last_update_stack: None,
            };
            profiler.aggregate_add(&record);
            profiler.records.insert(id, record);
            profiler.maybe_sample_timeline();
            profiler.maybe_take_snapshot();
            self.id = Some(id);
        });
    }

    pub fn new(descriptor: AllocationDescriptor, state: AllocationState) -> Self {
        Self::new_with_stack(descriptor, state, capture_stack_id())
    }

    fn new_with_stack(
        descriptor: AllocationDescriptor,
        state: AllocationState,
        create_stack: Option<StackId>,
    ) -> Self {
        let mut handle = Self {
            id: None,
            descriptor,
            create_stack,
        };
        handle.materialize(state);
        handle
    }

    /// Change the logical owner/type of one live allocation without ending
    /// its lifetime. The record id and creation stack deliberately survive:
    /// an in-place container retype is not a free followed by an allocation.
    fn retype(&mut self, descriptor: AllocationDescriptor) {
        if self.descriptor == descriptor {
            return;
        }

        let Some(id) = self.id else {
            self.descriptor = descriptor;
            return;
        };

        with_tracking_internal(|| {
            let mut profiler = profiler().lock().unwrap();
            let Some(mut record) = profiler.records.remove(&id) else {
                self.descriptor = descriptor;
                return;
            };

            let old_bucket = runtime_bucket(record.owner_kind, record.type_name);
            let new_bucket = runtime_bucket(descriptor.owner_kind, descriptor.type_name);
            if old_bucket != new_bucket {
                match old_bucket {
                    RuntimeBucket::Heap => {
                        profiler.total_bytes =
                            profiler.total_bytes.saturating_sub(record.size_bytes);
                    }
                    RuntimeBucket::CodeBuffer => {
                        profiler.code_buffer_bytes =
                            profiler.code_buffer_bytes.saturating_sub(record.size_bytes);
                    }
                    RuntimeBucket::GuardPage => {
                        profiler.guard_page_bytes =
                            profiler.guard_page_bytes.saturating_sub(record.size_bytes);
                    }
                }
                match new_bucket {
                    RuntimeBucket::Heap => {
                        profiler.total_bytes =
                            profiler.total_bytes.saturating_add(record.size_bytes);
                    }
                    RuntimeBucket::CodeBuffer => {
                        profiler.code_buffer_bytes =
                            profiler.code_buffer_bytes.saturating_add(record.size_bytes);
                    }
                    RuntimeBucket::GuardPage => {
                        profiler.guard_page_bytes =
                            profiler.guard_page_bytes.saturating_add(record.size_bytes);
                    }
                }
            }

            profiler.aggregate_remove(&record);
            record.owner_kind = descriptor.owner_kind;
            record.type_name = descriptor.type_name;
            record.element_type = descriptor.element_type;
            profiler.aggregate_add(&record);
            profiler.records.insert(id, record);
            self.descriptor = descriptor;
        });
    }

    pub fn update(&mut self, state: AllocationState) {
        if self.id.is_none() {
            self.materialize(state);
            return;
        }
        if state.size_bytes == 0 {
            self.remove();
            return;
        }
        let Some(id) = self.id else {
            return;
        };
        with_tracking_internal(|| {
            let mut profiler = profiler().lock().unwrap();
            let Some(record) = profiler.records.get(&id) else {
                return;
            };
            let size_changed = record.capacity != state.capacity
                || record.size_bytes != state.size_bytes
                || record.ptr != state.ptr;
            let old_size_bytes = record.size_bytes;
            let bucket = runtime_bucket(record.owner_kind, record.type_name);
            if !size_changed {
                if record.len != state.len {
                    let record = profiler
                        .records
                        .get_mut(&id)
                        .expect("tracked allocation exists");
                    record.len = state.len;
                }
                return;
            }

            let agg_key = {
                let record = profiler
                    .records
                    .get(&id)
                    .expect("tracked allocation exists");
                AggregateKey {
                    owner_kind: record.owner_kind,
                    type_name: record.type_name,
                    element_type: record.element_type,
                    create_stack: record.create_stack,
                }
            };
            {
                let record = profiler
                    .records
                    .get_mut(&id)
                    .expect("tracked allocation exists");
                record.len = state.len;
                record.capacity = state.capacity;
                record.size_bytes = state.size_bytes;
                record.ptr = state.ptr;
                record.last_update_stack = None;
            };
            match bucket {
                RuntimeBucket::Heap => {
                    profiler.total_bytes = profiler
                        .total_bytes
                        .saturating_sub(old_size_bytes)
                        .saturating_add(state.size_bytes);
                }
                RuntimeBucket::CodeBuffer => {
                    profiler.code_buffer_bytes = profiler
                        .code_buffer_bytes
                        .saturating_sub(old_size_bytes)
                        .saturating_add(state.size_bytes);
                }
                RuntimeBucket::GuardPage => {
                    profiler.guard_page_bytes = profiler
                        .guard_page_bytes
                        .saturating_sub(old_size_bytes)
                        .saturating_add(state.size_bytes);
                }
            }
            profiler.aggregate_update(&agg_key, old_size_bytes, state.size_bytes);
            profiler.maybe_sample_timeline();
            profiler.maybe_take_snapshot();
        });
    }

    pub fn remove(&mut self) {
        let Some(id) = self.id.take() else {
            return;
        };
        with_tracking_internal(|| {
            let mut profiler = profiler().lock().unwrap();
            let Some(record) = profiler.records.remove(&id) else {
                return;
            };
            match runtime_bucket(record.owner_kind, record.type_name) {
                RuntimeBucket::Heap => {
                    profiler.total_bytes = profiler.total_bytes.saturating_sub(record.size_bytes);
                }
                RuntimeBucket::CodeBuffer => {
                    profiler.code_buffer_bytes =
                        profiler.code_buffer_bytes.saturating_sub(record.size_bytes);
                }
                RuntimeBucket::GuardPage => {
                    profiler.guard_page_bytes =
                        profiler.guard_page_bytes.saturating_sub(record.size_bytes);
                }
            }
            profiler.aggregate_remove(&record);
            profiler.maybe_sample_timeline();
            profiler.maybe_take_snapshot();
        });
    }
}

#[cfg(feature = "memprof")]
impl Drop for AllocationHandle {
    fn drop(&mut self) {
        self.remove();
    }
}

#[cfg(feature = "memprof")]
#[derive(Debug)]
struct SharedAllocationRecord {
    refs: usize,
    trace: AllocationHandle,
}

#[cfg(feature = "memprof")]
static SHARED_ALLOCATIONS: OnceLock<Mutex<StdHashMap<usize, SharedAllocationRecord>>> =
    OnceLock::new();

#[cfg(feature = "memprof")]
fn shared_allocations() -> &'static Mutex<StdHashMap<usize, SharedAllocationRecord>> {
    SHARED_ALLOCATIONS.get_or_init(|| Mutex::new(StdHashMap::new()))
}

#[cfg(feature = "memprof")]
fn retain_shared_allocation_with_stack(
    ptr: usize,
    descriptor: AllocationDescriptor,
    state: AllocationState,
    create_stack: Option<StackId>,
) {
    if ptr == 0 {
        return;
    }
    let found_existing = {
        let mut shared = shared_allocations().lock().unwrap();
        if let Some(record) = shared.get_mut(&ptr) {
            record.refs = record.refs.saturating_add(1);
            true
        } else {
            false
        }
    };
    if found_existing || !tracking_enabled() {
        return;
    }
    let trace = AllocationHandle::new_with_stack(descriptor, state, create_stack);
    let mut shared = shared_allocations().lock().unwrap();
    if let Some(record) = shared.get_mut(&ptr) {
        record.refs = record.refs.saturating_add(1);
    } else {
        shared.insert(ptr, SharedAllocationRecord { refs: 1, trace });
    }
}

#[cfg(feature = "memprof")]
fn release_shared_allocation(ptr: usize) {
    if ptr == 0 || tracking_internal_active() {
        return;
    }
    let maybe_trace = {
        let mut shared = shared_allocations().lock().unwrap();
        let Some(mut record) = shared.remove(&ptr) else {
            return;
        };
        if record.refs > 1 {
            record.refs -= 1;
            shared.insert(ptr, record);
            None
        } else {
            Some(record.trace)
        }
    };
    if let Some(mut trace) = maybe_trace {
        trace.remove();
    }
}

#[cfg(feature = "memprof")]
pub fn snapshot() -> RegistrySnapshot {
    let profiler = profiler().lock().unwrap();
    snapshot_from_records(&profiler)
}

#[cfg(feature = "memprof")]
pub fn profile() -> AllocationProfile {
    let profiler = profiler().lock().unwrap();
    let mut stacks: inner::Vec<ProfileStack> = profiler
        .stacks
        .iter()
        .map(|(&id, stack)| ProfileStack {
            id,
            text: stack.render(),
        })
        .collect();
    stacks.sort_by(|a, b| a.id.cmp(&b.id));
    let mut phases = profiler.phases.clone();
    phases.sort_by(|a, b| {
        a.start_time_ns
            .cmp(&b.start_time_ns)
            .then_with(|| a.end_time_ns.cmp(&b.end_time_ns))
            .then_with(|| a.name.cmp(b.name))
            .then_with(|| a.function_index.cmp(&b.function_index))
    });
    AllocationProfile {
        now_ns: elapsed_ns(profiler.start),
        snapshot: snapshot_from_records(&profiler),
        timeline: profiler.timeline.clone(),
        phases,
        stacks,
        snapshots: profiler.snapshots.clone(),
    }
}

#[cfg(feature = "memprof")]
pub fn snapshot_at(time_ns: u64) -> RegistrySnapshot {
    let profiler = profiler().lock().unwrap();
    // Find the last aggregate snapshot at or before the requested time.
    let snap = profiler
        .snapshots
        .iter()
        .rev()
        .find(|s| s.time_ns <= time_ns);
    let Some(snap) = snap else {
        return RegistrySnapshot {
            records: inner::Vec::new(),
            total_bytes: 0,
            code_buffer_bytes: 0,
            guard_page_bytes: 0,
        };
    };
    // Synthesize RecordSnapshots from aggregate entries (one per group).
    let mut records: inner::Vec<RecordSnapshot> = snap
        .entries
        .iter()
        .map(|entry| RecordSnapshot {
            id: 0,
            owner_kind: entry.owner_kind,
            type_name: entry.type_name,
            element_type: entry.element_type,
            len: None,
            capacity: None,
            size_bytes: entry.total_bytes,
            ptr: 0,
            create_stack: render_stack(&profiler, entry.create_stack_id),
            last_update_stack: None,
        })
        .collect();
    sort_records(&mut records);
    RegistrySnapshot {
        total_bytes: snap.total_bytes,
        code_buffer_bytes: snap.code_buffer_bytes,
        guard_page_bytes: snap.guard_page_bytes,
        records,
    }
}

#[cfg(feature = "memprof")]
pub fn reset_tracking() {
    let mut profiler = profiler().lock().unwrap();
    profiler.start = std::time::Instant::now();
    profiler.next_stack_id = 1;
    profiler.total_bytes = 0;
    profiler.code_buffer_bytes = 0;
    profiler.guard_page_bytes = 0;
    profiler.stacks.clear();
    profiler.stack_ids_by_text.clear();
    profiler.records.clear();
    profiler.timeline.clear();
    profiler.timeline.push(TimelinePoint::default());
    profiler.phases.clear();
    profiler.aggregate.clear();
    profiler.snapshots.clear();
    profiler.event_count = 0;
    profiler.timeline_event_count = 0;
    profiler.last_snapshot_total_bytes = 0;
    NEXT_ID.store(1, AtomicOrdering::Relaxed);
    tracked_allocations().lock().unwrap().clear();
    shared_allocations().lock().unwrap().clear();
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseGuard {
    name: Option<&'static str>,
    function_index: Option<u32>,
    start_time_ns: u64,
}

#[cfg(feature = "memprof")]
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        let Some(name) = self.name.take() else {
            return;
        };
        let mut profiler = profiler().lock().unwrap();
        let end_time_ns = elapsed_ns(profiler.start);
        profiler.phases.push(ProfilePhase {
            name,
            start_time_ns: self.start_time_ns,
            end_time_ns,
            function_index: self.function_index,
        });
    }
}

#[cfg(feature = "memprof")]
#[inline]
#[must_use]
pub fn phase_span(name: &'static str) -> PhaseGuard {
    phase_span_with_function(name, None)
}

#[cfg(feature = "memprof")]
#[must_use]
pub fn phase_span_with_function(name: &'static str, function_index: Option<u32>) -> PhaseGuard {
    if !tracking_enabled() {
        return PhaseGuard::default();
    }
    let profiler = profiler().lock().unwrap();
    PhaseGuard {
        name: Some(name),
        function_index,
        start_time_ns: elapsed_ns(profiler.start),
    }
}

#[cfg(feature = "memprof")]
fn buffer_ptr<T>(inner: &inner::Vec<T>) -> usize {
    if inner.capacity() == 0 {
        0
    } else {
        inner.as_ptr() as usize
    }
}

#[cfg(feature = "memprof")]
fn vec_state<T>(inner: &inner::Vec<T>) -> AllocationState {
    AllocationState::new(inner.capacity().saturating_mul(mem::size_of::<T>()))
        .with_len(inner.len())
        .with_capacity(inner.capacity())
        .with_ptr(buffer_ptr(inner))
}

#[cfg(feature = "memprof")]
fn string_ptr(inner: &inner::String) -> usize {
    if inner.capacity() == 0 {
        0
    } else {
        inner.as_ptr() as usize
    }
}

#[cfg(feature = "memprof")]
fn string_state(inner: &inner::String) -> AllocationState {
    AllocationState::new(inner.capacity())
        .with_len(inner.len())
        .with_capacity(inner.capacity())
        .with_ptr(string_ptr(inner))
}

#[cfg(feature = "memprof")]
fn box_ptr<T: ?Sized>(inner: &AllocBox<T>) -> usize {
    let size = mem::size_of_val(inner.as_ref());
    if size == 0 {
        0
    } else {
        inner.as_ref() as *const T as *const () as usize
    }
}

#[cfg(feature = "memprof")]
fn box_state<T: ?Sized>(inner: &AllocBox<T>) -> AllocationState {
    AllocationState::new(mem::size_of_val(inner.as_ref())).with_ptr(box_ptr(inner))
}

#[cfg(feature = "memprof")]
fn box_slice_state<T>(inner: &AllocBox<[T]>) -> AllocationState {
    box_state(inner)
        .with_len(inner.len())
        .with_capacity(inner.len())
}

#[cfg(feature = "memprof")]
fn box_str_state(inner: &AllocBox<str>) -> AllocationState {
    box_state(inner)
        .with_len(inner.len())
        .with_capacity(inner.len())
}

#[cfg(feature = "memprof")]
fn rc_header_bytes() -> usize {
    2usize.saturating_mul(mem::size_of::<usize>())
}

#[cfg(feature = "memprof")]
fn rc_ptr<T: ?Sized>(inner: &inner::Rc<T>) -> usize {
    inner::Rc::as_ptr(inner) as *const () as usize
}

#[cfg(feature = "memprof")]
fn rc_state<T: ?Sized>(inner: &inner::Rc<T>) -> AllocationState {
    AllocationState::new(rc_header_bytes().saturating_add(mem::size_of_val(inner.as_ref())))
        .with_ptr(rc_ptr(inner))
}

#[cfg(feature = "memprof")]
fn rc_slice_state<T>(inner: &inner::Rc<[T]>) -> AllocationState {
    rc_state(inner).with_len(inner.len())
}

#[cfg(feature = "memprof")]
fn rc_str_state(inner: &inner::Rc<str>) -> AllocationState {
    rc_state(inner).with_len(inner.len())
}

#[cfg(feature = "memprof")]
/// A tracked box.
///
/// This records heap usage for uniquely-owned box allocations. For
/// `Box<[T]>` and `Box<str>`, length is reported directly from the boxed data.
#[repr(transparent)]
pub struct Box<T: ?Sized> {
    inner: AllocBox<T>,
}

#[cfg(feature = "memprof")]
impl<T> Box<T> {
    #[inline]
    #[track_caller]
    pub fn new(value: T) -> Self {
        Self::from_alloc_box(AllocBox::new(value))
    }

    #[inline]
    #[track_caller]
    pub fn from_alloc_box(inner: AllocBox<T>) -> Self {
        let state = box_state(&inner);
        Self::from_alloc_box_with(
            inner,
            AllocationDescriptor::new("Box", type_name::<T>()),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Box<T> {
    #[inline]
    #[track_caller]
    fn from_alloc_box_with(
        inner: AllocBox<T>,
        descriptor: AllocationDescriptor,
        state: AllocationState,
    ) -> Self {
        let boxed = Self { inner };
        sync_unique_allocation_with_stack(0, descriptor, state, capture_stack_id());
        boxed
    }

    /// Wrap an already-built `alloc` box whose target may be unsized
    /// (trait objects) — the sized `from_alloc_box`/`From` impls cannot
    /// cover `dyn` targets without overlapping the slice and str impls.
    #[inline]
    #[track_caller]
    pub fn from_alloc_box_unsized(inner: AllocBox<T>) -> Self {
        let state = box_state(&inner);
        Self::from_alloc_box_with(
            inner,
            AllocationDescriptor::new("Box", type_name::<T>()),
            state,
        )
    }

    #[inline]
    pub fn into_alloc_box(self) -> AllocBox<T> {
        let this = mem::ManuallyDrop::new(self);
        remove_unique_allocation(box_ptr(&this.inner));
        unsafe { core::ptr::read(&this.inner) }
    }

    #[inline]
    pub fn leak<'a>(value: Self) -> &'a mut T
    where
        T: 'a,
    {
        let this = mem::ManuallyDrop::new(value);
        let inner = unsafe { core::ptr::read(&this.inner) };
        AllocBox::leak(inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Box<[T]> {
    #[inline]
    #[track_caller]
    pub fn from_alloc_boxed_slice(inner: AllocBox<[T]>) -> Self {
        let state = box_slice_state(&inner);
        Self::from_alloc_box_with(
            inner,
            AllocationDescriptor::new("Box", type_name::<[T]>())
                .with_element_type(type_name::<T>()),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl Box<str> {
    #[inline]
    #[track_caller]
    pub fn from_alloc_boxed_str(inner: AllocBox<str>) -> Self {
        let state = box_str_state(&inner);
        Self::from_alloc_box_with(
            inner,
            AllocationDescriptor::new("Box", type_name::<str>()).with_element_type("u8"),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Drop for Box<T> {
    fn drop(&mut self) {
        remove_unique_allocation(box_ptr(&self.inner));
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> Clone for Box<T> {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_box(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> Clone for Box<[T]> {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_boxed_slice(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl Clone for Box<str> {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_boxed_str(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl<T> From<AllocBox<T>> for Box<T> {
    #[track_caller]
    fn from(value: AllocBox<T>) -> Self {
        Self::from_alloc_box(value)
    }
}

#[cfg(feature = "memprof")]
impl<T> From<AllocBox<[T]>> for Box<[T]> {
    #[track_caller]
    fn from(value: AllocBox<[T]>) -> Self {
        Self::from_alloc_boxed_slice(value)
    }
}

#[cfg(feature = "memprof")]
impl From<AllocBox<str>> for Box<str> {
    #[track_caller]
    fn from(value: AllocBox<str>) -> Self {
        Self::from_alloc_boxed_str(value)
    }
}

#[cfg(feature = "memprof")]
impl<T, const N: usize> From<[T; N]> for Box<[T]> {
    #[track_caller]
    fn from(value: [T; N]) -> Self {
        Self::from_alloc_boxed_slice(AllocBox::from(value))
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> From<&[T]> for Box<[T]> {
    #[track_caller]
    fn from(value: &[T]) -> Self {
        Self::from_alloc_boxed_slice(AllocBox::from(value))
    }
}

#[cfg(feature = "memprof")]
impl From<&str> for Box<str> {
    #[track_caller]
    fn from(value: &str) -> Self {
        Self::from_alloc_boxed_str(AllocBox::from(value))
    }
}

#[cfg(feature = "memprof")]
impl From<inner::String> for Box<str> {
    #[track_caller]
    fn from(value: inner::String) -> Self {
        Self::from_alloc_boxed_str(AllocBox::from(value))
    }
}

#[cfg(feature = "memprof")]
impl<T: Default> Default for Box<T> {
    #[track_caller]
    fn default() -> Self {
        Self::new(T::default())
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        self.inner.as_ref()
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> AsMut<T> for Box<T> {
    fn as_mut(&mut self) -> &mut T {
        self.inner.as_mut()
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> fmt::Debug for Box<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> PartialEq for Box<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Eq for Box<T> where T: Eq {}

#[cfg(feature = "memprof")]
impl<T: ?Sized> PartialOrd for Box<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Ord for Box<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Hash for Box<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

#[cfg(feature = "memprof")]
#[cfg(feature = "memprof")]
/// A tracked vector.
///
/// This records allocation state from the vector's backing buffer directly.
#[repr(transparent)]
pub struct Vec<T> {
    inner: inner::Vec<T>,
}

#[cfg(feature = "memprof")]
impl<T> Vec<T> {
    #[inline]
    #[track_caller]
    pub fn new() -> Self {
        Self::from_alloc_vec(inner::Vec::new())
    }

    #[inline]
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_alloc_vec(inner::Vec::with_capacity(capacity))
    }

    #[inline]
    #[track_caller]
    pub fn from_alloc_vec(inner: inner::Vec<T>) -> Self {
        let values = Self { inner };
        sync_unique_allocation(
            0,
            AllocationDescriptor::new("Vec", type_name::<T>()).with_element_type(type_name::<T>()),
            vec_state(&values.inner),
        );
        values
    }

    #[inline]
    #[track_caller]
    fn sync_from_old_ptr(&mut self, old_ptr: usize) {
        sync_unique_allocation(
            old_ptr,
            AllocationDescriptor::new("Vec", type_name::<T>()).with_element_type(type_name::<T>()),
            vec_state(&self.inner),
        );
    }

    #[track_caller]
    fn mutate<R>(&mut self, f: impl FnOnce(&mut inner::Vec<T>) -> R) -> R {
        let old_ptr = buffer_ptr(&self.inner);
        let result = f(&mut self.inner);
        self.sync_from_old_ptr(old_ptr);
        result
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    #[inline]
    #[track_caller]
    pub fn push(&mut self, value: T) {
        self.mutate(|inner| inner.push(value));
    }

    #[inline]
    #[track_caller]
    pub fn pop(&mut self) -> Option<T> {
        self.mutate(|inner| inner.pop())
    }

    #[inline]
    #[track_caller]
    pub fn clear(&mut self) {
        self.mutate(inner::Vec::clear);
    }

    #[inline]
    #[track_caller]
    pub fn truncate(&mut self, len: usize) {
        self.mutate(|inner| inner.truncate(len));
    }

    #[inline]
    #[track_caller]
    pub fn reserve(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve(additional));
    }

    #[inline]
    #[track_caller]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve_exact(additional));
    }

    #[inline]
    #[track_caller]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), collections::TryReserveError> {
        let old_ptr = buffer_ptr(&self.inner);
        let result = self.inner.try_reserve(additional);
        self.sync_from_old_ptr(old_ptr);
        result
    }

    #[inline]
    #[track_caller]
    pub fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), collections::TryReserveError> {
        let old_ptr = buffer_ptr(&self.inner);
        let result = self.inner.try_reserve_exact(additional);
        self.sync_from_old_ptr(old_ptr);
        result
    }

    #[inline]
    #[track_caller]
    pub fn shrink_to_fit(&mut self) {
        self.mutate(inner::Vec::shrink_to_fit);
    }

    #[inline]
    #[track_caller]
    pub fn insert(&mut self, index: usize, element: T) {
        self.mutate(|inner| inner.insert(index, element));
    }

    #[inline]
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> T {
        self.mutate(|inner| inner.remove(index))
    }

    #[inline]
    #[track_caller]
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.mutate(|inner| inner.swap_remove(index))
    }

    #[inline]
    #[track_caller]
    pub fn append(&mut self, other: &mut Self) {
        let self_old_ptr = buffer_ptr(&self.inner);
        let other_old_ptr = buffer_ptr(&other.inner);
        self.inner.append(&mut other.inner);
        self.sync_from_old_ptr(self_old_ptr);
        other.sync_from_old_ptr(other_old_ptr);
    }

    #[inline]
    #[track_caller]
    pub fn retain(&mut self, f: impl FnMut(&T) -> bool) {
        self.mutate(|inner| inner.retain(f));
    }

    #[inline]
    #[track_caller]
    pub fn retain_mut(&mut self, f: impl FnMut(&mut T) -> bool) {
        self.mutate(|inner| inner.retain_mut(f));
    }

    #[inline]
    #[track_caller]
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.mutate(inner::Vec::dedup);
    }

    #[inline]
    #[track_caller]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.mutate(|inner| inner.resize(new_len, value));
    }

    #[inline]
    #[track_caller]
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        self.mutate(|inner| inner.resize_with(new_len, f));
    }

    #[inline]
    #[track_caller]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.mutate(|inner| inner.extend_from_slice(other));
    }

    #[inline]
    #[track_caller]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.mutate(|inner| inner.as_mut_slice().sort());
    }

    #[inline]
    #[track_caller]
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.mutate(|inner| inner.sort_by(compare));
    }

    #[inline]
    #[track_caller]
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.mutate(|inner| inner.as_mut_slice().sort_unstable());
    }

    #[inline]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let owner = self as *mut Self;
        let old_ptr = buffer_ptr(&self.inner);
        let inner = self.inner.drain(range);
        Drain {
            inner: Some(inner),
            owner,
            old_ptr,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, T, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        let owner = self as *mut Self;
        let old_ptr = buffer_ptr(&self.inner);
        let inner = self.inner.splice(range, replace_with);
        Splice {
            inner: Some(inner),
            owner,
            old_ptr,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        remove_unique_allocation(buffer_ptr(&self.inner));
        Box::from_alloc_boxed_slice(mem::take(&mut self.inner).into_boxed_slice())
    }

    #[inline]
    pub fn into_alloc_vec(mut self) -> inner::Vec<T> {
        remove_unique_allocation(buffer_ptr(&self.inner));
        mem::take(&mut self.inner)
    }
}

#[cfg(feature = "memprof")]
#[inline]
#[track_caller]
pub fn from_alloc_vec<T>(inner: inner::Vec<T>) -> Vec<T> {
    Vec::from_alloc_vec(inner)
}

#[cfg(feature = "memprof")]
#[inline]
pub fn into_alloc_vec<T>(value: Vec<T>) -> inner::Vec<T> {
    value.into_alloc_vec()
}

/// Consume a vector and return its allocation as raw parts.
///
/// With memory profiling enabled the live allocation record remains attached
/// to the pointer so a matching [`from_raw_parts`] can retype it without a
/// false free/allocation pair. The caller must either rebuild an owning
/// collection or otherwise release the allocation according to
/// `alloc::vec::Vec`'s raw-parts contract.
pub fn into_raw_parts<T>(value: Vec<T>) -> (*mut T, usize, usize) {
    let mut value = core::mem::ManuallyDrop::new(value);
    (value.as_mut_ptr(), value.len(), value.capacity())
}

/// Rebuild a vector from an allocation previously split into raw parts.
///
/// With memory profiling enabled an existing live record for `ptr` is retyped
/// in place, preserving its id and creation site. A previously untracked
/// allocation is registered as a new vector owner. This is the tracking-aware
/// counterpart of `alloc::vec::Vec::from_raw_parts`.
///
/// # Safety
///
/// The caller must uphold all requirements of
/// `alloc::vec::Vec::from_raw_parts`: `ptr` must have been allocated by the
/// global allocator for `T` with exactly this capacity/layout, the first
/// `len` elements must be initialized valid `T` values, and no other owner or
/// live reference may access the allocation after this call.
pub unsafe fn from_raw_parts<T>(ptr: *mut T, len: usize, capacity: usize) -> Vec<T> {
    // Construct the owner before touching tracking state so unwinding from the
    // profiler cannot leak the raw allocation.
    // SAFETY: forwarded from this function's contract.
    let inner = unsafe { inner::Vec::from_raw_parts(ptr, len, capacity) };
    #[cfg(not(feature = "memprof"))]
    {
        inner
    }
    #[cfg(feature = "memprof")]
    {
        let values = Vec { inner };
        sync_unique_allocation_with_stack(
            buffer_ptr(&values.inner),
            AllocationDescriptor::new("Vec", type_name::<T>()).with_element_type(type_name::<T>()),
            vec_state(&values.inner),
            None,
        );
        values
    }
}

#[cfg(feature = "memprof")]
pub struct Drain<'a, T> {
    inner: Option<inner::Drain<'a, T>>,
    owner: *mut Vec<T>,
    old_ptr: usize,
    marker: PhantomData<&'a mut Vec<T>>,
}

#[cfg(feature = "memprof")]
impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(Iterator::next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner
            .as_ref()
            .map(Iterator::size_hint)
            .unwrap_or((0, Some(0)))
    }
}

#[cfg(feature = "memprof")]
impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(DoubleEndedIterator::next_back)
    }
}

#[cfg(feature = "memprof")]
impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

#[cfg(feature = "memprof")]
impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        drop(self.inner.take());
        unsafe {
            (*self.owner).sync_from_old_ptr(self.old_ptr);
        }
    }
}

#[cfg(feature = "memprof")]
pub struct Splice<'a, T, I: Iterator<Item = T>> {
    inner: Option<inner::Splice<'a, I>>,
    owner: *mut Vec<T>,
    old_ptr: usize,
    marker: PhantomData<&'a mut Vec<T>>,
}

#[cfg(feature = "memprof")]
impl<'a, T, I: Iterator<Item = T>> Iterator for Splice<'a, T, I> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(Iterator::next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner
            .as_ref()
            .map(Iterator::size_hint)
            .unwrap_or((0, Some(0)))
    }
}

#[cfg(feature = "memprof")]
impl<'a, T, I: Iterator<Item = T>> DoubleEndedIterator for Splice<'a, T, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().and_then(DoubleEndedIterator::next_back)
    }
}

#[cfg(feature = "memprof")]
impl<'a, T, I: Iterator<Item = T>> Drop for Splice<'a, T, I> {
    fn drop(&mut self) {
        drop(self.inner.take());
        unsafe {
            (*self.owner).sync_from_old_ptr(self.old_ptr);
        }
    }
}

#[cfg(feature = "memprof")]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        remove_unique_allocation(buffer_ptr(&self.inner));
    }
}

#[cfg(feature = "memprof")]
impl<T> Default for Vec<T> {
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> Clone for Vec<T> {
    #[track_caller]
    fn clone(&self) -> Self {
        from_alloc_vec(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl<T> From<inner::Vec<T>> for Vec<T> {
    #[track_caller]
    fn from(value: inner::Vec<T>) -> Self {
        from_alloc_vec(value)
    }
}

#[cfg(feature = "memprof")]
impl<T> From<Vec<T>> for inner::Vec<T> {
    fn from(value: Vec<T>) -> Self {
        value.into_alloc_vec()
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> From<&[T]> for Vec<T> {
    #[track_caller]
    fn from(value: &[T]) -> Self {
        from_alloc_vec(value.to_vec())
    }
}

#[cfg(feature = "memprof")]
impl<T, const N: usize> From<[T; N]> for Vec<T> {
    #[track_caller]
    fn from(value: [T; N]) -> Self {
        from_alloc_vec(inner::Vec::from(value))
    }
}

#[cfg(feature = "memprof")]
impl<T> Extend<T> for Vec<T> {
    #[track_caller]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let old_ptr = buffer_ptr(&self.inner);
        self.inner.extend(iter);
        self.sync_from_old_ptr(old_ptr);
    }
}

#[cfg(feature = "memprof")]
impl<'a, T: 'a + Clone> Extend<&'a T> for Vec<T> {
    #[track_caller]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let old_ptr = buffer_ptr(&self.inner);
        self.inner.extend(iter.into_iter().cloned());
        self.sync_from_old_ptr(old_ptr);
    }
}

#[cfg(feature = "memprof")]
impl<T> FromIterator<T> for Vec<T> {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        from_alloc_vec(inner::Vec::from_iter(iter))
    }
}

#[cfg(feature = "memprof")]
impl<T> Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

#[cfg(feature = "memprof")]
impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

#[cfg(feature = "memprof")]
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[cfg(feature = "memprof")]
impl<T> AsMut<[T]> for Vec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "memprof")]
impl<T> fmt::Debug for Vec<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl<T> PartialEq for Vec<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Eq for Vec<T> where T: Eq {}

#[cfg(feature = "memprof")]
impl<T> PartialOrd for Vec<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Ord for Vec<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Hash for Vec<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

#[cfg(feature = "memprof")]
impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = inner::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_alloc_vec().into_iter()
    }
}

#[cfg(feature = "memprof")]
impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(feature = "memprof")]
impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(feature = "memprof")]
/// A tracked string.
///
/// This records allocation state from the string's UTF-8 backing buffer.
#[repr(transparent)]
pub struct String {
    inner: inner::String,
}

#[cfg(feature = "memprof")]
impl String {
    #[inline]
    #[track_caller]
    pub fn new() -> Self {
        Self::from_alloc_string(inner::String::new())
    }

    #[inline]
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_alloc_string(inner::String::with_capacity(capacity))
    }

    #[inline]
    #[track_caller]
    pub fn from_alloc_string(inner: inner::String) -> Self {
        let value = Self { inner };
        sync_unique_allocation(
            0,
            AllocationDescriptor::new("String", type_name::<str>()).with_element_type("u8"),
            string_state(&value.inner),
        );
        value
    }

    #[inline]
    #[track_caller]
    fn sync_from_old_ptr(&mut self, old_ptr: usize) {
        sync_unique_allocation(
            old_ptr,
            AllocationDescriptor::new("String", type_name::<str>()).with_element_type("u8"),
            string_state(&self.inner),
        );
    }

    #[track_caller]
    fn mutate<R>(&mut self, f: impl FnOnce(&mut inner::String) -> R) -> R {
        let old_ptr = string_ptr(&self.inner);
        let result = f(&mut self.inner);
        self.sync_from_old_ptr(old_ptr);
        result
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self.inner.as_str()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    #[inline]
    #[track_caller]
    pub fn clear(&mut self) {
        self.mutate(inner::String::clear);
    }

    #[inline]
    #[track_caller]
    pub fn push(&mut self, ch: char) {
        self.mutate(|inner| inner.push(ch));
    }

    #[inline]
    #[track_caller]
    pub fn push_str(&mut self, string: &str) {
        self.mutate(|inner| inner.push_str(string));
    }

    #[inline]
    #[track_caller]
    pub fn truncate(&mut self, new_len: usize) {
        self.mutate(|inner| inner.truncate(new_len));
    }

    #[inline]
    #[track_caller]
    pub fn reserve(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve(additional));
    }

    #[inline]
    #[track_caller]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve_exact(additional));
    }

    #[inline]
    #[track_caller]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), collections::TryReserveError> {
        let old_ptr = string_ptr(&self.inner);
        let result = self.inner.try_reserve(additional);
        self.sync_from_old_ptr(old_ptr);
        result
    }

    #[inline]
    #[track_caller]
    pub fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), collections::TryReserveError> {
        let old_ptr = string_ptr(&self.inner);
        let result = self.inner.try_reserve_exact(additional);
        self.sync_from_old_ptr(old_ptr);
        result
    }

    #[inline]
    #[track_caller]
    pub fn shrink_to_fit(&mut self) {
        self.mutate(inner::String::shrink_to_fit);
    }

    #[inline]
    pub fn into_boxed_str(mut self) -> Box<str> {
        remove_unique_allocation(string_ptr(&self.inner));
        Box::from_alloc_boxed_str(mem::take(&mut self.inner).into_boxed_str())
    }

    #[inline]
    pub fn into_bytes(mut self) -> Vec<u8> {
        remove_unique_allocation(string_ptr(&self.inner));
        from_alloc_vec(mem::take(&mut self.inner).into_bytes())
    }

    #[inline]
    pub fn into_alloc_string(mut self) -> inner::String {
        remove_unique_allocation(string_ptr(&self.inner));
        mem::take(&mut self.inner)
    }
}

#[cfg(feature = "memprof")]
impl Drop for String {
    fn drop(&mut self) {
        remove_unique_allocation(string_ptr(&self.inner));
    }
}

#[cfg(feature = "memprof")]
impl Default for String {
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memprof")]
impl Clone for String {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_string(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl From<inner::String> for String {
    #[track_caller]
    fn from(value: inner::String) -> Self {
        Self::from_alloc_string(value)
    }
}

#[cfg(feature = "memprof")]
impl From<String> for inner::String {
    fn from(value: String) -> Self {
        value.into_alloc_string()
    }
}

#[cfg(feature = "memprof")]
impl From<&str> for String {
    #[track_caller]
    fn from(value: &str) -> Self {
        Self::from_alloc_string(inner::String::from(value))
    }
}

#[cfg(feature = "memprof")]
impl From<&String> for inner::String {
    fn from(value: &String) -> Self {
        value.inner.clone()
    }
}

#[cfg(feature = "memprof")]
impl From<char> for String {
    #[track_caller]
    fn from(value: char) -> Self {
        Self::from_alloc_string(inner::String::from(value))
    }
}

#[cfg(feature = "memprof")]
impl core::ops::Deref for String {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

#[cfg(feature = "memprof")]
impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[cfg(feature = "memprof")]
impl AsRef<[u8]> for String {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[cfg(feature = "memprof")]
impl AsRef<std::path::Path> for String {
    fn as_ref(&self) -> &std::path::Path {
        std::path::Path::new(self.as_str())
    }
}

#[cfg(feature = "memprof")]
impl Borrow<str> for String {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[cfg(feature = "memprof")]
impl fmt::Write for String {
    #[track_caller]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let old_ptr = string_ptr(&self.inner);
        self.inner.push_str(s);
        self.sync_from_old_ptr(old_ptr);
        Ok(())
    }

    #[track_caller]
    fn write_char(&mut self, c: char) -> fmt::Result {
        let old_ptr = string_ptr(&self.inner);
        self.inner.push(c);
        self.sync_from_old_ptr(old_ptr);
        Ok(())
    }
}

#[cfg(feature = "memprof")]
impl Extend<char> for String {
    #[track_caller]
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        let old_ptr = string_ptr(&self.inner);
        self.inner.extend(iter);
        self.sync_from_old_ptr(old_ptr);
    }
}

#[cfg(feature = "memprof")]
impl<'a> Extend<&'a str> for String {
    #[track_caller]
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        let old_ptr = string_ptr(&self.inner);
        self.inner.extend(iter);
        self.sync_from_old_ptr(old_ptr);
    }
}

#[cfg(feature = "memprof")]
impl FromIterator<char> for String {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> Self {
        Self::from_alloc_string(inner::String::from_iter(iter))
    }
}

#[cfg(feature = "memprof")]
impl<'a> FromIterator<&'a str> for String {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        Self::from_alloc_string(inner::String::from_iter(iter))
    }
}

#[cfg(feature = "memprof")]
impl fmt::Debug for String {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl fmt::Display for String {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl PartialEq for String {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl PartialEq<&str> for String {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[cfg(feature = "memprof")]
impl PartialEq<str> for String {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

#[cfg(feature = "memprof")]
impl Eq for String {}

#[cfg(feature = "memprof")]
impl PartialOrd for String {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl Ord for String {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl Hash for String {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

#[cfg(feature = "memprof")]
/// A tracked reference-counted pointer.
///
/// Tracking is shared across all clones so one `Rc` allocation appears once in
/// the profiler until the last clone is dropped.
#[repr(transparent)]
pub struct Rc<T: ?Sized> {
    inner: inner::Rc<T>,
}

#[cfg(feature = "memprof")]
impl<T> Rc<T> {
    #[inline]
    #[track_caller]
    pub fn new(value: T) -> Self {
        Self::from_alloc_rc(inner::Rc::new(value))
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Rc<T> {
    #[inline]
    #[track_caller]
    fn from_alloc_rc_with_state(
        inner: inner::Rc<T>,
        descriptor: AllocationDescriptor,
        state: AllocationState,
    ) -> Self {
        retain_shared_allocation_with_stack(state.ptr, descriptor, state, capture_stack_id());
        Self { inner }
    }

    #[inline]
    #[track_caller]
    pub fn from_alloc_rc(inner: inner::Rc<T>) -> Self {
        let state = rc_state(&inner);
        Self::from_alloc_rc_with_state(
            inner,
            AllocationDescriptor::new("Rc", type_name::<T>()),
            state,
        )
    }

    #[inline]
    pub fn clone(this: &Self) -> Self {
        Self::from_alloc_rc(inner::Rc::clone(&this.inner))
    }

    #[inline]
    pub fn downgrade(this: &Self) -> rc::Weak<T> {
        inner::Rc::downgrade(&this.inner)
    }

    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        inner::Rc::get_mut(&mut this.inner)
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        inner::Rc::ptr_eq(&this.inner, &other.inner)
    }

    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        inner::Rc::as_ptr(&this.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Drop for Rc<T> {
    fn drop(&mut self) {
        release_shared_allocation(rc_ptr(&self.inner));
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Clone for Rc<T> {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_rc(inner::Rc::clone(&self.inner))
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Deref for Rc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        self.inner.as_ref()
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> fmt::Debug for Rc<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> PartialEq for Rc<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T: ?Sized> Eq for Rc<T> where T: Eq {}

#[cfg(feature = "memprof")]
impl<T: ?Sized> From<inner::Rc<T>> for Rc<T> {
    #[track_caller]
    fn from(value: inner::Rc<T>) -> Self {
        Self::from_alloc_rc(value)
    }
}

#[cfg(feature = "memprof")]
impl<T> From<inner::Vec<T>> for Rc<[T]> {
    #[track_caller]
    fn from(value: inner::Vec<T>) -> Self {
        let inner = inner::Rc::<[T]>::from(value);
        let state = rc_slice_state(&inner);
        Self::from_alloc_rc_with_state(
            inner,
            AllocationDescriptor::new("Rc", type_name::<[T]>()).with_element_type(type_name::<T>()),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl<T, const N: usize> From<[T; N]> for Rc<[T]> {
    #[track_caller]
    fn from(value: [T; N]) -> Self {
        let inner = inner::Rc::<[T]>::from(value);
        let state = rc_slice_state(&inner);
        Self::from_alloc_rc_with_state(
            inner,
            AllocationDescriptor::new("Rc", type_name::<[T]>()).with_element_type(type_name::<T>()),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> From<&[T]> for Rc<[T]> {
    #[track_caller]
    fn from(value: &[T]) -> Self {
        let inner = inner::Rc::<[T]>::from(value);
        let state = rc_slice_state(&inner);
        Self::from_alloc_rc_with_state(
            inner,
            AllocationDescriptor::new("Rc", type_name::<[T]>()).with_element_type(type_name::<T>()),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl From<&str> for Rc<str> {
    #[track_caller]
    fn from(value: &str) -> Self {
        let inner = inner::Rc::<str>::from(value);
        let state = rc_str_state(&inner);
        Self::from_alloc_rc_with_state(
            inner,
            AllocationDescriptor::new("Rc", type_name::<str>()).with_element_type("u8"),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
impl From<inner::String> for Rc<str> {
    #[track_caller]
    fn from(value: inner::String) -> Self {
        let inner = inner::Rc::<str>::from(value);
        let state = rc_str_state(&inner);
        Self::from_alloc_rc_with_state(
            inner,
            AllocationDescriptor::new("Rc", type_name::<str>()).with_element_type("u8"),
            state,
        )
    }
}

#[cfg(feature = "memprof")]
/// A tracked ordered map.
///
/// Heap usage is tracked from the actual B-tree node allocations made by the
/// standard library map implementation.
#[repr(transparent)]
pub struct BTreeMap<K, V> {
    inner: inner::BTreeMap<K, V>,
}

#[cfg(feature = "memprof")]
impl<K, V> BTreeMap<K, V> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: inner::BTreeMap::new(),
        }
    }

    #[inline]
    fn descriptor() -> AllocationDescriptor {
        AllocationDescriptor::new("BTreeMap", type_name::<(K, V)>())
    }

    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.inner.get_mut(key)
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.inner.remove(key)
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> BTreeMap<K, V> {
    #[inline]
    #[track_caller]
    pub fn from_alloc_btree_map(inner: inner::BTreeMap<K, V>) -> Self {
        if !tracking_enabled() || inner.is_empty() {
            return Self { inner };
        }
        let descriptor = Self::descriptor();
        let rebuilt = with_allocation_context(descriptor, || inner.into_iter().collect());
        Self { inner: rebuilt }
    }

    #[inline]
    pub fn into_alloc_btree_map(self) -> inner::BTreeMap<K, V> {
        let this = mem::ManuallyDrop::new(self);
        let inner = unsafe { core::ptr::read(&this.inner) };
        if !tracking_enabled() || inner.is_empty() {
            return inner;
        }
        with_tracking_internal(|| inner.into_iter().collect())
    }

    #[inline]
    pub fn into_values(self) -> alloc::collections::btree_map::IntoValues<K, V> {
        self.into_alloc_btree_map().into_values()
    }

    #[inline]
    pub fn into_keys(self) -> alloc::collections::btree_map::IntoKeys<K, V> {
        self.into_alloc_btree_map().into_keys()
    }

    #[inline]
    #[track_caller]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        with_allocation_context(Self::descriptor(), || self.inner.insert(key, value))
    }

    #[inline]
    #[track_caller]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[inline]
    #[track_caller]
    pub fn append(&mut self, other: &mut Self) {
        with_allocation_context(Self::descriptor(), || self.inner.append(&mut other.inner));
    }

    #[inline]
    #[track_caller]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let context = AllocationContext {
            descriptor: Self::descriptor(),
            create_stack: capture_stack_id(),
        };
        match self.inner.entry(key) {
            inner::BTreeMapEntry::Occupied(inner) => Entry::Occupied(OccupiedEntry { inner }),
            inner::BTreeMapEntry::Vacant(inner) => Entry::Vacant(VacantEntry { inner, context }),
        }
    }
}

#[cfg(feature = "memprof")]
impl<K, V> Default for BTreeMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memprof")]
impl<K: Clone + Ord, V: Clone> Clone for BTreeMap<K, V> {
    #[track_caller]
    fn clone(&self) -> Self {
        let descriptor = Self::descriptor();
        let inner = with_allocation_context(descriptor, || self.inner.clone());
        Self { inner }
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> From<inner::BTreeMap<K, V>> for BTreeMap<K, V> {
    #[track_caller]
    fn from(value: inner::BTreeMap<K, V>) -> Self {
        Self::from_alloc_btree_map(value)
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> From<BTreeMap<K, V>> for inner::BTreeMap<K, V> {
    fn from(value: BTreeMap<K, V>) -> Self {
        value.into_alloc_btree_map()
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V, const N: usize> From<[(K, V); N]> for BTreeMap<K, V> {
    #[track_caller]
    fn from(value: [(K, V); N]) -> Self {
        let descriptor = Self::descriptor();
        let inner = with_allocation_context(descriptor, || inner::BTreeMap::from(value));
        Self { inner }
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let descriptor = Self::descriptor();
        let inner = with_allocation_context(descriptor, || inner::BTreeMap::from_iter(iter));
        Self { inner }
    }
}

#[cfg(feature = "memprof")]
impl<K, V> Deref for BTreeMap<K, V> {
    type Target = inner::BTreeMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(feature = "memprof")]
impl<K, V> fmt::Debug for BTreeMap<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl<K, V> PartialEq for BTreeMap<K, V>
where
    K: PartialEq,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<K, V> Eq for BTreeMap<K, V>
where
    K: Eq,
    V: Eq,
{
}

#[cfg(feature = "memprof")]
impl<K, V> PartialOrd for BTreeMap<K, V>
where
    K: PartialOrd,
    V: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<K, V> Ord for BTreeMap<K, V>
where
    K: Ord,
    V: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> IntoIterator for BTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = inner::BTreeMapIntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_alloc_btree_map().into_iter()
    }
}

#[cfg(feature = "memprof")]
impl<'a, K, V> IntoIterator for &'a BTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = alloc::collections::btree_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

#[cfg(feature = "memprof")]
impl<'a, K, V> IntoIterator for &'a mut BTreeMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = alloc::collections::btree_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

#[cfg(feature = "memprof")]
pub enum Entry<'a, K: 'a, V: 'a> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

#[cfg(feature = "memprof")]
pub struct VacantEntry<'a, K, V> {
    inner: inner::BTreeMapVacantEntry<'a, K, V>,
    context: AllocationContext,
}

#[cfg(feature = "memprof")]
pub struct OccupiedEntry<'a, K, V> {
    inner: inner::BTreeMapOccupiedEntry<'a, K, V>,
}

#[cfg(feature = "memprof")]
impl<'a, K: Ord, V> Entry<'a, K, V> {
    pub fn key(&self) -> &K {
        match self {
            Self::Vacant(entry) => entry.key(),
            Self::Occupied(entry) => entry.key(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Self::Occupied(mut entry) => {
                f(entry.get_mut());
                Self::Occupied(entry)
            }
            Self::Vacant(entry) => Self::Vacant(entry),
        }
    }

    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(default()),
        }
    }

    pub fn or_insert_with_key<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }
}

#[cfg(feature = "memprof")]
impl<'a, K: Ord, V: Default> Entry<'a, K, V> {
    pub fn or_default(self) -> &'a mut V {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(V::default()),
        }
    }
}

#[cfg(feature = "memprof")]
impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    pub fn key(&self) -> &K {
        self.inner.key()
    }

    pub fn into_key(self) -> K {
        self.inner.into_key()
    }
}

#[cfg(feature = "memprof")]
impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        with_existing_allocation_context(self.context, || self.inner.insert(value))
    }
}

#[cfg(feature = "memprof")]
impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    pub fn key(&self) -> &K {
        self.inner.key()
    }

    pub fn get(&self) -> &V {
        self.inner.get()
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.inner.get_mut()
    }

    pub fn into_mut(self) -> &'a mut V {
        self.inner.into_mut()
    }
}

#[cfg(feature = "memprof")]
impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    pub fn insert(&mut self, value: V) -> V {
        self.inner.insert(value)
    }

    pub fn remove(self) -> V {
        self.inner.remove()
    }

    pub fn remove_entry(self) -> (K, V) {
        self.inner.remove_entry()
    }
}

#[cfg(feature = "memprof")]
/// A tracked ordered set.
///
/// Heap usage is tracked from the actual B-tree node allocations made by the
/// standard library set implementation.
#[repr(transparent)]
pub struct BTreeSet<T> {
    inner: inner::BTreeSet<T>,
}

#[cfg(feature = "memprof")]
impl<T> BTreeSet<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: inner::BTreeSet::new(),
        }
    }

    #[inline]
    fn descriptor() -> AllocationDescriptor {
        AllocationDescriptor::new("BTreeSet", type_name::<T>()).with_element_type(type_name::<T>())
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> BTreeSet<T> {
    #[inline]
    #[track_caller]
    pub fn from_alloc_btree_set(inner: inner::BTreeSet<T>) -> Self {
        if !tracking_enabled() || inner.is_empty() {
            return Self { inner };
        }
        let descriptor = Self::descriptor();
        let rebuilt = with_allocation_context(descriptor, || inner.into_iter().collect());
        Self { inner: rebuilt }
    }

    #[inline]
    pub fn into_alloc_btree_set(self) -> inner::BTreeSet<T> {
        let this = mem::ManuallyDrop::new(self);
        let inner = unsafe { core::ptr::read(&this.inner) };
        if !tracking_enabled() || inner.is_empty() {
            return inner;
        }
        with_tracking_internal(|| inner.into_iter().collect())
    }

    #[inline]
    #[track_caller]
    pub fn insert(&mut self, value: T) -> bool {
        with_allocation_context(Self::descriptor(), || self.inner.insert(value))
    }

    #[inline]
    #[track_caller]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[inline]
    #[track_caller]
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord,
    {
        self.inner.remove(value)
    }

    #[inline]
    #[track_caller]
    pub fn append(&mut self, other: &mut Self) {
        with_allocation_context(Self::descriptor(), || self.inner.append(&mut other.inner));
    }
}

#[cfg(feature = "memprof")]
impl<T> Default for BTreeSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone + Ord> Clone for BTreeSet<T> {
    #[track_caller]
    fn clone(&self) -> Self {
        let descriptor = Self::descriptor();
        let inner = with_allocation_context(descriptor, || self.inner.clone());
        Self { inner }
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> From<inner::BTreeSet<T>> for BTreeSet<T> {
    #[track_caller]
    fn from(value: inner::BTreeSet<T>) -> Self {
        Self::from_alloc_btree_set(value)
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> From<BTreeSet<T>> for inner::BTreeSet<T> {
    fn from(value: BTreeSet<T>) -> Self {
        value.into_alloc_btree_set()
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord, const N: usize> From<[T; N]> for BTreeSet<T> {
    #[track_caller]
    fn from(value: [T; N]) -> Self {
        let descriptor = Self::descriptor();
        let inner = with_allocation_context(descriptor, || inner::BTreeSet::from(value));
        Self { inner }
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> FromIterator<T> for BTreeSet<T> {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let descriptor = Self::descriptor();
        let inner = with_allocation_context(descriptor, || inner::BTreeSet::from_iter(iter));
        Self { inner }
    }
}

#[cfg(feature = "memprof")]
impl<T> Deref for BTreeSet<T> {
    type Target = inner::BTreeSet<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(feature = "memprof")]
impl<T> fmt::Debug for BTreeSet<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(feature = "memprof")]
impl<T> PartialEq for BTreeSet<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Eq for BTreeSet<T> where T: Eq {}

#[cfg(feature = "memprof")]
impl<T> PartialOrd for BTreeSet<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Ord for BTreeSet<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T> Hash for BTreeSet<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> IntoIterator for BTreeSet<T> {
    type Item = T;
    type IntoIter = inner::BTreeSetIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_alloc_btree_set().into_iter()
    }
}

#[cfg(feature = "memprof")]
impl<'a, T> IntoIterator for &'a BTreeSet<T> {
    type Item = &'a T;
    type IntoIter = alloc::collections::btree_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

#[cfg(all(test, feature = "memprof"))]
#[global_allocator]
static TEST_TRACKING_ALLOCATOR: TrackingAllocator<std::alloc::System> =
    TrackingAllocator::new(std::alloc::System);

// === Facade modules ==========================================================
//
// These modules expose the `tracked_alloc` surface in an `alloc`-shaped layout.
// Some items are tracked wrappers (`collections::BTreeMap`, `collections::BTreeSet`),
// while others intentionally remain plain `alloc` re-exports.

/// Tracked box surface.
pub mod boxed {
    pub use crate::Box;
}

/// Convert an `alloc` box into the tracked facade. Unlike the `From`
/// impls this accepts unsized targets (`dyn` closures), and it exists
/// under both cfgs so callers need no feature knowledge.
#[cfg(feature = "memprof")]
#[track_caller]
pub fn box_from_alloc<T: ?Sized>(inner: AllocBox<T>) -> Box<T> {
    Box::from_alloc_box_unsized(inner)
}

/// Convert an `alloc` box into the tracked facade (plain alias here).
#[cfg(not(feature = "memprof"))]
pub fn box_from_alloc<T: ?Sized>(inner: AllocBox<T>) -> Box<T> {
    inner
}

/// Mixed tracked/pass-through collection facade.
///
/// Tracked:
/// - [`BTreeMap`]
/// - [`BTreeSet`]
///
/// Pass-through:
/// - `BinaryHeap`
/// - `TryReserveError`
/// - `VecDeque`
pub mod collections {
    pub use alloc::collections::{BinaryHeap, TryReserveError, VecDeque};

    pub use crate::{BTreeMap, BTreeSet};
}

/// Tracked `Rc` surface plus pass-through `Weak`.
#[cfg(feature = "memprof")]
pub mod rc {
    pub use alloc::rc::Weak;

    pub use crate::Rc;
}

/// Pass-through `alloc::rc` re-exports.
#[cfg(not(feature = "memprof"))]
pub mod rc {
    pub use alloc::rc::{Rc, Weak};
}

/// Tracked string surface.
///
/// [`String`] is tracked directly from its backing buffer.
#[cfg(feature = "memprof")]
pub mod string {
    pub use crate::String;

    pub trait ToString {
        fn to_string(&self) -> String;
    }

    impl<T> ToString for T
    where
        T: core::fmt::Display + ?Sized,
    {
        fn to_string(&self) -> String {
            String::from(alloc::string::ToString::to_string(self))
        }
    }
}

/// Pass-through `alloc::string` re-exports.
#[cfg(not(feature = "memprof"))]
pub mod string {
    pub use alloc::string::{String, ToString};
}

/// Tracked vector surface.
///
/// [`Vec`] is tracked directly from its backing buffer. Other tracked
/// collections are available through [`collections`].
pub mod vec {
    pub use crate::inner::IntoIter;

    pub use crate::{from_alloc_vec, from_raw_parts, into_alloc_vec, into_raw_parts, Vec};
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "memprof")]
    use std::sync::{Mutex, OnceLock};

    #[cfg(feature = "memprof")]
    fn tracking_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn vec_macro_matches_standard_behavior() {
        #[cfg(feature = "memprof")]
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        #[cfg(feature = "memprof")]
        {
            set_tracking_enabled(false);
            reset_tracking();
        }

        {
            let values = vec![1u32, 2, 3];
            assert_eq!(values.len(), 3);
            assert_eq!(values[1], 2);
        }

        #[cfg(feature = "memprof")]
        reset_tracking();
    }

    #[test]
    fn from_alloc_vec_round_trip() {
        #[cfg(feature = "memprof")]
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        #[cfg(feature = "memprof")]
        {
            set_tracking_enabled(false);
            reset_tracking();
        }

        {
            let values = from_alloc_vec(alloc::vec![1u8, 2, 3]);
            let raw: inner::Vec<u8> = values.into();
            assert_eq!(raw, alloc::vec![1u8, 2, 3]);
        }

        #[cfg(feature = "memprof")]
        reset_tracking();
    }

    #[test]
    fn raw_parts_retype_transfers_tracking_and_drops_cleanly() {
        #[cfg(feature = "memprof")]
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        #[cfg(feature = "memprof")]
        {
            set_tracking_enabled(true);
            reset_tracking();
        }

        let mut values = Vec::<u32>::with_capacity(8);
        values.extend([1, 2, 3]);
        let allocation = values.as_ptr() as usize;
        #[cfg(feature = "memprof")]
        let before = {
            let live = snapshot();
            assert_eq!(live.records.len(), 1);
            assert_eq!(live.records[0].type_name, type_name::<u32>());
            assert_eq!(live.records[0].element_type, Some(type_name::<u32>()));
            assert_eq!(live.total_bytes, 8 * mem::size_of::<u32>());
            live.records[0].clone()
        };

        let (ptr, len, capacity) = into_raw_parts(values);
        assert_eq!(ptr as usize, allocation);
        assert_eq!((len, capacity), (3, 8));
        #[cfg(feature = "memprof")]
        {
            let live = snapshot();
            assert_eq!(live.records.len(), 1, "raw owner must stay tracked");
            assert_eq!(live.records[0].id, before.id);
        }

        // SAFETY: i32 and u32 have identical allocation layout/alignment and
        // every initialized u32 bit pattern is also a valid i32 value. The
        // raw allocation has no other owner between the paired calls.
        let values: Vec<i32> = unsafe { from_raw_parts(ptr.cast(), len, capacity) };
        assert_eq!(values.as_ptr() as usize, allocation);
        assert_eq!(values.as_slice(), &[1, 2, 3]);
        #[cfg(feature = "memprof")]
        {
            let live = snapshot();
            assert_eq!(live.records.len(), 1);
            assert_eq!(
                live.records[0].id, before.id,
                "retype is not a new allocation"
            );
            assert_eq!(live.records[0].ptr, before.ptr);
            assert_eq!(live.records[0].create_stack, before.create_stack);
            assert_eq!(live.records[0].type_name, type_name::<i32>());
            assert_eq!(live.records[0].element_type, Some(type_name::<i32>()));
            assert_eq!(live.records[0].len, Some(3));
            assert_eq!(live.records[0].capacity, Some(8));
            assert_eq!(live.total_bytes, before.size_bytes);
        }

        drop(values);
        #[cfg(feature = "memprof")]
        {
            let live = snapshot();
            assert!(live.records.is_empty(), "new owner must free exactly once");
            assert_eq!(live.total_bytes, 0);
            assert!(tracked_allocations().lock().unwrap().is_empty());
            set_tracking_enabled(false);
            reset_tracking();
        }
    }

    #[test]
    fn alloc_like_surface_is_available() {
        #[cfg(feature = "memprof")]
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        #[cfg(feature = "memprof")]
        set_tracking_enabled(false);
        #[cfg(feature = "memprof")]
        reset_tracking();

        let mut values = vec::Vec::new();
        values.push(1u32);
        let message = format!("values={}", values.len());
        let boxed = boxed::Box::new(message);
        let shared = rc::Rc::new(string::String::from(boxed.as_str()));
        let mut names = collections::BTreeSet::new();
        names.insert(shared.as_str());
        assert!(names.contains("values=1"));

        #[cfg(feature = "memprof")]
        {
            drop(names);
            drop(shared);
            drop(boxed);
            drop(values);
            assert!(snapshot().records.is_empty());
            set_tracking_enabled(false);
            reset_tracking();
        }
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn tracked_types_match_alloc_layout() {
        struct WithTrackedVec {
            _values: Vec<u16>,
        }

        struct WithAllocVec {
            _values: alloc::vec::Vec<u16>,
        }

        assert_eq!(
            mem::size_of::<Vec<u8>>(),
            mem::size_of::<alloc::vec::Vec<u8>>()
        );
        assert_eq!(
            mem::size_of::<String>(),
            mem::size_of::<alloc::string::String>()
        );
        assert_eq!(
            mem::size_of::<Box<u64>>(),
            mem::size_of::<alloc::boxed::Box<u64>>()
        );
        assert_eq!(
            mem::size_of::<Rc<u64>>(),
            mem::size_of::<alloc::rc::Rc<u64>>()
        );
        assert_eq!(
            mem::size_of::<BTreeMap<u8, u8>>(),
            mem::size_of::<alloc::collections::BTreeMap<u8, u8>>()
        );
        assert_eq!(
            mem::size_of::<BTreeSet<u8>>(),
            mem::size_of::<alloc::collections::BTreeSet<u8>>()
        );
        assert_eq!(
            mem::size_of::<WithTrackedVec>(),
            mem::size_of::<WithAllocVec>()
        );
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn tracking_records_lifecycle_and_clone() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();
        let mut values = Vec::<u32>::new();
        let snapshot0 = snapshot();
        assert!(snapshot0.records.is_empty());

        values.push(7);
        values.push(9);
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records.len(), 1);
        assert_eq!(snapshot1.records[0].len, Some(2));
        assert_eq!(snapshot1.records[0].owner_kind, "Vec");
        assert_eq!(snapshot1.records[0].type_name, type_name::<u32>());
        assert_eq!(snapshot1.records[0].element_type, Some(type_name::<u32>()));
        assert!(snapshot1.records[0].size_bytes >= 2 * mem::size_of::<u32>());
        assert!(!snapshot1.records[0].create_stack.is_empty());

        let cloned = values.clone();
        let snapshot2 = snapshot();
        assert_eq!(
            snapshot2.records.len(),
            2,
            "ids={:?}",
            snapshot2
                .records
                .iter()
                .map(|record| record.id)
                .collect::<inner::Vec<_>>()
        );
        assert!(snapshot2.total_bytes >= snapshot1.total_bytes);

        drop(cloned);
        drop(values);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn tracking_updates_after_drain_and_splice() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();
        let mut values = vec![1u8, 2, 3, 4];
        let drained: inner::Vec<u8> = values.drain(1..3).collect();
        assert_eq!(drained, alloc::vec![2u8, 3]);
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records.len(), 1);
        assert_eq!(snapshot1.records[0].len, Some(2));

        let removed: inner::Vec<u8> = values.splice(1..1, alloc::vec![9u8, 10]).collect();
        assert!(removed.is_empty());
        let snapshot2 = snapshot();
        assert_eq!(snapshot2.records[0].len, Some(4));
        assert_eq!(&values[..], &[1, 9, 10, 4]);
        drop(values);
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn len_only_vec_updates_do_not_change_aggregate_bytes() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut values = Vec::<u8>::with_capacity(8);
        let bytes_after_reserve = snapshot().total_bytes;
        values.push(1);
        values.push(2);
        values.push(3);
        assert_eq!(snapshot().total_bytes, bytes_after_reserve);
        assert_eq!(snapshot().records[0].len, Some(3));

        drop(values);
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn box_tracking_replaces_vec_owner_for_boxed_slice() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let values = vec![1u8, 2, 3, 4];
        let snapshot0 = snapshot();
        assert_eq!(snapshot0.records.len(), 1);
        assert_eq!(snapshot0.records[0].owner_kind, "Vec");

        let boxed = values.into_boxed_slice();
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records.len(), 1);
        assert_eq!(snapshot1.records[0].owner_kind, "Box");
        assert_eq!(snapshot1.records[0].type_name, type_name::<[u8]>());
        assert_eq!(snapshot1.records[0].element_type, Some(type_name::<u8>()));
        assert_eq!(snapshot1.records[0].len, Some(4));
        assert_eq!(snapshot1.records[0].capacity, Some(4));
        assert_eq!(&boxed[..], &[1, 2, 3, 4]);

        drop(boxed);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn string_tracking_records_buffer_and_format_output() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut text = String::from("hello");
        text.push('!');
        let formatted = format!("len={}", text.len());
        let snapshot0 = snapshot();
        assert_eq!(snapshot0.records.len(), 2);
        assert!(snapshot0
            .records
            .iter()
            .all(|record| record.owner_kind == "String"));
        assert!(snapshot0
            .records
            .iter()
            .any(|record| record.type_name == type_name::<str>() && record.len == Some(6)));
        assert!(snapshot0
            .records
            .iter()
            .all(|record| record.element_type == Some("u8")));

        drop(formatted);
        drop(text);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn rc_tracking_counts_shared_allocations_once() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let value = Rc::new(123u64);
        let snapshot0 = snapshot();
        assert_eq!(snapshot0.records.len(), 1);
        assert_eq!(snapshot0.records[0].owner_kind, "Rc");
        assert_eq!(snapshot0.records[0].type_name, type_name::<u64>());
        assert!(snapshot0.records[0].size_bytes >= rc_header_bytes() + mem::size_of::<u64>());

        let cloned = Rc::clone(&value);
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records.len(), 1);
        assert!(Rc::ptr_eq(&value, &cloned));

        let bytes = Rc::<[u8]>::from([1u8, 2, 3, 4]);
        let snapshot2 = snapshot();
        assert_eq!(snapshot2.records.len(), 2);
        assert!(snapshot2
            .records
            .iter()
            .any(|record| record.type_name == type_name::<[u8]>() && record.len == Some(4)));

        drop(bytes);
        drop(cloned);
        drop(value);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn rc_downgrade_upgrade_cycle_preserves_live_bytes() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let value = Rc::new(123u64);
        let live_bytes = snapshot().total_bytes;
        assert!(live_bytes > 0);

        let weak = Rc::downgrade(&value);
        assert_eq!(snapshot().total_bytes, live_bytes);

        let upgraded = Rc::from_alloc_rc(weak.upgrade().expect("strong Rc remains live"));
        assert_eq!(snapshot().total_bytes, live_bytes);

        drop(upgraded);
        assert_eq!(snapshot().total_bytes, live_bytes);

        drop(value);
        assert_eq!(snapshot().total_bytes, 0);
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn tracking_profile_reconstructs_runtime_allocations() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut code = AllocationHandle::new(
            AllocationDescriptor::new(RUNTIME_MEMORY_OWNER, RUNTIME_TYPE_CODE_BUFFER),
            AllocationState::new(64)
                .with_len(0)
                .with_capacity(64)
                .with_ptr(0x1000),
        );
        let mut guard = AllocationHandle::new(
            AllocationDescriptor::new(RUNTIME_MEMORY_OWNER, RUNTIME_TYPE_GUARD_PAGE),
            AllocationState::new(4096)
                .with_len(4096)
                .with_capacity(4096)
                .with_ptr(0x2000),
        );
        let profile0 = profile();
        assert_eq!(profile0.snapshot.records.len(), 2);
        // Each runtime type is routed to its own series counter.
        assert_eq!(profile0.snapshot.total_bytes, 0);
        assert_eq!(profile0.snapshot.code_buffer_bytes, 64);
        assert_eq!(profile0.snapshot.guard_page_bytes, 4096);

        code.update(
            AllocationState::new(128)
                .with_len(32)
                .with_capacity(128)
                .with_ptr(0x1000),
        );
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.total_bytes, 0);
        assert_eq!(snapshot1.code_buffer_bytes, 128);
        assert_eq!(snapshot1.guard_page_bytes, 4096);

        code.remove();
        guard.remove();
        let profile2 = profile();
        assert!(profile2.snapshot.records.is_empty());
        assert_eq!(profile2.snapshot.total_bytes, 0);
        assert_eq!(profile2.snapshot.code_buffer_bytes, 0);
        assert_eq!(profile2.snapshot.guard_page_bytes, 0);
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn btree_map_tracking_updates_for_insert_entry_and_remove() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut map = BTreeMap::<u32, u64>::new();
        let snapshot0 = snapshot();
        assert!(snapshot0.records.is_empty());

        map.insert(1, 10);
        map.entry(2)
            .and_modify(|value| *value += 1)
            .or_insert_with(|| 20);
        map.entry(2).and_modify(|value| *value += 1).or_insert(0);
        let snapshot1 = snapshot();
        assert!(!snapshot1.records.is_empty());
        assert!(snapshot1
            .records
            .iter()
            .all(|record| record.owner_kind == "BTreeMap"));
        assert!(snapshot1.records.iter().all(|record| record.size_bytes > 0));

        assert_eq!(map.remove(&1), Some(10));
        let snapshot2 = snapshot();
        assert!(!snapshot2.records.is_empty());
        assert!(snapshot2
            .records
            .iter()
            .all(|record| record.owner_kind == "BTreeMap"));

        drop(map);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn btree_set_tracking_updates_for_insert_and_remove() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut set = BTreeSet::from([1u32, 3, 5]);
        let snapshot0 = snapshot();
        assert!(!snapshot0.records.is_empty());
        assert!(snapshot0
            .records
            .iter()
            .all(|record| record.owner_kind == "BTreeSet"));
        assert!(snapshot0
            .records
            .iter()
            .all(|record| record.type_name == type_name::<u32>()));

        assert!(!set.insert(3));
        assert!(set.insert(7));
        assert!(set.remove(&1));
        let snapshot1 = snapshot();
        assert!(!snapshot1.records.is_empty());
        assert!(snapshot1
            .records
            .iter()
            .all(|record| record.owner_kind == "BTreeSet"));
        assert!(snapshot1.records.iter().all(|record| record.size_bytes > 0));

        drop(set);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }

    #[cfg(feature = "memprof")]
    #[test]
    fn tracking_stays_dormant_until_enabled() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(false);
        reset_tracking();

        let mut dormant = Vec::<u32>::new();
        dormant.push(1);
        assert!(snapshot().records.is_empty());

        set_tracking_enabled(true);
        let mut tracked = Vec::<u32>::new();
        tracked.push(7);
        let tracked_snapshot = snapshot();
        assert_eq!(tracked_snapshot.records.len(), 1);
        assert_eq!(tracked_snapshot.records[0].len, Some(1));
        assert_eq!(tracked_snapshot.records[0].type_name, type_name::<u32>());

        drop(tracked);
        drop(dormant);
        assert!(snapshot().records.is_empty());
        set_tracking_enabled(false);
        reset_tracking();
    }
}
