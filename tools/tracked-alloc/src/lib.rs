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
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
#[cfg(feature = "memprof")]
use std::collections::HashMap as StdHashMap;
#[cfg(feature = "memprof")]
use std::sync::{Mutex, OnceLock};

mod inner {
    pub use alloc::collections::btree_map::{
        Entry as BTreeMapEntry, IntoIter as BTreeMapIntoIter,
        OccupiedEntry as BTreeMapOccupiedEntry, VacantEntry as BTreeMapVacantEntry,
    };
    pub use alloc::collections::btree_set::IntoIter as BTreeSetIntoIter;
    pub type BTreeMap<K, V> = alloc::collections::BTreeMap<K, V>;
    pub type BTreeSet<T> = alloc::collections::BTreeSet<T>;
    pub type Rc<T> = alloc::rc::Rc<T>;
    pub type String = alloc::string::String;
    pub use alloc::vec::{Drain, IntoIter, Splice};
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

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalTimelinePoint {
    pub time_ns: u64,
    pub live_bytes: usize,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GlobalAllocationProfile {
    pub now_ns: u64,
    pub peak_bytes: usize,
    pub final_bytes: usize,
    pub timeline: inner::Vec<GlobalTimelinePoint>,
}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn reset_global_tracking() {}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn set_global_tracking_enabled(_enabled: bool) {}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn global_tracking_enabled() -> bool {
    false
}

#[cfg(not(feature = "memprof"))]
#[inline]
pub fn global_profile() -> GlobalAllocationProfile {
    GlobalAllocationProfile::default()
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
            record_global_delta(layout.size() as isize);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: core::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        #[cfg(feature = "memprof")]
        if !ptr.is_null() {
            record_global_delta(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        unsafe { self.inner.dealloc(ptr, layout) };
        #[cfg(feature = "memprof")]
        record_global_delta(-(layout.size() as isize));
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
            let delta = new_size as isize - layout.size() as isize;
            if delta != 0 {
                record_global_delta(delta);
            }
        }
        new_ptr
    }
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub records: inner::Vec<RecordSnapshot>,
    pub total_bytes: usize,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileEventKind {
    Create,
    Update,
    Remove,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileEvent {
    pub time_ns: u64,
    pub total_bytes: usize,
    pub live_records: usize,
    pub kind: ProfileEventKind,
    pub id: u64,
    pub owner_kind: Option<&'static str>,
    pub type_name: Option<&'static str>,
    pub element_type: Option<&'static str>,
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: Option<usize>,
    pub ptr: Option<usize>,
    pub create_stack_id: Option<u64>,
    pub last_update_stack_id: Option<u64>,
}

#[cfg(not(feature = "memprof"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimelinePoint {
    pub time_ns: u64,
    pub total_bytes: usize,
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
    pub events: inner::Vec<ProfileEvent>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileEventKind {
    Create,
    Update,
    Remove,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileEvent {
    pub time_ns: u64,
    pub total_bytes: usize,
    pub live_records: usize,
    pub kind: ProfileEventKind,
    pub id: u64,
    pub owner_kind: Option<&'static str>,
    pub type_name: Option<&'static str>,
    pub element_type: Option<&'static str>,
    pub len: Option<usize>,
    pub capacity: Option<usize>,
    pub size_bytes: Option<usize>,
    pub ptr: Option<usize>,
    pub create_stack_id: Option<u64>,
    pub last_update_stack_id: Option<u64>,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimelinePoint {
    pub time_ns: u64,
    pub total_bytes: usize,
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
    pub events: inner::Vec<ProfileEvent>,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalTimelinePoint {
    pub time_ns: u64,
    pub live_bytes: usize,
}

#[cfg(feature = "memprof")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GlobalAllocationProfile {
    pub now_ns: u64,
    pub peak_bytes: usize,
    pub final_bytes: usize,
    pub timeline: inner::Vec<GlobalTimelinePoint>,
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
#[derive(Clone, Debug, PartialEq, Eq)]
struct EventRecord {
    time_ns: u64,
    total_bytes: usize,
    live_records: usize,
    kind: ProfileEventKind,
    id: u64,
    owner_kind: Option<&'static str>,
    type_name: Option<&'static str>,
    element_type: Option<&'static str>,
    len: Option<usize>,
    capacity: Option<usize>,
    size_bytes: Option<usize>,
    ptr: Option<usize>,
    create_stack: Option<StackId>,
    last_update_stack: Option<StackId>,
}

#[cfg(feature = "memprof")]
struct ProfilerState {
    start: std::time::Instant,
    next_stack_id: StackId,
    total_bytes: usize,
    stacks: StdHashMap<StackId, StoredStack>,
    stack_ids_by_text: StdHashMap<AllocBox<str>, StackId>,
    records: StdHashMap<u64, Record>,
    timeline: inner::Vec<TimelinePoint>,
    phases: inner::Vec<ProfilePhase>,
    events: inner::Vec<EventRecord>,
}

#[cfg(feature = "memprof")]
struct GlobalTrackerState {
    start: std::time::Instant,
    current_bytes: usize,
    peak_bytes: usize,
    sample_stride: u64,
    since_sample: u64,
    timeline: inner::Vec<GlobalTimelinePoint>,
}

#[cfg(feature = "memprof")]
impl Default for ProfilerState {
    fn default() -> Self {
        Self {
            start: std::time::Instant::now(),
            next_stack_id: 1,
            total_bytes: 0,
            stacks: StdHashMap::new(),
            stack_ids_by_text: StdHashMap::new(),
            records: StdHashMap::new(),
            timeline: inner::Vec::from([TimelinePoint::default()]),
            phases: inner::Vec::new(),
            events: inner::Vec::new(),
        }
    }
}

#[cfg(feature = "memprof")]
impl Default for GlobalTrackerState {
    fn default() -> Self {
        Self {
            start: std::time::Instant::now(),
            current_bytes: 0,
            peak_bytes: 0,
            sample_stride: 64,
            since_sample: 0,
            timeline: inner::Vec::from([GlobalTimelinePoint::default()]),
        }
    }
}

#[cfg(feature = "memprof")]
impl GlobalTrackerState {
    fn reset(&mut self) {
        self.start = std::time::Instant::now();
        self.current_bytes = 0;
        self.peak_bytes = 0;
        self.sample_stride = 64;
        self.since_sample = 0;
        self.timeline.clear();
        self.timeline.push(GlobalTimelinePoint::default());
    }
}

#[cfg(feature = "memprof")]
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
#[cfg(feature = "memprof")]
static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "memprof")]
static GLOBAL_TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "memprof")]
static PROFILER: OnceLock<Mutex<ProfilerState>> = OnceLock::new();
#[cfg(feature = "memprof")]
static GLOBAL_TRACKER: OnceLock<Mutex<GlobalTrackerState>> = OnceLock::new();

#[cfg(feature = "memprof")]
std::thread_local! {
    static GLOBAL_TRACKING_REENTRANT: Cell<bool> = const { Cell::new(false) };
}

#[cfg(feature = "memprof")]
fn profiler() -> &'static Mutex<ProfilerState> {
    PROFILER.get_or_init(|| Mutex::new(ProfilerState::default()))
}

#[cfg(feature = "memprof")]
fn global_tracker() -> &'static Mutex<GlobalTrackerState> {
    GLOBAL_TRACKER.get_or_init(|| Mutex::new(GlobalTrackerState::default()))
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
#[inline]
pub fn set_global_tracking_enabled(enabled: bool) {
    GLOBAL_TRACKING_ENABLED.store(enabled, AtomicOrdering::Relaxed);
}

#[cfg(feature = "memprof")]
#[inline]
pub fn global_tracking_enabled() -> bool {
    GLOBAL_TRACKING_ENABLED.load(AtomicOrdering::Relaxed)
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
fn intern_stack(profiler: &mut ProfilerState, text: AllocBox<str>) -> StackId {
    if let Some(&id) = profiler.stack_ids_by_text.get(text.as_ref()) {
        return id;
    }
    let id = profiler.next_stack_id;
    profiler.next_stack_id = profiler.next_stack_id.saturating_add(1);
    profiler.stack_ids_by_text.insert(text.clone(), id);
    profiler.stacks.insert(id, StoredStack::new(text));
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
fn compact_global_timeline(global: &mut GlobalTrackerState) {
    if global.timeline.len() <= 32_768 {
        return;
    }
    let last_index = global.timeline.len().saturating_sub(1);
    let mut compacted = inner::Vec::with_capacity(global.timeline.len() / 2 + 2);
    for (index, point) in global.timeline.iter().copied().enumerate() {
        if index == 0 || index == last_index || index % 2 == 0 {
            compacted.push(point);
        }
    }
    global.timeline = compacted;
    global.sample_stride = global.sample_stride.saturating_mul(2);
    global.since_sample = 0;
}

#[cfg(feature = "memprof")]
fn push_global_sample(global: &mut GlobalTrackerState, force: bool) {
    let point = GlobalTimelinePoint {
        time_ns: elapsed_ns(global.start),
        live_bytes: global.current_bytes,
    };
    let changed = global
        .timeline
        .last()
        .map(|last| last.live_bytes != point.live_bytes || last.time_ns != point.time_ns)
        .unwrap_or(true);
    if force || changed {
        global.timeline.push(point);
        compact_global_timeline(global);
    }
    global.since_sample = 0;
}

#[cfg(feature = "memprof")]
fn record_global_delta(delta_bytes: isize) {
    if !global_tracking_enabled() {
        return;
    }
    GLOBAL_TRACKING_REENTRANT.with(|flag| {
        if flag.get() {
            return;
        }
        flag.set(true);
        {
            let mut global = global_tracker()
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if delta_bytes >= 0 {
                global.current_bytes = global.current_bytes.saturating_add(delta_bytes as usize);
            } else {
                global.current_bytes = global
                    .current_bytes
                    .saturating_sub(delta_bytes.unsigned_abs());
            }
            let hit_peak = global.current_bytes > global.peak_bytes;
            if hit_peak {
                global.peak_bytes = global.current_bytes;
            }
            global.since_sample = global.since_sample.saturating_add(1);
            if hit_peak || global.current_bytes == 0 || global.since_sample >= global.sample_stride
            {
                push_global_sample(&mut global, hit_peak);
            }
        }
        flag.set(false);
    });
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
    let total_bytes = records.iter().map(|record| record.size_bytes).sum();
    RegistrySnapshot {
        records,
        total_bytes,
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
    #[track_caller]
    fn capture_stack_id() -> Option<StackId> {
        if !tracking_enabled() {
            return None;
        }
        let create_stack = capture_site();
        let mut profiler = profiler().lock().unwrap();
        Some(intern_stack(&mut profiler, create_stack))
    }

    fn materialize(&mut self, state: AllocationState) {
        if self.id.is_some() || !tracking_enabled() || state.size_bytes == 0 {
            return;
        }
        let id = NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let create_stack = self.create_stack.unwrap_or_else(|| {
            let create_stack = capture_site();
            let mut profiler = profiler().lock().unwrap();
            intern_stack(&mut profiler, create_stack)
        });
        let mut profiler = profiler().lock().unwrap();
        profiler.total_bytes = profiler.total_bytes.saturating_add(state.size_bytes);
        profiler.records.insert(
            id,
            Record {
                owner_kind: self.descriptor.owner_kind,
                type_name: self.descriptor.type_name,
                element_type: self.descriptor.element_type,
                len: state.len,
                capacity: state.capacity,
                size_bytes: state.size_bytes,
                ptr: state.ptr,
                create_stack,
                last_update_stack: None,
            },
        );
        let time_ns = elapsed_ns(profiler.start);
        let live_records = profiler.records.len();
        let total_bytes = profiler.total_bytes;
        profiler.timeline.push(TimelinePoint {
            time_ns,
            total_bytes,
            live_records,
        });
        profiler.events.push(EventRecord {
            time_ns,
            total_bytes,
            live_records,
            kind: ProfileEventKind::Create,
            id,
            owner_kind: Some(self.descriptor.owner_kind),
            type_name: Some(self.descriptor.type_name),
            element_type: self.descriptor.element_type,
            len: state.len,
            capacity: state.capacity,
            size_bytes: Some(state.size_bytes),
            ptr: Some(state.ptr),
            create_stack: Some(create_stack),
            last_update_stack: None,
        });
        self.id = Some(id);
    }

    #[track_caller]
    pub fn new(descriptor: AllocationDescriptor, state: AllocationState) -> Self {
        let mut handle = Self {
            id: None,
            descriptor,
            create_stack: Self::capture_stack_id(),
        };
        handle.materialize(state);
        handle
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
        let mut profiler = profiler().lock().unwrap();
        let total_before = profiler.total_bytes;
        let Some(record) = profiler.records.get(&id) else {
            return;
        };
        let size_changed = record.capacity != state.capacity
            || record.size_bytes != state.size_bytes
            || record.ptr != state.ptr;
        let old_size_bytes = record.size_bytes;
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

        let total_bytes = total_before
            .saturating_sub(old_size_bytes)
            .saturating_add(state.size_bytes);
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
        let live_records = profiler.records.len();
        profiler.total_bytes = total_bytes;
        let time_ns = elapsed_ns(profiler.start);
        profiler.timeline.push(TimelinePoint {
            time_ns,
            total_bytes,
            live_records,
        });
        profiler.events.push(EventRecord {
            time_ns,
            total_bytes,
            live_records,
            kind: ProfileEventKind::Update,
            id,
            owner_kind: None,
            type_name: None,
            element_type: None,
            len: state.len,
            capacity: state.capacity,
            size_bytes: Some(state.size_bytes),
            ptr: Some(state.ptr),
            create_stack: None,
            last_update_stack: None,
        });
    }

    pub fn remove(&mut self) {
        let Some(id) = self.id.take() else {
            return;
        };
        let mut profiler = profiler().lock().unwrap();
        let Some(record) = profiler.records.remove(&id) else {
            return;
        };
        profiler.total_bytes = profiler.total_bytes.saturating_sub(record.size_bytes);
        let time_ns = elapsed_ns(profiler.start);
        let live_records = profiler.records.len();
        let total_bytes = profiler.total_bytes;
        profiler.timeline.push(TimelinePoint {
            time_ns,
            total_bytes,
            live_records,
        });
        profiler.events.push(EventRecord {
            time_ns,
            total_bytes,
            live_records,
            kind: ProfileEventKind::Remove,
            id,
            owner_kind: None,
            type_name: None,
            element_type: None,
            len: None,
            capacity: None,
            size_bytes: None,
            ptr: None,
            create_stack: None,
            last_update_stack: None,
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
#[derive(Debug)]
struct SharedAllocationHandle {
    ptr: usize,
}

#[cfg(feature = "memprof")]
impl SharedAllocationHandle {
    #[track_caller]
    fn new(ptr: usize, descriptor: AllocationDescriptor, state: AllocationState) -> Self {
        if ptr == 0 || !tracking_enabled() {
            return Self { ptr: 0 };
        }
        let mut shared = shared_allocations().lock().unwrap();
        if let Some(record) = shared.get_mut(&ptr) {
            record.refs = record.refs.saturating_add(1);
            return Self { ptr };
        }
        let trace = AllocationHandle::new(descriptor, state);
        shared.insert(ptr, SharedAllocationRecord { refs: 1, trace });
        Self { ptr }
    }

    fn remove(&mut self) {
        let ptr = mem::take(&mut self.ptr);
        if ptr == 0 {
            return;
        }
        let mut shared = shared_allocations().lock().unwrap();
        let Some(mut record) = shared.remove(&ptr) else {
            return;
        };
        if record.refs > 1 {
            record.refs -= 1;
            shared.insert(ptr, record);
            return;
        }
        record.trace.remove();
    }
}

#[cfg(feature = "memprof")]
impl Drop for SharedAllocationHandle {
    fn drop(&mut self) {
        self.remove();
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
        events: profiler
            .events
            .iter()
            .map(|event| ProfileEvent {
                time_ns: event.time_ns,
                total_bytes: event.total_bytes,
                live_records: event.live_records,
                kind: event.kind,
                id: event.id,
                owner_kind: event.owner_kind,
                type_name: event.type_name,
                element_type: event.element_type,
                len: event.len,
                capacity: event.capacity,
                size_bytes: event.size_bytes,
                ptr: event.ptr,
                create_stack_id: event.create_stack,
                last_update_stack_id: event.last_update_stack,
            })
            .collect(),
    }
}

#[cfg(feature = "memprof")]
pub fn global_profile() -> GlobalAllocationProfile {
    let global = global_tracker()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    GlobalAllocationProfile {
        now_ns: elapsed_ns(global.start),
        peak_bytes: global.peak_bytes,
        final_bytes: global.current_bytes,
        timeline: global.timeline.clone(),
    }
}

#[cfg(feature = "memprof")]
pub fn snapshot_at(time_ns: u64) -> RegistrySnapshot {
    let profiler = profiler().lock().unwrap();
    let mut live = StdHashMap::<u64, RecordSnapshot>::new();
    for event in &profiler.events {
        if event.time_ns > time_ns {
            break;
        }
        match event.kind {
            ProfileEventKind::Create => {
                let record = RecordSnapshot {
                    id: event.id,
                    owner_kind: event.owner_kind.expect("create owner kind"),
                    type_name: event.type_name.expect("create type name"),
                    element_type: event.element_type,
                    len: event.len,
                    capacity: event.capacity,
                    size_bytes: event.size_bytes.expect("create size"),
                    ptr: event.ptr.expect("create ptr"),
                    create_stack: render_stack(
                        &profiler,
                        event.create_stack.expect("create stack present"),
                    ),
                    last_update_stack: None,
                };
                live.insert(event.id, record);
            }
            ProfileEventKind::Update => {
                let Some(record) = live.get_mut(&event.id) else {
                    continue;
                };
                record.len = event.len;
                record.capacity = event.capacity;
                record.size_bytes = event.size_bytes.expect("update size");
                record.ptr = event.ptr.expect("update ptr");
                if let Some(stack_id) = event.last_update_stack {
                    record.last_update_stack = Some(render_stack(&profiler, stack_id));
                }
            }
            ProfileEventKind::Remove => {
                live.remove(&event.id);
            }
        }
    }
    let mut records: inner::Vec<RecordSnapshot> = live.into_values().collect();
    sort_records(&mut records);
    let total_bytes = records.iter().map(|record| record.size_bytes).sum();
    RegistrySnapshot {
        records,
        total_bytes,
    }
}

#[cfg(feature = "memprof")]
pub fn reset_tracking() {
    let mut profiler = profiler().lock().unwrap();
    profiler.start = std::time::Instant::now();
    profiler.next_stack_id = 1;
    profiler.total_bytes = 0;
    profiler.stacks.clear();
    profiler.stack_ids_by_text.clear();
    profiler.records.clear();
    profiler.timeline.clear();
    profiler.timeline.push(TimelinePoint::default());
    profiler.phases.clear();
    profiler.events.clear();
    NEXT_ID.store(1, AtomicOrdering::Relaxed);
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
pub fn reset_global_tracking() {
    let mut global = global_tracker()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    global.reset();
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
const BTREE_B: usize = 6;
#[cfg(feature = "memprof")]
const BTREE_CAPACITY: usize = 2 * BTREE_B - 1;
#[cfg(feature = "memprof")]
const BTREE_MAX_CHILDREN: usize = 2 * BTREE_B;

#[cfg(feature = "memprof")]
#[allow(dead_code)]
struct BTreeLeafLayout<K, V> {
    parent: Option<core::ptr::NonNull<u8>>,
    parent_idx: core::mem::MaybeUninit<u16>,
    len: u16,
    keys: [core::mem::MaybeUninit<K>; BTREE_CAPACITY],
    vals: [core::mem::MaybeUninit<V>; BTREE_CAPACITY],
}

#[cfg(feature = "memprof")]
#[repr(C)]
struct BTreeInternalLayout<K, V> {
    data: BTreeLeafLayout<K, V>,
    edges: [core::mem::MaybeUninit<core::ptr::NonNull<u8>>; BTREE_MAX_CHILDREN],
}

#[cfg(feature = "memprof")]
fn btree_node_counts(len: usize) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let mut leaf_nodes = len.div_ceil(BTREE_CAPACITY);
    let mut internal_nodes = 0usize;
    while leaf_nodes > 1 {
        leaf_nodes = leaf_nodes.div_ceil(BTREE_MAX_CHILDREN);
        internal_nodes = internal_nodes.saturating_add(leaf_nodes);
    }
    (len.div_ceil(BTREE_CAPACITY), internal_nodes)
}

#[cfg(feature = "memprof")]
fn estimate_btree_map_bytes<K, V>(len: usize) -> usize {
    let (leaf_nodes, internal_nodes) = btree_node_counts(len);
    leaf_nodes
        .saturating_mul(mem::size_of::<BTreeLeafLayout<K, V>>())
        .saturating_add(internal_nodes.saturating_mul(mem::size_of::<BTreeInternalLayout<K, V>>()))
}

#[cfg(feature = "memprof")]
fn btree_map_state<K, V>(len: usize) -> AllocationState {
    AllocationState::new(estimate_btree_map_bytes::<K, V>(len)).with_len(len)
}

#[cfg(feature = "memprof")]
fn btree_set_state<T>(len: usize) -> AllocationState {
    AllocationState::new(estimate_btree_map_bytes::<T, ()>(len)).with_len(len)
}

#[cfg(feature = "memprof")]
unsafe fn sync_btree_map_trace<K, V>(trace: *mut AllocationHandle, len: usize) {
    unsafe {
        (*trace).update(btree_map_state::<K, V>(len));
    }
}

#[cfg(feature = "memprof")]
/// A tracked box.
///
/// This records heap usage for uniquely-owned box allocations. For
/// `Box<[T]>` and `Box<str>`, length is reported directly from the boxed data.
pub struct Box<T: ?Sized> {
    inner: AllocBox<T>,
    trace: AllocationHandle,
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
        let trace = AllocationHandle::new(descriptor, state);
        Self { inner, trace }
    }

    #[inline]
    pub fn into_alloc_box(self) -> AllocBox<T> {
        let mut this = mem::ManuallyDrop::new(self);
        this.trace.remove();
        unsafe { core::ptr::read(&this.inner) }
    }

    #[inline]
    pub fn leak<'a>(value: Self) -> &'a mut T
    where
        T: 'a,
    {
        let this = mem::ManuallyDrop::new(value);
        let inner = unsafe { core::ptr::read(&this.inner) };
        let trace = unsafe { core::ptr::read(&this.trace) };
        mem::forget(trace);
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
        self.trace.remove();
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
pub struct Vec<T> {
    inner: inner::Vec<T>,
    trace: AllocationHandle,
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
        let trace = AllocationHandle::new(
            AllocationDescriptor::new("Vec", type_name::<T>()).with_element_type(type_name::<T>()),
            vec_state(&inner),
        );
        Self { inner, trace }
    }

    #[inline]
    fn sync(&mut self) {
        self.trace.update(vec_state(&self.inner));
    }

    #[inline]
    fn mutate<R>(&mut self, f: impl FnOnce(&mut inner::Vec<T>) -> R) -> R {
        let result = f(&mut self.inner);
        self.sync();
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
    pub fn push(&mut self, value: T) {
        self.mutate(|inner| inner.push(value));
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.mutate(|inner| inner.pop())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.mutate(inner::Vec::clear);
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.mutate(|inner| inner.truncate(len));
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve(additional));
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve_exact(additional));
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), collections::TryReserveError> {
        let result = self.inner.try_reserve(additional);
        self.sync();
        result
    }

    #[inline]
    pub fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), collections::TryReserveError> {
        let result = self.inner.try_reserve_exact(additional);
        self.sync();
        result
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.mutate(inner::Vec::shrink_to_fit);
    }

    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        self.mutate(|inner| inner.insert(index, element));
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        self.mutate(|inner| inner.remove(index))
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        self.mutate(|inner| inner.swap_remove(index))
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.inner.append(&mut other.inner);
        self.sync();
        other.sync();
    }

    #[inline]
    pub fn retain(&mut self, f: impl FnMut(&T) -> bool) {
        self.mutate(|inner| inner.retain(f));
    }

    #[inline]
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.mutate(inner::Vec::dedup);
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.mutate(|inner| inner.resize(new_len, value));
    }

    #[inline]
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        self.mutate(|inner| inner.resize_with(new_len, f));
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.mutate(|inner| inner.extend_from_slice(other));
    }

    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.mutate(|inner| inner.as_mut_slice().sort());
    }

    #[inline]
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.mutate(|inner| inner.sort_by(compare));
    }

    #[inline]
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
        let inner = self.inner.drain(range);
        Drain {
            inner: Some(inner),
            owner,
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
        let inner = self.inner.splice(range, replace_with);
        Splice {
            inner: Some(inner),
            owner,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        self.trace.remove();
        Box::from_alloc_boxed_slice(mem::take(&mut self.inner).into_boxed_slice())
    }

    #[inline]
    pub fn into_alloc_vec(mut self) -> inner::Vec<T> {
        self.trace.remove();
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

#[cfg(feature = "memprof")]
pub struct Drain<'a, T> {
    inner: Option<inner::Drain<'a, T>>,
    owner: *mut Vec<T>,
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
            (*self.owner).sync();
        }
    }
}

#[cfg(feature = "memprof")]
pub struct Splice<'a, T, I: Iterator<Item = T>> {
    inner: Option<inner::Splice<'a, I>>,
    owner: *mut Vec<T>,
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
            (*self.owner).sync();
        }
    }
}

#[cfg(feature = "memprof")]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        self.trace.remove();
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
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
        self.sync();
    }
}

#[cfg(feature = "memprof")]
impl<'a, T: 'a + Clone> Extend<&'a T> for Vec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.inner.extend(iter.into_iter().cloned());
        self.sync();
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
pub struct String {
    inner: inner::String,
    trace: AllocationHandle,
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
        let trace = AllocationHandle::new(
            AllocationDescriptor::new("String", type_name::<str>()).with_element_type("u8"),
            string_state(&inner),
        );
        Self { inner, trace }
    }

    #[inline]
    fn sync(&mut self) {
        self.trace.update(string_state(&self.inner));
    }

    #[inline]
    fn mutate<R>(&mut self, f: impl FnOnce(&mut inner::String) -> R) -> R {
        let result = f(&mut self.inner);
        self.sync();
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
    pub fn clear(&mut self) {
        self.mutate(inner::String::clear);
    }

    #[inline]
    pub fn push(&mut self, ch: char) {
        self.mutate(|inner| inner.push(ch));
    }

    #[inline]
    pub fn push_str(&mut self, string: &str) {
        self.mutate(|inner| inner.push_str(string));
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.mutate(|inner| inner.truncate(new_len));
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve(additional));
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.mutate(|inner| inner.reserve_exact(additional));
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), collections::TryReserveError> {
        let result = self.inner.try_reserve(additional);
        self.sync();
        result
    }

    #[inline]
    pub fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), collections::TryReserveError> {
        let result = self.inner.try_reserve_exact(additional);
        self.sync();
        result
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.mutate(inner::String::shrink_to_fit);
    }

    #[inline]
    pub fn into_boxed_str(mut self) -> Box<str> {
        self.trace.remove();
        Box::from_alloc_boxed_str(mem::take(&mut self.inner).into_boxed_str())
    }

    #[inline]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.trace.remove();
        from_alloc_vec(mem::take(&mut self.inner).into_bytes())
    }

    #[inline]
    pub fn into_alloc_string(mut self) -> inner::String {
        self.trace.remove();
        mem::take(&mut self.inner)
    }
}

#[cfg(feature = "memprof")]
impl Drop for String {
    fn drop(&mut self) {
        self.trace.remove();
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
impl Borrow<str> for String {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[cfg(feature = "memprof")]
impl fmt::Write for String {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.inner.push_str(s);
        self.sync();
        Ok(())
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        self.inner.push(c);
        self.sync();
        Ok(())
    }
}

#[cfg(feature = "memprof")]
impl Extend<char> for String {
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        self.inner.extend(iter);
        self.sync();
    }
}

#[cfg(feature = "memprof")]
impl<'a> Extend<&'a str> for String {
    fn extend<I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        self.inner.extend(iter);
        self.sync();
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
pub struct Rc<T: ?Sized> {
    inner: inner::Rc<T>,
    trace: SharedAllocationHandle,
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
        let trace = SharedAllocationHandle::new(state.ptr, descriptor, state);
        Self { inner, trace }
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
        self.trace.remove();
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
/// Heap usage is estimated from the standard library B-tree node layout and
/// current `len()`, because stable `alloc::collections::BTreeMap` does not
/// expose its internal allocation size.
pub struct BTreeMap<K, V> {
    inner: inner::BTreeMap<K, V>,
    trace: AllocationHandle,
}

#[cfg(feature = "memprof")]
impl<K, V> BTreeMap<K, V> {
    #[inline]
    #[track_caller]
    pub fn new() -> Self {
        Self::from_alloc_btree_map(inner::BTreeMap::new())
    }

    #[inline]
    #[track_caller]
    pub fn from_alloc_btree_map(inner: inner::BTreeMap<K, V>) -> Self {
        let trace = AllocationHandle::new(
            AllocationDescriptor::new("BTreeMap", type_name::<(K, V)>()),
            btree_map_state::<K, V>(inner.len()),
        );
        Self { inner, trace }
    }

    #[inline]
    fn sync(&mut self) {
        self.trace.update(btree_map_state::<K, V>(self.inner.len()));
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
        let removed = self.inner.remove(key);
        if removed.is_some() {
            self.sync();
        }
        removed
    }

    #[inline]
    pub fn into_alloc_btree_map(mut self) -> inner::BTreeMap<K, V> {
        self.trace.remove();
        mem::take(&mut self.inner)
    }

    #[inline]
    pub fn into_values(self) -> alloc::collections::btree_map::IntoValues<K, V> {
        self.into_alloc_btree_map().into_values()
    }

    #[inline]
    pub fn into_keys(self) -> alloc::collections::btree_map::IntoKeys<K, V> {
        self.into_alloc_btree_map().into_keys()
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> BTreeMap<K, V> {
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let previous = self.inner.insert(key, value);
        self.sync();
        previous
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.sync();
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.inner.append(&mut other.inner);
        self.sync();
        other.sync();
    }

    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let len_before = self.inner.len();
        let trace = &mut self.trace as *mut AllocationHandle;
        match self.inner.entry(key) {
            inner::BTreeMapEntry::Occupied(inner) => Entry::Occupied(OccupiedEntry {
                inner,
                trace,
                len_before,
            }),
            inner::BTreeMapEntry::Vacant(inner) => Entry::Vacant(VacantEntry {
                inner,
                trace,
                len_before,
            }),
        }
    }
}

#[cfg(feature = "memprof")]
impl<K, V> Drop for BTreeMap<K, V> {
    fn drop(&mut self) {
        self.trace.remove();
    }
}

#[cfg(feature = "memprof")]
impl<K, V> Default for BTreeMap<K, V> {
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memprof")]
impl<K: Clone, V: Clone> Clone for BTreeMap<K, V> {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_btree_map(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl<K, V> From<inner::BTreeMap<K, V>> for BTreeMap<K, V> {
    #[track_caller]
    fn from(value: inner::BTreeMap<K, V>) -> Self {
        Self::from_alloc_btree_map(value)
    }
}

#[cfg(feature = "memprof")]
impl<K, V> From<BTreeMap<K, V>> for inner::BTreeMap<K, V> {
    fn from(value: BTreeMap<K, V>) -> Self {
        value.into_alloc_btree_map()
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V, const N: usize> From<[(K, V); N]> for BTreeMap<K, V> {
    #[track_caller]
    fn from(value: [(K, V); N]) -> Self {
        Self::from_alloc_btree_map(inner::BTreeMap::from(value))
    }
}

#[cfg(feature = "memprof")]
impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self::from_alloc_btree_map(inner::BTreeMap::from_iter(iter))
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
impl<K, V> IntoIterator for BTreeMap<K, V> {
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
    trace: *mut AllocationHandle,
    len_before: usize,
}

#[cfg(feature = "memprof")]
pub struct OccupiedEntry<'a, K, V> {
    inner: inner::BTreeMapOccupiedEntry<'a, K, V>,
    trace: *mut AllocationHandle,
    len_before: usize,
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
        let value_ref = self.inner.insert(value);
        unsafe {
            sync_btree_map_trace::<K, V>(self.trace, self.len_before.saturating_add(1));
        }
        value_ref
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
        let value = self.inner.remove();
        unsafe {
            sync_btree_map_trace::<K, V>(self.trace, self.len_before.saturating_sub(1));
        }
        value
    }

    pub fn remove_entry(self) -> (K, V) {
        let pair = self.inner.remove_entry();
        unsafe {
            sync_btree_map_trace::<K, V>(self.trace, self.len_before.saturating_sub(1));
        }
        pair
    }
}

#[cfg(feature = "memprof")]
/// A tracked ordered set.
///
/// Heap usage is estimated from the standard library B-tree node layout and
/// current `len()`.
pub struct BTreeSet<T> {
    inner: inner::BTreeSet<T>,
    trace: AllocationHandle,
}

#[cfg(feature = "memprof")]
impl<T> BTreeSet<T> {
    #[inline]
    #[track_caller]
    pub fn new() -> Self {
        Self::from_alloc_btree_set(inner::BTreeSet::new())
    }

    #[inline]
    #[track_caller]
    pub fn from_alloc_btree_set(inner: inner::BTreeSet<T>) -> Self {
        let trace = AllocationHandle::new(
            AllocationDescriptor::new("BTreeSet", type_name::<T>())
                .with_element_type(type_name::<T>()),
            btree_set_state::<T>(inner.len()),
        );
        Self { inner, trace }
    }

    #[inline]
    fn sync(&mut self) {
        self.trace.update(btree_set_state::<T>(self.inner.len()));
    }

    #[inline]
    pub fn into_alloc_btree_set(mut self) -> inner::BTreeSet<T> {
        self.trace.remove();
        mem::take(&mut self.inner)
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> BTreeSet<T> {
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        let inserted = self.inner.insert(value);
        if inserted {
            self.sync();
        }
        inserted
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.sync();
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord,
    {
        let removed = self.inner.remove(value);
        if removed {
            self.sync();
        }
        removed
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.inner.append(&mut other.inner);
        self.sync();
        other.sync();
    }
}

#[cfg(feature = "memprof")]
impl<T> Drop for BTreeSet<T> {
    fn drop(&mut self) {
        self.trace.remove();
    }
}

#[cfg(feature = "memprof")]
impl<T> Default for BTreeSet<T> {
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memprof")]
impl<T: Clone> Clone for BTreeSet<T> {
    #[track_caller]
    fn clone(&self) -> Self {
        Self::from_alloc_btree_set(self.inner.clone())
    }
}

#[cfg(feature = "memprof")]
impl<T> From<inner::BTreeSet<T>> for BTreeSet<T> {
    #[track_caller]
    fn from(value: inner::BTreeSet<T>) -> Self {
        Self::from_alloc_btree_set(value)
    }
}

#[cfg(feature = "memprof")]
impl<T> From<BTreeSet<T>> for inner::BTreeSet<T> {
    fn from(value: BTreeSet<T>) -> Self {
        value.into_alloc_btree_set()
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord, const N: usize> From<[T; N]> for BTreeSet<T> {
    #[track_caller]
    fn from(value: [T; N]) -> Self {
        Self::from_alloc_btree_set(inner::BTreeSet::from(value))
    }
}

#[cfg(feature = "memprof")]
impl<T: Ord> FromIterator<T> for BTreeSet<T> {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_alloc_btree_set(inner::BTreeSet::from_iter(iter))
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
impl<T> IntoIterator for BTreeSet<T> {
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

// === Facade modules ==========================================================
//
// These modules expose the `tracked_alloc` surface in an `alloc`-shaped layout.
// Some items are tracked wrappers (`collections::BTreeMap`, `collections::BTreeSet`),
// while others intentionally remain plain `alloc` re-exports.

/// Tracked box surface.
pub mod boxed {
    pub use crate::Box;
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

    pub use crate::{from_alloc_vec, into_alloc_vec, Vec};
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
        let values = vec![1u32, 2, 3];
        assert_eq!(values.len(), 3);
        assert_eq!(values[1], 2);
    }

    #[test]
    fn from_alloc_vec_round_trip() {
        let values = from_alloc_vec(alloc::vec![1u8, 2, 3]);
        let raw: inner::Vec<u8> = values.into();
        assert_eq!(raw, alloc::vec![1u8, 2, 3]);
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
    fn len_only_vec_updates_do_not_emit_profile_events() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut values = Vec::<u8>::with_capacity(8);
        let events_after_reserve = profile().events.len();
        values.push(1);
        values.push(2);
        values.push(3);
        let profile_after_pushes = profile();
        assert_eq!(profile_after_pushes.events.len(), events_after_reserve);
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
    fn tracking_profile_reconstructs_runtime_allocations() {
        let _guard = tracking_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        set_tracking_enabled(true);
        reset_tracking();

        let mut code = AllocationHandle::new(
            AllocationDescriptor::new("RuntimeMemory", "CodeBuffer"),
            AllocationState::new(64)
                .with_len(0)
                .with_capacity(64)
                .with_ptr(0x1000),
        );
        let profile0 = profile();
        let create_time = profile0.events.last().expect("create event").time_ns;
        assert_eq!(profile0.snapshot.records.len(), 1);
        assert_eq!(profile0.snapshot.records[0].owner_kind, "RuntimeMemory");
        assert_eq!(profile0.snapshot.records[0].type_name, "CodeBuffer");

        code.update(
            AllocationState::new(128)
                .with_len(32)
                .with_capacity(128)
                .with_ptr(0x1000),
        );
        let profile1 = profile();
        let update_time = profile1.events.last().expect("update event").time_ns;
        let create_snapshot = snapshot_at(create_time);
        let update_snapshot = snapshot_at(update_time);
        assert_eq!(create_snapshot.records[0].size_bytes, 64);
        assert_eq!(update_snapshot.records[0].size_bytes, 128);
        assert_eq!(update_snapshot.records[0].len, Some(32));

        code.remove();
        let profile2 = profile();
        assert!(profile2.snapshot.records.is_empty());
        assert_eq!(
            profile2.events.last().expect("remove event").kind,
            ProfileEventKind::Remove
        );
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
        assert_eq!(snapshot1.records.len(), 1);
        assert_eq!(snapshot1.records[0].len, Some(2));
        assert!(snapshot1.records[0].size_bytes > 0);

        assert_eq!(map.remove(&1), Some(10));
        let snapshot2 = snapshot();
        assert_eq!(snapshot2.records[0].len, Some(1));

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
        assert_eq!(snapshot0.records.len(), 1);
        assert_eq!(snapshot0.records[0].owner_kind, "BTreeSet");
        assert_eq!(snapshot0.records[0].type_name, type_name::<u32>());
        assert_eq!(snapshot0.records[0].len, Some(3));

        assert!(!set.insert(3));
        assert!(set.insert(7));
        assert!(set.remove(&1));
        let snapshot1 = snapshot();
        assert_eq!(snapshot1.records[0].len, Some(3));
        assert!(snapshot1.records[0].size_bytes > 0);

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
