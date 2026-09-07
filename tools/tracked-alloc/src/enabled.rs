use super::*;
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
std::thread_local! { static INTERNAL: Cell<bool> = const { Cell::new(false) }; }

struct InternalGuard;
impl InternalGuard {
    fn enter() -> Option<Self> {
        INTERNAL
            .try_with(|v| if v.replace(true) { None } else { Some(Self) })
            .ok()
            .flatten()
    }
}
impl Drop for InternalGuard {
    fn drop(&mut self) {
        // A successfully entered guard is dropped before its TLS is destroyed.
        INTERNAL.with(|v| v.set(false));
    }
}

pub(super) fn initialized() -> bool {
    STATE.get().is_some()
}
pub(super) fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let _guard = InternalGuard::enter()?;
    // No allocation from bookkeeping may recurse into the profiler. The guard
    // also covers initialization, snapshots, mutex poisoning and phase records.
    let mut state = STATE
        .get_or_init(|| Mutex::new(State::new()))
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    Some(f(&mut state))
}

pub(super) struct State {
    start: Instant,
    generation: u64,
    next_id: u64,
    heap: HashMap<usize, usize>,
    runtime: HashMap<u64, (AllocationDescriptor, AllocationState)>,
    stats: RegistrySnapshot,
    phases: Vec<ProfilePhase>,
    omitted_phases: usize,
}
impl State {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            generation: 0,
            next_id: 1,
            heap: HashMap::new(),
            runtime: HashMap::new(),
            stats: RegistrySnapshot::default(),
            phases: Vec::new(),
            omitted_phases: 0,
        }
    }
    fn time_ns(&self) -> u64 {
        self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
    pub(super) fn allocate(&mut self, ptr: usize, size: usize) {
        if !tracking_enabled() {
            return;
        }
        self.heap.insert(ptr, size);
        self.stats.total_bytes += size;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.total_bytes);
        self.stats.allocated_bytes = self.stats.allocated_bytes.saturating_add(size as u64);
        self.stats.allocations += 1;
    }
    pub(super) fn deallocate(&mut self, ptr: usize) {
        if let Some(size) = self.heap.remove(&ptr) {
            self.stats.total_bytes -= size;
            self.stats.deallocations += 1;
        }
    }
    pub(super) fn reallocate(&mut self, old: usize, new: usize, size: usize) {
        if let Some(previous) = self.heap.remove(&old) {
            self.heap.insert(new, size);
            self.stats.total_bytes = self.stats.total_bytes - previous + size;
            self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.total_bytes);
            self.stats.allocated_bytes = self
                .stats
                .allocated_bytes
                .saturating_add(size.saturating_sub(previous) as u64);
            self.stats.reallocations += 1;
        } else {
            self.allocate(new, size);
        }
    }
    fn snapshot(&self) -> RegistrySnapshot {
        let mut snapshot = self.stats.clone();
        snapshot.live_allocations = self.heap.len();
        for &(descriptor, state) in self.runtime.values() {
            snapshot.records.push((descriptor, state));
            match descriptor.type_name {
                RUNTIME_TYPE_CODE_BUFFER => snapshot.code_buffer_bytes += state.size_bytes,
                RUNTIME_TYPE_GUARD_PAGE => snapshot.guard_page_bytes += state.size_bytes,
                _ => snapshot.other_runtime_bytes += state.size_bytes,
            }
        }
        snapshot.records.sort_by_key(|(d, s)| (d.type_name, s.ptr));
        snapshot
    }

    fn record_runtime_peaks(&mut self) {
        let mut code = 0;
        let mut guard = 0;
        let mut other = 0;
        for (descriptor, state) in self.runtime.values() {
            match descriptor.type_name {
                RUNTIME_TYPE_CODE_BUFFER => code += state.size_bytes,
                RUNTIME_TYPE_GUARD_PAGE => guard += state.size_bytes,
                _ => other += state.size_bytes,
            }
        }
        self.stats.peak_code_buffer_bytes = self.stats.peak_code_buffer_bytes.max(code);
        self.stats.peak_guard_page_bytes = self.stats.peak_guard_page_bytes.max(guard);
        self.stats.peak_other_runtime_bytes = self.stats.peak_other_runtime_bytes.max(other);
    }
}

/// Start/pause recording new allocations. Install `TrackingAllocator` in the
/// executable to collect heap data; explicit runtime and phase hooks work alone.
pub fn set_tracking_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}
pub fn tracking_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Begin a fresh measurement, forgetting earlier live allocations. Existing
/// runtime handles and phase guards cannot mutate the new measurement.
pub fn reset_tracking() {
    with_state(|s| {
        let generation = s
            .generation
            .checked_add(1)
            .expect("profiler generation exhausted");
        *s = State::new();
        s.generation = generation;
    });
}
pub fn snapshot() -> RegistrySnapshot {
    with_state(|s| s.snapshot()).unwrap_or_default()
}
pub fn profile() -> AllocationProfile {
    with_state(|s| AllocationProfile {
        snapshot: s.snapshot(),
        phases: s.phases.clone(),
        omitted_phases: s.omitted_phases,
    })
    .unwrap_or_default()
}

/// Existing explicit hooks account for executable/guard memory separately from
/// heap allocations. No allocation ownership or runtime lifetime is changed.
#[derive(Debug)]
pub struct AllocationHandle {
    key: Option<(u64, u64)>,
}
impl AllocationHandle {
    pub fn new(descriptor: AllocationDescriptor, state: AllocationState) -> Self {
        let key = if tracking_enabled() {
            with_state(|s| {
                let id = s.next_id;
                s.next_id = id
                    .checked_add(1)
                    .expect("profiler handle identity exhausted");
                s.runtime.insert(id, (descriptor, state));
                s.record_runtime_peaks();
                (s.generation, id)
            })
        } else {
            None
        };
        Self { key }
    }
    pub fn update(&mut self, state: AllocationState) {
        if let Some((generation, id)) = self.key {
            with_state(|s| {
                if generation == s.generation {
                    if let Some(record) = s.runtime.get_mut(&id) {
                        record.1 = state;
                    }
                    s.record_runtime_peaks();
                }
            });
        }
    }
    pub fn remove(&mut self) {
        if let Some((generation, id)) = self.key.take() {
            with_state(|s| {
                if generation == s.generation {
                    s.runtime.remove(&id);
                }
            });
        }
    }
}
impl Drop for AllocationHandle {
    fn drop(&mut self) {
        self.remove();
    }
}

pub struct PhaseGuard {
    start: Option<(u64, u64, u64)>,
    name: &'static str,
    function_index: Option<u32>,
}
#[must_use]
pub fn phase_span(name: &'static str) -> PhaseGuard {
    phase_span_with_function(name, None)
}
#[must_use]
pub fn phase_span_with_function(name: &'static str, function_index: Option<u32>) -> PhaseGuard {
    let start = if tracking_enabled() {
        with_state(|s| (s.generation, s.time_ns(), s.stats.allocated_bytes))
    } else {
        None
    };
    PhaseGuard {
        start,
        name,
        function_index,
    }
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if let Some((generation, start_time_ns, allocated_bytes)) = self.start {
            with_state(|s| {
                if generation != s.generation {
                    return;
                }
                // Large modules must not create unbounded diagnostic history.
                if s.phases.len() == 16_384 {
                    s.omitted_phases += 1;
                    return;
                }
                s.phases.push(ProfilePhase {
                    name: self.name,
                    function_index: self.function_index,
                    start_time_ns,
                    end_time_ns: s.time_ns(),
                    allocated_bytes: s.stats.allocated_bytes.saturating_sub(allocated_bytes),
                });
            });
        }
    }
}
