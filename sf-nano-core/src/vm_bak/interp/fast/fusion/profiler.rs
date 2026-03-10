//! Fast interpreter instruction sequence profiler.
//!
//! Captures N-instruction sequences to identify super-instruction candidates.
//! Enabled via `profile` feature which passes `FAST_PROFILE_ENABLED` to C handlers.
//!
//! The profiler works with handler name strings passed directly from C macros,
//! eliminating the need for function pointer lookup tables.

extern crate std;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Maximum supported window size for sequence capture.
pub const MAX_WINDOW_SIZE: usize = 8;

// Global configuration
static ENABLED: AtomicBool = AtomicBool::new(false);
static WINDOW_SIZE: AtomicUsize = AtomicUsize::new(2);

/// Sequence key: array of handler names representing an instruction sequence.
///
/// Uses pointer-based Hash/Eq since all names are interned (unique pointer per
/// unique string content). This makes HashMap operations faster than
/// string-content hashing.
#[derive(Debug, Clone)]
pub struct SequenceKey {
    names: [&'static str; MAX_WINDOW_SIZE],
    len: usize,
}

impl core::hash::Hash for SequenceKey {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for name in &self.names[..self.len] {
            (name.as_ptr() as usize).hash(state);
        }
    }
}

impl PartialEq for SequenceKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.names[..self.len]
                .iter()
                .zip(other.names[..other.len].iter())
                .all(|(a, b)| core::ptr::eq(a.as_ptr(), b.as_ptr()))
    }
}

impl Eq for SequenceKey {}

impl SequenceKey {
    /// Format the sequence as human-readable string.
    pub fn format(&self) -> std::string::String {
        self.names[..self.len].join(" -> ")
    }

    /// Get the sequence length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the op names as a slice.
    pub fn ops(&self) -> &[&'static str] {
        &self.names[..self.len]
    }

    /// Check if this sequence is fuseable (no control flow in middle positions).
    pub fn is_fuseable(&self) -> bool {
        if self.len <= 1 {
            return true;
        }
        for &name in &self.names[..self.len - 1] {
            if is_control_flow(name) {
                return false;
            }
        }
        true
    }
}

/// Check if a handler name represents a control flow instruction.
/// Handles variant-qualified names like "br_if_d1", "if__d1".
fn is_control_flow(name: &str) -> bool {
    let base = strip_variant_suffix_str(name);
    matches!(
        base,
        "br" | "br_if" | "br_table"
        | "block" | "loop" | "if_" | "else_" | "end"
        | "call" | "call_indirect" | "return"
        | "return_call" | "return_call_indirect"
        | "unreachable"
    )
}

/// Strip _dN variant suffix from a handler name.
fn strip_variant_suffix_str(name: &str) -> &str {
    if name.len() >= 3 {
        let suffix = &name[name.len() - 3..];
        if matches!(suffix, "_d1" | "_d2" | "_d3" | "_d4") {
            return &name[..name.len() - 3];
        }
    }
    name
}

// All profiler state in a single thread-local to minimize TLS lookups.
struct ProfilerState {
    window: SlidingWindow,
    sequences: HashMap<SequenceKey, u64>,
    // Cache: C pointer address -> interned &'static str (already normalized)
    name_cache: HashMap<usize, &'static str>,
    // Pool: ensures unique pointer per unique string content (for pointer-based Eq)
    str_pool: HashMap<&'static str, &'static str>,
    total: u64,
}

impl ProfilerState {
    fn new() -> Self {
        Self {
            window: SlidingWindow::new(),
            sequences: HashMap::new(),
            name_cache: HashMap::new(),
            str_pool: HashMap::new(),
            total: 0,
        }
    }

    /// Intern a C string pointer. Fast path: HashMap lookup by integer key.
    /// Slow path (first call per unique C pointer): parse C string, normalize,
    /// deduplicate via str_pool.
    #[inline]
    fn intern(&mut self, c_name: *const core::ffi::c_char) -> &'static str {
        let addr = c_name as usize;
        if let Some(&cached) = self.name_cache.get(&addr) {
            return cached;
        }
        self.intern_slow(c_name, addr)
    }

    #[inline(never)]
    #[cold]
    fn intern_slow(&mut self, c_name: *const core::ffi::c_char, addr: usize) -> &'static str {
        let raw = intern_name_from_c(c_name);
        let normalized = normalize_handler_name(raw);
        // Ensure pointer uniqueness: same content -> same pointer
        let interned = *self.str_pool.entry(normalized).or_insert(normalized);
        self.name_cache.insert(addr, interned);
        interned
    }

    fn clear(&mut self) {
        self.window.clear();
        self.sequences.clear();
        self.total = 0;
    }

    fn set_capacity(&mut self, cap: usize) {
        self.window.set_capacity(cap);
    }
}

std::thread_local! {
    static STATE: RefCell<ProfilerState> = RefCell::new(ProfilerState::new());
}

struct SlidingWindow {
    buffer: [&'static str; MAX_WINDOW_SIZE],
    count: usize,
    capacity: usize,
}

impl SlidingWindow {
    fn new() -> Self {
        Self {
            buffer: [""; MAX_WINDOW_SIZE],
            count: 0,
            capacity: 2,
        }
    }

    #[inline]
    fn push(&mut self, name: &'static str) -> Option<SequenceKey> {
        if self.capacity == 1 {
            self.buffer[0] = name;
            self.count = 1;
            return Some(self.to_key());
        }

        if self.count < self.capacity {
            self.buffer[self.count] = name;
            self.count += 1;
            if self.count == self.capacity {
                return Some(self.to_key());
            }
            None
        } else {
            for i in 0..self.capacity - 1 {
                self.buffer[i] = self.buffer[i + 1];
            }
            self.buffer[self.capacity - 1] = name;
            Some(self.to_key())
        }
    }

    #[inline]
    fn to_key(&self) -> SequenceKey {
        SequenceKey {
            names: self.buffer,
            len: self.count,
        }
    }

    fn set_capacity(&mut self, cap: usize) {
        self.capacity = cap.min(MAX_WINDOW_SIZE);
        self.count = 0;
    }

    fn clear(&mut self) {
        self.count = 0;
    }
}

// ============================================================================
// FFI Entry Point (called from C handlers)
// ============================================================================

/// Parse a C string and leak it to get a `&'static str`.
fn intern_name_from_c(c_name: *const core::ffi::c_char) -> &'static str {
    let c_bytes = unsafe {
        let mut len = 0;
        let mut p = c_name;
        while *p != 0 {
            len += 1;
            p = p.add(1);
        }
        core::slice::from_raw_parts(c_name as *const u8, len)
    };

    let owned = std::string::String::from(core::str::from_utf8(c_bytes).unwrap_or("?"));
    std::boxed::Box::leak(owned.into_boxed_str())
}

/// Normalize handler names so profiler data matches the discovery pipeline.
/// Handles variant-qualified names: "br_if_simple_d1" → "br_if_d1".
#[inline]
fn normalize_handler_name(name: &'static str) -> &'static str {
    // Check for variant suffix (_d1.._d4)
    let (base, variant_suffix) = if name.len() >= 3 {
        let suffix = &name[name.len() - 3..];
        if matches!(suffix, "_d1" | "_d2" | "_d3" | "_d4") {
            (&name[..name.len() - 3], suffix)
        } else {
            (name, "")
        }
    } else {
        (name, "")
    };

    let normalized_base = match base {
        "br_if_simple" => "br_if",
        _ => base,
    };

    if core::ptr::eq(normalized_base, base) {
        name // no normalization needed
    } else if variant_suffix.is_empty() {
        // Exact match without variant suffix (shouldn't happen with new profile hooks, but safe)
        match normalized_base {
            "br_if" => "br_if",
            _ => name,
        }
    } else {
        // Need to create a new string: normalized_base + variant_suffix
        let new_name = std::format!("{}{}", normalized_base, variant_suffix);
        std::boxed::Box::leak(new_name.into_boxed_str())
    }
}

/// FFI entry point called from C handlers when profiling is enabled.
///
/// Split into thin wrapper + separate impl to minimize register spills.
/// The wrapper is just an ENABLED check + branch (36 bytes).
#[no_mangle]
pub unsafe extern "C" fn fast_profile_record(name: *const core::ffi::c_char) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    fast_profile_record_impl(name);
}

#[inline(never)]
unsafe fn fast_profile_record_impl(name: *const core::ffi::c_char) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let interned = s.intern(name);
        s.total += 1;
        if let Some(key) = s.window.push(interned) {
            *s.sequences.entry(key).or_insert(0) += 1;
        }
    });
}

// ============================================================================
// Public API
// ============================================================================

/// Enable profiling with specified window size (1-8).
pub fn enable(window_size: usize) {
    let ws = window_size.clamp(1, MAX_WINDOW_SIZE);
    WINDOW_SIZE.store(ws, Ordering::Relaxed);

    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.clear();
        s.set_capacity(ws);
    });

    ENABLED.store(true, Ordering::Release);
}

/// Disable profiling and return collected statistics.
pub fn take_stats() -> FastProfileStats {
    ENABLED.store(false, Ordering::Release);

    let window_size = WINDOW_SIZE.load(Ordering::Relaxed);

    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let total = s.total;
        let sequences = core::mem::take(&mut s.sequences);
        s.clear();

        FastProfileStats {
            total_instructions: total,
            window_size,
            sequences,
        }
    })
}

/// Check if profiling is currently enabled.
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

// ============================================================================
// Statistics
// ============================================================================

/// Profiling statistics for instruction sequences.
pub struct FastProfileStats {
    pub total_instructions: u64,
    pub window_size: usize,
    /// Sequence counts: sequence -> execution count.
    pub sequences: HashMap<SequenceKey, u64>,
}

impl FastProfileStats {
    /// Get top N sequences by frequency.
    pub fn top_sequences(&self, n: usize) -> std::vec::Vec<(&SequenceKey, u64)> {
        let mut vec: std::vec::Vec<_> = self.sequences.iter().map(|(k, &v)| (k, v)).collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1));
        vec.truncate(n);
        vec
    }

    /// Get top N fuseable sequences by frequency.
    pub fn top_fuseable_sequences(&self, n: usize) -> std::vec::Vec<(&SequenceKey, u64)> {
        let mut vec: std::vec::Vec<_> = self
            .sequences
            .iter()
            .filter(|(k, _)| k.is_fuseable())
            .map(|(k, &v)| (k, v))
            .collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1));
        vec.truncate(n);
        vec
    }

    /// Aggregate sequences — already normalized since we use base handler names.
    pub fn aggregate_normalized(&self) -> HashMap<SequenceKey, u64> {
        self.sequences.clone()
    }

    /// Get top N fuseable sequences by frequency.
    pub fn top_fuseable_reduction_potential(&self) -> std::vec::Vec<(&SequenceKey, u64, u64)> {
        let mut results: std::vec::Vec<_> = self
            .sequences
            .iter()
            .filter(|(seq, _)| seq.is_fuseable())
            .map(|(seq, &count)| {
                let saved = (seq.len.saturating_sub(1) as u64) * count;
                (seq, count, saved)
            })
            .collect();
        results.sort_by(|a, b| b.2.cmp(&a.2));
        results
    }
}

/// Merge multiple workload stats by frequency averaging.
///
/// Each workload's counts are normalized to frequencies (count / total),
/// then averaged across workloads, then scaled back to integer counts
/// using the average total_instructions as the reference.
pub fn merge_stats(all_stats: std::vec::Vec<FastProfileStats>) -> FastProfileStats {
    if all_stats.len() == 1 {
        return all_stats.into_iter().next().unwrap();
    }

    let n = all_stats.len() as f64;
    let avg_total: u64 =
        (all_stats.iter().map(|s| s.total_instructions).sum::<u64>() as f64 / n) as u64;
    let max_window = all_stats.iter().map(|s| s.window_size).max().unwrap_or(2);

    // Max-frequency merge: for each pattern, use its highest frequency
    // across any workload. This ensures hot patterns aren't diluted by
    // workloads that don't use them.
    let mut freq_max: HashMap<SequenceKey, f64> = HashMap::new();
    for stats in &all_stats {
        let total = stats.total_instructions as f64;
        if total == 0.0 {
            continue;
        }
        for (key, &count) in &stats.sequences {
            let freq = count as f64 / total;
            let entry = freq_max.entry(key.clone()).or_insert(0.0);
            if freq > *entry {
                *entry = freq;
            }
        }
    }

    // Scale max frequencies to reference total
    let mut merged = HashMap::new();
    for (key, max_freq) in freq_max {
        let count = (max_freq * avg_total as f64) as u64;
        if count > 0 {
            merged.insert(key, count);
        }
    }

    FastProfileStats {
        total_instructions: avg_total,
        window_size: max_window,
        sequences: merged,
    }
}
