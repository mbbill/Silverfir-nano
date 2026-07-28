pub(crate) use tracked_alloc::{into_alloc_vec, vec, Vec};
// `phase_span` and `phase_span_with_function` are only taken by the JIT
// pipeline's stage timers.
#[cfg(sf_jit)]
pub(crate) use tracked_alloc::phase_span;
#[cfg(sf_jit)]
pub(crate) use tracked_alloc::phase_span_with_function;

#[cfg(feature = "memprof")]
#[allow(
    unused_imports,
    reason = "memprof facade: the aliases exist for embedders and tools; the core crate itself does not reset or snapshot"
)]
pub(crate) use tracked_alloc::{
    reset_tracking as reset_tracked_alloc, snapshot as tracked_alloc_snapshot,
};
