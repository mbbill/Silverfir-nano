pub(crate) use tracked_alloc::{into_alloc_vec, vec, Vec};
#[cfg(feature = "memprof")]
#[allow(
    unused_imports,
    reason = "memprof facade: the aliases exist for embedders and tools; the core crate itself does not reset or snapshot"
)]
pub(crate) use tracked_alloc::{
    reset_tracking as reset_tracked_alloc, snapshot as tracked_alloc_snapshot,
};
