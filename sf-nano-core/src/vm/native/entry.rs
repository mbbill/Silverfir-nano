//! Native entry pointer and direct-entry concepts.

/// Direct native entry pointer.
pub type NativeEntry = unsafe extern "C" fn();
