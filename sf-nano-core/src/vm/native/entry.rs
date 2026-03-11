//! Native entry ABI.
//!
//! A native entry is the runtime boundary after native finalization.
//! The current bring-up path uses one shared Rust entry that executes finalized
//! native metadata. Later direct ARM64 emission can replace that entry with
//! real machine-code entry points without changing the runtime call shape.

use super::context::NativeContext;

/// Direct native entry pointer.
pub type NativeEntry = unsafe extern "C" fn(
    ctx: *mut NativeContext,
    fp: *mut u64,
    l0: u64,
    l1: u64,
    l2: u64,
    t0: u64,
    t1: u64,
    t2: u64,
    t3: u64,
);
