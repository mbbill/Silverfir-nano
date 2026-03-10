//! Cold/complex Rust helpers for the native backend.
//!
//! Helpers should use one uniform normal-ABI boundary and typed metadata.

use super::{
    bridge::NativeHelper,
    context::NativeContext,
    helper_meta::{ColdHelperKind, HelperMetadataHeader, HelperResult},
};

/// Default cold helper stub used while the new backend is still being filled.
pub unsafe extern "C" fn unimplemented_helper(
    _ctx: *mut NativeContext,
    _fp: *mut u64,
    meta: *const HelperMetadataHeader,
) -> HelperResult {
    // TODO: Replace with typed helper implementations.
    unsafe {
        HelperResult {
            next_entry: (*meta).next_entry,
        }
    }
}

pub const DEFAULT_HELPER: NativeHelper = unimplemented_helper;

pub fn helper_for_kind(_kind: ColdHelperKind) -> NativeHelper {
    // TODO: Route each cold helper kind to its typed Rust implementation while
    // keeping the ABI uniform.
    DEFAULT_HELPER
}
