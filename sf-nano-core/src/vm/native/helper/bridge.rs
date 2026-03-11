//! Native helper-call bridge generation.
//!
//! This file must not contain:
//! - `read_t0` / `read_topN` / `write_tN` entry families
//! - runtime TOS reconstruction
//! - interpreter-style descriptor decoding
//!
//! Cold wrappers obey the same lowered contract as hot native entries.

use alloc::vec::Vec;

use crate::vm::native::runtime::context::NativeContext;

use super::meta::{ColdHelperKind, HelperMetadataArena, HelperMetadataHeader, HelperResult};

/// Uniform native helper ABI.
pub type NativeHelper = unsafe extern "C" fn(
    ctx: *mut NativeContext,
    fp: *mut u64,
    meta: *const HelperMetadataHeader,
) -> HelperResult;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColdWrapperSite {
    pub helper: ColdHelperKind,
    pub metadata_index: u32,
    pub lir_range: core::ops::Range<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct ColdWrapperPlan {
    pub metadata: HelperMetadataArena,
    pub sites: Vec<ColdWrapperSite>,
}

// TODO: Rebuild uniform cold-wrapper ABI glue around this single helper shape.
// The wrapper builder should accept `ColdWrapperSite` + typed metadata and emit
// one uniform ABI bridge, not per-TOS helper families.
