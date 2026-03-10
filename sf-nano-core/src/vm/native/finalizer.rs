//! Native finalization and target patching.
//!
//! This should patch direct native entries and helper metadata without
//! reintroducing stack/T-window reconstruction.

use alloc::vec::Vec;

use crate::vm::lir::legacy::lower::LirProgram;

use super::{
    code::{DirectCallPatch, NativeCode},
    entry::NativeEntry,
    helper_meta::{ColdHelperKind, HelperMetadataArena},
    resolve::ResolvedNativeEntry,
};

#[derive(Clone, Debug)]
struct FinalEntryInfo {
    resolved_index: usize,
    group_id: Option<crate::vm::plan::group::GroupId>,
    lir_range: core::ops::Range<usize>,
    entry: Option<NativeEntry>,
    cold_helper: Option<ColdHelperKind>,
}

/// Finalization output for one native function.
#[derive(Debug, Default)]
pub struct NativeFinalized {
    pub code: NativeCode,
}

pub fn finalize_native(_lir: &LirProgram, resolved: &[ResolvedNativeEntry]) -> NativeFinalized {
    let _entries = build_final_entry_table(resolved);
    // TODO: Build direct native entry table, helper metadata, and patch lists
    // without reintroducing generic continuation slot descriptors.
    NativeFinalized {
        code: NativeCode::from_parts(
            Option::<NativeEntry>::None,
            Vec::new(),
            HelperMetadataArena::new(),
            Vec::<DirectCallPatch>::new(),
        ),
    }
}

fn build_final_entry_table(resolved: &[ResolvedNativeEntry]) -> Vec<FinalEntryInfo> {
    resolved
        .iter()
        .enumerate()
        .map(|(resolved_index, entry)| FinalEntryInfo {
            resolved_index,
            group_id: entry.group_id,
            lir_range: entry.lir_range.clone(),
            entry: entry.entry,
            cold_helper: entry.cold_helper,
        })
        .collect()
}
