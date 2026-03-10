//! Native resolution from LIR + planned groups to native-owned resolved entries.
//!
//! This must consume preplanned groups. It must not rediscover groups.

use alloc::vec::Vec;

use crate::vm::lir::lower::LirProgram;
use crate::vm::plan::group::GroupId;

use super::{entry::NativeEntry, helper_meta::ColdHelperKind};

/// Native-side resolved entry.
#[derive(Clone, Debug)]
pub struct ResolvedNativeEntry {
    pub kind: ResolvedNativeEntryKind,
    pub group_id: Option<GroupId>,
    pub lir_range: core::ops::Range<usize>,
    pub entry: Option<NativeEntry>,
    pub cold_helper: Option<ColdHelperKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedNativeEntryKind {
    Group,
    Singleton,
    ColdHelper,
}

/// Resolve LIR and planned groups into native-owned entries.
pub fn resolve_native(lir: &LirProgram) -> Vec<ResolvedNativeEntry> {
    // TODO: Replace this structural placeholder with real native resolution.
    // The important current property is that it starts from planned groups,
    // not backend-local discovery.
    if !lir.groups.is_empty() {
        return lir
            .groups
            .iter()
            .map(|group| ResolvedNativeEntry {
                kind: ResolvedNativeEntryKind::Group,
                group_id: Some(group.id),
                lir_range: group.lir_range.clone(),
                entry: None,
                cold_helper: None,
            })
            .collect();
    }

    lir.ops
        .iter()
        .enumerate()
        .map(|(lir_index, _)| ResolvedNativeEntry {
            kind: ResolvedNativeEntryKind::Singleton,
            group_id: None,
            lir_range: lir_index..lir_index + 1,
            entry: None,
            cold_helper: None,
        })
        .collect()
}
