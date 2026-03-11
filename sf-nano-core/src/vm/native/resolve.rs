//! Resolve native IR into native-owned entry identities.
//!
//! This layer decides which native blocks/helpers become direct-entry patch
//! targets. It does not rediscover frontend grouping policy or reinterpret LIR.

use alloc::vec::Vec;

use super::{
    entry::NativeEntry,
    helper_meta::ColdHelperKind,
    ir::{NativeBlockId, NativeProgram},
};

/// Native-side resolved entry.
#[derive(Clone, Debug)]
pub struct ResolvedNativeEntry {
    pub kind: ResolvedNativeEntryKind,
    pub block: Option<NativeBlockId>,
    pub entry: Option<NativeEntry>,
    pub cold_helper: Option<ColdHelperKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedNativeEntryKind {
    Block,
    ColdHelper,
}

/// Resolve native blocks into native-owned entry identities.
pub fn resolve_native(program: &NativeProgram) -> Vec<ResolvedNativeEntry> {
    program
        .blocks
        .iter()
        .map(|block| ResolvedNativeEntry {
            kind: ResolvedNativeEntryKind::Block,
            block: Some(block.id),
            entry: None,
            cold_helper: None,
        })
        .collect()
}
