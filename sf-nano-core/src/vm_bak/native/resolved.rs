use alloc::vec::Vec;

use crate::vm::abi::compaction::CompactionDisposition;
use crate::vm::wasm::SlotRef;
use crate::vm::lir::{IrOpKind, OpIndex};
use crate::vm::native::bridge::ColdHelperKind;
use crate::vm::native::arm64::EntryPatchSites;
use crate::vm::native::entry::NativeEntry;

#[derive(Clone)]
pub struct ResolvedNativeInst {
    pub entry: NativeEntry,
    pub kind: IrOpKind,
    pub window: u8,
    pub entry_input_count: u8,
    pub frame_slot: Option<SlotRef>,
    #[cfg(feature = "native-dump")]
    pub original_ir_idx: usize,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
    pub cold_helper: Option<ColdHelperKind>,
    pub cold_frame_slot: Option<SlotRef>,
    pub entry_patches: EntryPatchSites,
    pub compaction: CompactionDisposition,
}

impl ResolvedNativeInst {
    #[inline]
    pub fn is_removed(&self) -> bool {
        !self.compaction.is_kept()
    }

    #[inline]
    pub fn redirects_branch_target(&self) -> bool {
        self.compaction.may_redirect_branch_target()
    }

    #[inline]
    pub fn is_internal_only(&self) -> bool {
        matches!(self.compaction, CompactionDisposition::InternalOnly)
    }
}

pub type NativeResolvedVec = Vec<ResolvedNativeInst>;
