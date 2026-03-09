use alloc::vec::Vec;

use crate::vm::compaction::CompactionDisposition;
use crate::vm::compile::SlotRef;
use crate::vm::lowered::{IrOpKind, OpIndex};
use crate::vm::native::bridge::ColdHelperKind;
use crate::vm::native::arm64::EntryPatchSites;
use crate::vm::native::instruction::NativeEntry;

#[derive(Clone)]
pub struct ResolvedNativeInst {
    pub entry: NativeEntry,
    pub kind: IrOpKind,
    pub pre_height: u16,
    #[cfg(feature = "native-dump")]
    pub original_ir_idx: usize,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
    pub cold_helper: Option<ColdHelperKind>,
    pub cold_stack_slot: Option<SlotRef>,
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
