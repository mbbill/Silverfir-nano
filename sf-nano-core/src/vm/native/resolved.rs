use alloc::vec::Vec;

use crate::vm::compaction::CompactionDisposition;
use crate::vm::lowered::{IrOpKind, OpIndex};
use crate::vm::native::instruction::NativeEntry;

#[derive(Clone)]
pub struct ResolvedNativeInst {
    pub entry: NativeEntry,
    pub kind: IrOpKind,
    pub alt_target: Option<OpIndex>,
    pub has_target: bool,
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
