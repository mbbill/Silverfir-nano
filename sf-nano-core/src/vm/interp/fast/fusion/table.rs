//! Prebuilt fusion pattern table.
//!
//! In the refactored design this is a planning input, not a late backend-local
//! matcher. The planning layer uses it to constrain shared grouping.

use alloc::vec::Vec;

use crate::vm::wasm::core_op::CoreOpKind;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FusionPattern {
    pub id: u32,
    pub ops: Vec<CoreOpKind>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FusionPatternTable {
    pub patterns: Vec<FusionPattern>,
}

impl FusionPatternTable {
    #[inline]
    pub fn empty() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }
}
