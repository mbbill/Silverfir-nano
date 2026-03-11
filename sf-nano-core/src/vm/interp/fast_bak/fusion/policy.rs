//! Fusion-specific grouping policy support.
//!
//! The important architectural point is that fusion no longer discovers groups
//! inside the fast backend. Instead, planning asks whether a semantic prefix is
//! still compatible with the prebuilt fusion table.

use crate::vm::{interp::fast::fusion::table::FusionPatternTable, wasm::core_op::CoreOpKind};

pub trait FusionPatternMatcher {
    fn can_start(&self, op: &CoreOpKind) -> bool;
    fn can_extend(&self, prefix: &[CoreOpKind], next: &CoreOpKind) -> bool;
}

impl FusionPatternMatcher for FusionPatternTable {
    fn can_start(&self, op: &CoreOpKind) -> bool {
        self.patterns
            .iter()
            .any(|pattern| pattern.ops.first().is_some_and(|first| first == op))
    }

    fn can_extend(&self, prefix: &[CoreOpKind], next: &CoreOpKind) -> bool {
        self.patterns.iter().any(|pattern| {
            pattern.ops.len() > prefix.len()
                && pattern.ops[..prefix.len()] == *prefix
                && pattern.ops[prefix.len()] == *next
        })
    }
}
