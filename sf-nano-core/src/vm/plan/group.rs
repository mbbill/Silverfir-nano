//! Prepared-LIR grouping metadata.
//!
//! Grouping is computed after semantic-to-LIR preparation so backends consume
//! real prepared operations instead of a separate pseudo-op stream.
//!
//! In maximal mode this forms the largest regions possible between explicit
//! control or runtime boundaries.

use alloc::vec::Vec;

use crate::vm::lir::{
    ir::{LirInstKind, LirProgram, LirTerminator},
    target::LirTarget,
};

use super::policy::{GroupingMode, PlanPolicy};

/// Planned group identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(pub u32);

impl GroupId {
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// One grouped region inside one prepared LIR block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRange {
    pub id: GroupId,
    pub block: LirTarget,
    pub start: u32,
    pub end: u32,
    pub terminator: Option<GroupTerminator>,
}

/// Full grouping plan for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroupPlan {
    pub groups: Vec<GroupRange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupTerminator {
    Branch,
    BrTable,
    Return,
    TrapUnreachable,
}

#[inline]
pub fn plan_groups(program: &LirProgram, policy: PlanPolicy) -> GroupPlan {
    if matches!(policy.grouping, GroupingMode::Ignore) {
        return GroupPlan::default();
    }

    let mut groups = Vec::new();
    let mut next_id = 0u32;

    for block in &program.blocks {
        let mut i = 0usize;
        while i < block.ops.len() {
            if !groupable_inst(&block.ops[i].kind) {
                i += 1;
                continue;
            }

            let start = i;
            i += 1;
            while i < block.ops.len() && groupable_inst(&block.ops[i].kind) {
                i += 1;
            }

            groups.push(GroupRange {
                id: GroupId(next_id),
                block: block.id,
                start: start as u32,
                end: i as u32,
                terminator: (i == block.ops.len()).then(|| terminator_for(&block.terminator)).flatten(),
            });
            next_id += 1;
        }
    }

    GroupPlan { groups }
}

#[inline]
fn groupable_inst(kind: &LirInstKind) -> bool {
    !matches!(
        kind,
        LirInstKind::Runtime { .. }
            | LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. }
    )
}

#[inline]
fn terminator_for(terminator: &LirTerminator) -> Option<GroupTerminator> {
    match terminator {
        LirTerminator::Goto(_) => None,
        LirTerminator::Branch { .. } => Some(GroupTerminator::Branch),
        LirTerminator::BrTable { .. } => Some(GroupTerminator::BrTable),
        LirTerminator::Return { .. } => Some(GroupTerminator::Return),
        LirTerminator::TrapUnreachable => Some(GroupTerminator::TrapUnreachable),
    }
}
