//! Native selection from shared LIR.
//!
//! This phase is where native-owned policy will decide:
//! - inline leaf op vs cold helper
//! - fast `call_local` vs generic call path
//! - direct return shapes
//! - block tail fusion opportunities before placement
//!
//! The live native backend has not been migrated here yet; this is the stable
//! file boundary for that work.

use alloc::vec::Vec;

use crate::vm::{
    lir::{
        ir::{LirEdge, LirInstKind, LirProgram, LirTerminator, LirValue},
        leaf::LirLeafOp,
        slot::FrameSlot,
    },
    native::ir::NativeBlockId,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SelectedProgram {
    pub entry: NativeBlockId,
    pub blocks: Vec<SelectedBlock>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SelectedBlock {
    pub id: NativeBlockId,
    pub tos_params: Vec<LirValue>,
    pub ops: Vec<SelectedInst>,
    pub terminator: SelectedTerminator,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SelectedInst {
    pub kind: SelectedInstKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedSource {
    Value(LirValue),
    Imm64(u64),
    Hot(u8),
    Operand(FrameSlot),
    FrameLocal(FrameSlot),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedTarget {
    Value(LirValue),
    Hot(u8),
    Operand(FrameSlot),
    FrameLocal(FrameSlot),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SelectedInstKind {
    Copy {
        dst: SelectedTarget,
        src: SelectedSource,
    },
    Leaf {
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallExternal {
        func_idx: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallLocal {
        callee: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index: LirValue,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct SelectedEdge {
    pub target: NativeBlockId,
    pub tos: Vec<LirValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SelectedTerminator {
    Goto(SelectedEdge),
    Branch {
        cond: LirValue,
        then_edge: SelectedEdge,
        else_edge: SelectedEdge,
    },
    BrTable {
        index: LirValue,
        entries: Vec<SelectedEdge>,
    },
    Return {
        values: Vec<LirValue>,
    },
    TrapUnreachable,
}

#[inline]
pub(super) fn select_program(lir: &LirProgram) -> SelectedProgram {
    SelectedProgram {
        entry: NativeBlockId::from(lir.entry),
        blocks: lir
            .blocks
            .iter()
            .map(|block| SelectedBlock {
                id: NativeBlockId::from(block.id),
                tos_params: block.params.tos.clone(),
                ops: block.ops.iter().map(select_inst).collect(),
                terminator: select_terminator(&block.terminator),
            })
            .collect(),
    }
}

fn select_inst(inst: &crate::vm::lir::ir::LirInst) -> SelectedInst {
    let kind = match &inst.kind {
        LirInstKind::Leaf { op, args, results } => select_leaf(op, args, results),
        LirInstKind::WriteOperandSlot { slot, src } => SelectedInstKind::Copy {
            dst: SelectedTarget::Operand(*slot),
            src: SelectedSource::Value(*src),
        },
        LirInstKind::ReadOperandSlot { slot, dst } => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*dst),
            src: SelectedSource::Operand(*slot),
        },
        LirInstKind::ReadHotLocal { reg, dst } => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*dst),
            src: SelectedSource::Hot(*reg),
        },
        LirInstKind::WriteHotLocal { reg, src } => SelectedInstKind::Copy {
            dst: SelectedTarget::Hot(*reg),
            src: SelectedSource::Value(*src),
        },
        LirInstKind::ReadFrameLocal { frame_slot, dst } => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*dst),
            src: SelectedSource::FrameLocal(*frame_slot),
        },
        LirInstKind::WriteFrameLocal { frame_slot, src } => SelectedInstKind::Copy {
            dst: SelectedTarget::FrameLocal(*frame_slot),
            src: SelectedSource::Value(*src),
        },
        LirInstKind::CallExternal {
            func_idx,
            args,
            results,
        } => SelectedInstKind::CallExternal {
            func_idx: *func_idx,
            args: args.clone(),
            results: results.clone(),
        },
        LirInstKind::CallInternal {
            callee,
            args,
            results,
        } => SelectedInstKind::CallLocal {
            callee: *callee,
            args: args.clone(),
            results: results.clone(),
        },
        LirInstKind::CallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            results,
        } => SelectedInstKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            index: *index,
            args: args.clone(),
            results: results.clone(),
        },
    };
    SelectedInst { kind }
}

fn select_leaf(op: &LirLeafOp, args: &[LirValue], results: &[LirValue]) -> SelectedInstKind {
    match (op, results) {
        (LirLeafOp::I32Const { value }, [result]) => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*result),
            src: SelectedSource::Imm64(u64::from(*value)),
        },
        (LirLeafOp::I64Const { value }, [result]) => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*result),
            src: SelectedSource::Imm64(*value),
        },
        (LirLeafOp::F32Const { value }, [result]) => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*result),
            src: SelectedSource::Imm64(u64::from(*value)),
        },
        (LirLeafOp::F64Const { value }, [result]) => SelectedInstKind::Copy {
            dst: SelectedTarget::Value(*result),
            src: SelectedSource::Imm64(*value),
        },
        _ => SelectedInstKind::Leaf {
            op: op.clone(),
            args: args.to_vec(),
            results: results.to_vec(),
        },
    }
}

fn select_terminator(term: &LirTerminator) -> SelectedTerminator {
    match term {
        LirTerminator::Goto(edge) => SelectedTerminator::Goto(select_edge(edge)),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => SelectedTerminator::Branch {
            cond: *cond,
            then_edge: select_edge(then_edge),
            else_edge: select_edge(else_edge),
        },
        LirTerminator::BrTable { index, entries } => SelectedTerminator::BrTable {
            index: *index,
            entries: entries.iter().map(select_edge).collect(),
        },
        LirTerminator::Return { values } => SelectedTerminator::Return {
            values: values.clone(),
        },
        LirTerminator::TrapUnreachable => SelectedTerminator::TrapUnreachable,
    }
}

fn select_edge(edge: &LirEdge) -> SelectedEdge {
    SelectedEdge {
        target: NativeBlockId::from(edge.target),
        tos: edge.tos.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{select_inst, SelectedInstKind, SelectedSource, SelectedTarget};
    use crate::vm::lir::{
        ir::{LirInst, LirInstKind, LirValue},
        leaf::LirLeafOp,
    };

    #[test]
    fn selects_i32_const_as_immediate_copy() {
        let inst = LirInst {
            kind: LirInstKind::Leaf {
                op: LirLeafOp::I32Const { value: 7 },
                args: alloc::vec![],
                results: alloc::vec![LirValue(3)],
            },
        };

        assert_eq!(
            select_inst(&inst).kind,
            SelectedInstKind::Copy {
                dst: SelectedTarget::Value(LirValue(3)),
                src: SelectedSource::Imm64(7),
            }
        );
    }

    #[test]
    fn keeps_non_const_leaf_as_leaf() {
        let inst = LirInst {
            kind: LirInstKind::Leaf {
                op: LirLeafOp::I32Add,
                args: alloc::vec![LirValue(0), LirValue(1)],
                results: alloc::vec![LirValue(2)],
            },
        };

        assert_eq!(
            select_inst(&inst).kind,
            SelectedInstKind::Leaf {
                op: LirLeafOp::I32Add,
                args: alloc::vec![LirValue(0), LirValue(1)],
                results: alloc::vec![LirValue(2)],
            }
        );
    }
}
