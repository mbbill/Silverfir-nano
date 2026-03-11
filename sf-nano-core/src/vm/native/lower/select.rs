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

use crate::{
    error::WasmError,
    vm::{
        lir::{
            ir::{LirEdge, LirInstKind, LirProgram, LirTerminator, LirValue},
            leaf::LirLeafOp,
            slot::FrameSlot,
        },
        native::{
            compile::build::LocalCallContract,
            ir::{NativeBlockId, NativeHelperEffect},
        },
    },
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
pub(super) enum SelectedCompareKind {
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedBranchCond {
    Value(LirValue),
    I32Eqz(LirValue),
    I64Eqz(LirValue),
    I32Compare {
        kind: SelectedCompareKind,
        lhs: LirValue,
        rhs: LirValue,
    },
    I64Compare {
        kind: SelectedCompareKind,
        lhs: LirValue,
        rhs: LirValue,
    },
    F32Compare {
        kind: SelectedCompareKind,
        lhs: LirValue,
        rhs: LirValue,
    },
    F64Compare {
        kind: SelectedCompareKind,
        lhs: LirValue,
        rhs: LirValue,
    },
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
pub(super) enum SelectedHelperInst {
    Leaf {
        effect: NativeHelperEffect,
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
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
    Helper(SelectedHelperInst),
    CallExternal {
        func_idx: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallLocal {
        callee: u32,
        callee_params_count: u16,
        callee_frame_size: u16,
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
        cond: SelectedBranchCond,
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
pub(super) fn select_program(
    lir: &LirProgram,
    local_call_contracts: &[Option<LocalCallContract>],
) -> Result<SelectedProgram, WasmError> {
    let mut blocks = Vec::with_capacity(lir.blocks.len());
    for block in &lir.blocks {
        let mut selected = SelectedBlock {
            id: NativeBlockId::from(block.id),
            tos_params: block.params.tos.clone(),
            ops: block
                .ops
                .iter()
                .map(|inst| select_inst(inst, local_call_contracts))
                .collect::<Result<Vec<_>, _>>()?,
            terminator: select_terminator(&block.terminator),
        };
        fuse_compare_branch(&mut selected);
        blocks.push(selected);
    }

    Ok(SelectedProgram {
        entry: NativeBlockId::from(lir.entry),
        blocks,
    })
}

fn select_inst(
    inst: &crate::vm::lir::ir::LirInst,
    local_call_contracts: &[Option<LocalCallContract>],
) -> Result<SelectedInst, WasmError> {
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
        } => {
            let contract = local_call_contracts
                .get(*callee as usize)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "native lowering is missing local call contract for callee {}",
                        callee
                    ))
                })?;
            SelectedInstKind::CallLocal {
                callee: *callee,
                callee_params_count: contract.params_count,
                callee_frame_size: contract.frame_size,
                args: args.clone(),
                results: results.clone(),
            }
        }
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
    Ok(SelectedInst { kind })
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
        _ if lower_leaf_to_helper(op) => SelectedInstKind::Helper(SelectedHelperInst::Leaf {
            effect: NativeHelperEffect::Transparent,
            op: op.clone(),
            args: args.to_vec(),
            results: results.to_vec(),
        }),
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
            cond: SelectedBranchCond::Value(*cond),
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

fn lower_leaf_to_helper(op: &LirLeafOp) -> bool {
    !matches!(
        op,
        LirLeafOp::I32Add
            | LirLeafOp::I32Sub
            | LirLeafOp::I32Mul
            | LirLeafOp::I32And
            | LirLeafOp::I32Or
            | LirLeafOp::I32Xor
            | LirLeafOp::I32Shl
            | LirLeafOp::I32ShrS
            | LirLeafOp::I32ShrU
            | LirLeafOp::I32Eq
            | LirLeafOp::I32Ne
            | LirLeafOp::I32LtS
            | LirLeafOp::I32LtU
            | LirLeafOp::I32GtS
            | LirLeafOp::I32GtU
            | LirLeafOp::I32LeS
            | LirLeafOp::I32LeU
            | LirLeafOp::I32GeS
            | LirLeafOp::I32GeU
            | LirLeafOp::I32Eqz
            | LirLeafOp::I32Clz
            | LirLeafOp::I32Ctz
            | LirLeafOp::I64Add
            | LirLeafOp::I64Sub
            | LirLeafOp::I64Mul
            | LirLeafOp::I64And
            | LirLeafOp::I64Or
            | LirLeafOp::I64Xor
            | LirLeafOp::I64Shl
            | LirLeafOp::I64ShrS
            | LirLeafOp::I64ShrU
            | LirLeafOp::I64Eq
            | LirLeafOp::I64Ne
            | LirLeafOp::I64LtS
            | LirLeafOp::I64LtU
            | LirLeafOp::I64GtS
            | LirLeafOp::I64GtU
            | LirLeafOp::I64LeS
            | LirLeafOp::I64LeU
            | LirLeafOp::I64GeS
            | LirLeafOp::I64GeU
            | LirLeafOp::I64Eqz
            | LirLeafOp::I64Clz
            | LirLeafOp::I64Ctz
            | LirLeafOp::I32WrapI64
            | LirLeafOp::I64ExtendI32S
            | LirLeafOp::I64ExtendI32U
            | LirLeafOp::I32Extend8S
            | LirLeafOp::I32Extend16S
            | LirLeafOp::I64Extend8S
            | LirLeafOp::I64Extend16S
            | LirLeafOp::I64Extend32S
            | LirLeafOp::Nop
            | LirLeafOp::Drop
    )
}

fn fuse_compare_branch(block: &mut SelectedBlock) {
    let SelectedTerminator::Branch {
        cond: SelectedBranchCond::Value(cond),
        ..
    } = &block.terminator
    else {
        return;
    };
    let cond_value = *cond;

    let Some(SelectedInst {
        kind:
            SelectedInstKind::Leaf {
                op,
                args,
                results,
            },
    }) = block.ops.last()
    else {
        return;
    };

    let [result] = results.as_slice() else {
        return;
    };
    if *result != cond_value {
        return;
    }

    let uses = block_use_counts(block);
    if uses.get(cond_value.0 as usize).copied().unwrap_or(0) != 1 {
        return;
    }

    let Some(fused) = fused_branch_cond(op, args) else {
        return;
    };

    block.ops.pop();
    if let SelectedTerminator::Branch { cond, .. } = &mut block.terminator {
        *cond = fused;
    }
}

fn fused_branch_cond(op: &LirLeafOp, args: &[LirValue]) -> Option<SelectedBranchCond> {
    let unary = |ctor: fn(LirValue) -> SelectedBranchCond| {
        let [value] = args else {
            return None;
        };
        Some(ctor(*value))
    };
    let binary = |kind: SelectedCompareKind,
                  ctor: fn(SelectedCompareKind, LirValue, LirValue) -> SelectedBranchCond| {
        let [lhs, rhs] = args else {
            return None;
        };
        Some(ctor(kind, *lhs, *rhs))
    };

    match op {
        LirLeafOp::I32Eqz => unary(SelectedBranchCond::I32Eqz),
        LirLeafOp::I64Eqz => unary(SelectedBranchCond::I64Eqz),
        LirLeafOp::I32Eq => binary(SelectedCompareKind::Eq, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32Ne => binary(SelectedCompareKind::Ne, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32LtS => binary(SelectedCompareKind::LtS, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32LtU => binary(SelectedCompareKind::LtU, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32GtS => binary(SelectedCompareKind::GtS, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32GtU => binary(SelectedCompareKind::GtU, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32LeS => binary(SelectedCompareKind::LeS, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32LeU => binary(SelectedCompareKind::LeU, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32GeS => binary(SelectedCompareKind::GeS, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I32GeU => binary(SelectedCompareKind::GeU, |kind, lhs, rhs| {
            SelectedBranchCond::I32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64Eq => binary(SelectedCompareKind::Eq, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64Ne => binary(SelectedCompareKind::Ne, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64LtS => binary(SelectedCompareKind::LtS, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64LtU => binary(SelectedCompareKind::LtU, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64GtS => binary(SelectedCompareKind::GtS, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64GtU => binary(SelectedCompareKind::GtU, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64LeS => binary(SelectedCompareKind::LeS, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64LeU => binary(SelectedCompareKind::LeU, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64GeS => binary(SelectedCompareKind::GeS, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::I64GeU => binary(SelectedCompareKind::GeU, |kind, lhs, rhs| {
            SelectedBranchCond::I64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F32Eq => binary(SelectedCompareKind::Eq, |kind, lhs, rhs| {
            SelectedBranchCond::F32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F32Ne => binary(SelectedCompareKind::Ne, |kind, lhs, rhs| {
            SelectedBranchCond::F32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F32Lt => binary(SelectedCompareKind::Lt, |kind, lhs, rhs| {
            SelectedBranchCond::F32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F32Gt => binary(SelectedCompareKind::Gt, |kind, lhs, rhs| {
            SelectedBranchCond::F32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F32Le => binary(SelectedCompareKind::Le, |kind, lhs, rhs| {
            SelectedBranchCond::F32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F32Ge => binary(SelectedCompareKind::Ge, |kind, lhs, rhs| {
            SelectedBranchCond::F32Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F64Eq => binary(SelectedCompareKind::Eq, |kind, lhs, rhs| {
            SelectedBranchCond::F64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F64Ne => binary(SelectedCompareKind::Ne, |kind, lhs, rhs| {
            SelectedBranchCond::F64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F64Lt => binary(SelectedCompareKind::Lt, |kind, lhs, rhs| {
            SelectedBranchCond::F64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F64Gt => binary(SelectedCompareKind::Gt, |kind, lhs, rhs| {
            SelectedBranchCond::F64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F64Le => binary(SelectedCompareKind::Le, |kind, lhs, rhs| {
            SelectedBranchCond::F64Compare { kind, lhs, rhs }
        }),
        LirLeafOp::F64Ge => binary(SelectedCompareKind::Ge, |kind, lhs, rhs| {
            SelectedBranchCond::F64Compare { kind, lhs, rhs }
        }),
        _ => None,
    }
}

fn block_use_counts(block: &SelectedBlock) -> Vec<u32> {
    let mut counts = Vec::<u32>::new();

    let mut mark = |value: LirValue| {
        let index = value.0 as usize;
        if index >= counts.len() {
            counts.resize(index + 1, 0);
        }
        counts[index] += 1;
    };

    for inst in &block.ops {
        match &inst.kind {
            SelectedInstKind::Copy { src, .. } => {
                if let SelectedSource::Value(value) = src {
                    mark(*value);
                }
            }
            SelectedInstKind::Leaf { args, .. }
            | SelectedInstKind::CallExternal { args, .. }
            | SelectedInstKind::CallLocal { args, .. } => {
                for value in args {
                    mark(*value);
                }
            }
            SelectedInstKind::Helper(SelectedHelperInst::Leaf { args, .. }) => {
                for value in args {
                    mark(*value);
                }
            }
            SelectedInstKind::CallIndirect { index, args, .. } => {
                mark(*index);
                for value in args {
                    mark(*value);
                }
            }
        }
    }

    match &block.terminator {
        SelectedTerminator::Goto(edge) => {
            for value in &edge.tos {
                mark(*value);
            }
        }
        SelectedTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            mark_branch_cond(cond, &mut mark);
            for value in &then_edge.tos {
                mark(*value);
            }
            for value in &else_edge.tos {
                mark(*value);
            }
        }
        SelectedTerminator::BrTable { index, entries } => {
            mark(*index);
            for edge in entries {
                for value in &edge.tos {
                    mark(*value);
                }
            }
        }
        SelectedTerminator::Return { values } => {
            for value in values {
                mark(*value);
            }
        }
        SelectedTerminator::TrapUnreachable => {}
    }

    counts
}

fn mark_branch_cond(cond: &SelectedBranchCond, mark: &mut impl FnMut(LirValue)) {
    match cond {
        SelectedBranchCond::Value(value)
        | SelectedBranchCond::I32Eqz(value)
        | SelectedBranchCond::I64Eqz(value) => mark(*value),
        SelectedBranchCond::I32Compare { lhs, rhs, .. }
        | SelectedBranchCond::I64Compare { lhs, rhs, .. }
        | SelectedBranchCond::F32Compare { lhs, rhs, .. }
        | SelectedBranchCond::F64Compare { lhs, rhs, .. } => {
            mark(*lhs);
            mark(*rhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        fuse_compare_branch, select_inst, SelectedBlock, SelectedBranchCond, SelectedCompareKind,
        SelectedEdge, SelectedHelperInst, SelectedInst, SelectedInstKind, SelectedSource,
        SelectedTarget, SelectedTerminator,
    };
    use crate::vm::lir::{
        ir::{LirInst, LirInstKind, LirValue},
        leaf::LirLeafOp,
    };
    use crate::vm::native::ir::{NativeBlockId, NativeHelperEffect};

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
            select_inst(&inst, &[]).unwrap().kind,
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
            select_inst(&inst, &[]).unwrap().kind,
            SelectedInstKind::Leaf {
                op: LirLeafOp::I32Add,
                args: alloc::vec![LirValue(0), LirValue(1)],
                results: alloc::vec![LirValue(2)],
            }
        );
    }

    #[test]
    fn lowers_trapping_leaf_to_helper() {
        let inst = LirInst {
            kind: LirInstKind::Leaf {
                op: LirLeafOp::I32DivS,
                args: alloc::vec![LirValue(1), LirValue(2)],
                results: alloc::vec![LirValue(3)],
            },
        };

        assert_eq!(
            select_inst(&inst, &[]).unwrap().kind,
            SelectedInstKind::Helper(SelectedHelperInst::Leaf {
                effect: NativeHelperEffect::Transparent,
                op: LirLeafOp::I32DivS,
                args: alloc::vec![LirValue(1), LirValue(2)],
                results: alloc::vec![LirValue(3)],
            })
        );
    }

    #[test]
    fn fuses_compare_branch_tail() {
        let mut block = SelectedBlock {
            id: NativeBlockId(0),
            tos_params: alloc::vec![],
            ops: alloc::vec![SelectedInst {
                kind: SelectedInstKind::Leaf {
                    op: LirLeafOp::I32Eq,
                    args: alloc::vec![LirValue(1), LirValue(2)],
                    results: alloc::vec![LirValue(3)],
                },
            }],
            terminator: SelectedTerminator::Branch {
                cond: SelectedBranchCond::Value(LirValue(3)),
                then_edge: SelectedEdge {
                    target: NativeBlockId(1),
                    tos: alloc::vec![],
                },
                else_edge: SelectedEdge {
                    target: NativeBlockId(2),
                    tos: alloc::vec![],
                },
            },
        };

        fuse_compare_branch(&mut block);

        assert!(block.ops.is_empty());
        assert_eq!(
            block.terminator,
            SelectedTerminator::Branch {
                cond: SelectedBranchCond::I32Compare {
                    kind: SelectedCompareKind::Eq,
                    lhs: LirValue(1),
                    rhs: LirValue(2),
                },
                then_edge: SelectedEdge {
                    target: NativeBlockId(1),
                    tos: alloc::vec![],
                },
                else_edge: SelectedEdge {
                    target: NativeBlockId(2),
                    tos: alloc::vec![],
                },
            }
        );
    }
}
