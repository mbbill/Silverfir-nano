//! Lower shared CFG + SSA LIR into target-independent native IR.
//!
//! This is the single native-owned semantic lowering step after LIR.
//! It owns:
//! - native virtual registers
//! - explicit successor copies
//! - direct native-facing call/control shapes
//!
//! It must not reintroduce stack height or rotating-window semantics.

use alloc::vec::Vec;

use crate::vm::{
    backend::BackendConfig,
    lir::ir::{LirEdge, LirInstKind, LirProgram, LirTerminator, LirValue},
    plan::PlannedProgram,
};

use super::ir::{
    NativeAbi, NativeBlock, NativeBlockId, NativeCopy, NativeEdge, NativeInst, NativeInstKind,
    NativeProgram, NativeReg, NativeTerminator,
};

pub fn lower_native(
    lir: &LirProgram,
    planned: &PlannedProgram,
    backend_config: BackendConfig,
) -> NativeProgram {
    NativeProgram {
        entry: NativeBlockId::from(lir.entry),
        blocks: lir
            .blocks
            .iter()
            .map(|block| NativeBlock {
                id: NativeBlockId::from(block.id),
                params: block.params.tos.iter().copied().map(native_reg).collect(),
                ops: block.ops.iter().map(lower_inst).collect(),
                terminator: lower_terminator(lir, &block.terminator),
            })
            .collect(),
        abi: NativeAbi::from(backend_config),
        frame: planned.frame,
        groups: planned.groups.clone(),
        hot_locals: planned.hot_locals.clone(),
    }
}

fn lower_inst(inst: &crate::vm::lir::ir::LirInst) -> NativeInst {
    let kind = match &inst.kind {
        LirInstKind::Leaf { op, args, results } => NativeInstKind::Leaf {
            op: op.clone(),
            args: args.iter().copied().map(native_reg).collect(),
            results: results.iter().copied().map(native_reg).collect(),
        },
        LirInstKind::WriteOperandSlot { slot, src } => NativeInstKind::WriteOperandSlot {
            slot: *slot,
            src: native_reg(*src),
        },
        LirInstKind::ReadOperandSlot { slot, dst } => NativeInstKind::ReadOperandSlot {
            slot: *slot,
            dst: native_reg(*dst),
        },
        LirInstKind::ReadHotLocal { reg, dst } => NativeInstKind::ReadHotLocal {
            reg: *reg,
            dst: native_reg(*dst),
        },
        LirInstKind::WriteHotLocal { reg, src } => NativeInstKind::WriteHotLocal {
            reg: *reg,
            src: native_reg(*src),
        },
        LirInstKind::ReadFrameLocal { frame_slot, dst } => NativeInstKind::ReadFrameLocal {
            frame_slot: *frame_slot,
            dst: native_reg(*dst),
        },
        LirInstKind::WriteFrameLocal { frame_slot, src } => NativeInstKind::WriteFrameLocal {
            frame_slot: *frame_slot,
            src: native_reg(*src),
        },
        LirInstKind::CallExternal {
            func_idx,
            args,
            results,
        } => NativeInstKind::CallExternal {
            func_idx: *func_idx,
            args: args.iter().copied().map(native_reg).collect(),
            results: results.iter().copied().map(native_reg).collect(),
        },
        LirInstKind::CallInternal {
            callee,
            args,
            results,
        } => NativeInstKind::CallInternal {
            callee: *callee,
            args: args.iter().copied().map(native_reg).collect(),
            results: results.iter().copied().map(native_reg).collect(),
        },
        LirInstKind::CallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            results,
        } => NativeInstKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            index: native_reg(*index),
            args: args.iter().copied().map(native_reg).collect(),
            results: results.iter().copied().map(native_reg).collect(),
        },
    };
    NativeInst { kind }
}

fn lower_terminator(lir: &LirProgram, term: &LirTerminator) -> NativeTerminator {
    match term {
        LirTerminator::Goto(edge) => NativeTerminator::Goto(lower_edge(lir, edge)),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => NativeTerminator::Branch {
            cond: native_reg(*cond),
            then_edge: lower_edge(lir, then_edge),
            else_edge: lower_edge(lir, else_edge),
        },
        LirTerminator::BrTable { index, entries } => NativeTerminator::BrTable {
            index: native_reg(*index),
            entries: entries.iter().map(|entry| lower_edge(lir, entry)).collect(),
        },
        LirTerminator::Return { values } => NativeTerminator::Return {
            values: values.iter().copied().map(native_reg).collect(),
        },
        LirTerminator::TrapUnreachable => NativeTerminator::TrapUnreachable,
    }
}

fn lower_edge(lir: &LirProgram, edge: &LirEdge) -> NativeEdge {
    let target = &lir.blocks[edge.target.as_usize()];
    let copies = edge
        .tos
        .iter()
        .zip(target.params.tos.iter())
        .filter_map(|(src, dst)| {
            let src = native_reg(*src);
            let dst = native_reg(*dst);
            (src != dst).then_some(NativeCopy { src, dst })
        })
        .collect::<Vec<_>>();

    NativeEdge {
        target: NativeBlockId::from(edge.target),
        copies,
    }
}

#[inline]
fn native_reg(value: LirValue) -> NativeReg {
    NativeReg(value.0)
}
