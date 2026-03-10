//! Resolve shared LIR ops to fast-interpreter handlers.

use alloc::vec::Vec;

use crate::vm::{
    interp::fast::{
        handler_lookup,
        handlers::{full_set::*, OpHandler},
    },
    lir::legacy::ir::{LirMarkerKind, LirOp, LirOpKind},
    plan::{
        frame::FrameSpan,
        spill::{CacheTransferDirection, SpillArtifact},
    },
    wasm::{
        common::BrTableEntry,
        core_op::{for_each_core_op, CoreOpKind},
    },
};

type Handler = OpHandler;

macro_rules! define_resolve_kind {
    ($(
        $name:ident $( { $($field:ident : $ty:ty),* $(,)? } )? => ($pops:expr, $pushes:expr),
    )* ) => {
        #[derive(Clone, Debug)]
        enum IrOpKind {
            $( $name $( { $($field : $ty),* } )?, )*
            LocalGetHot { reg: u8 },
            LocalSetHot { reg: u8 },
            LocalTeeHot { reg: u8 },
            LocalGetFrame { idx: u16 },
            LocalSetFrame { idx: u16 },
            LocalTeeFrame { idx: u16 },
            Spill { slot: u16, count: u8 },
            Fill { slot: u16, count: u8 },
            Block,
            Loop,
            If,
            Else,
            End,
            Br { has_fixup: bool },
            BrIfSimple,
            BrIf { has_fixup: bool },
            BrTable {
                entry_count: u32,
                data_slot_count: u32,
            },
            CallExternal { func_idx: u32, delta: u16 },
            CallInternal { callee: u64, delta: u16 },
            CallIndirect { type_idx: u32, table_idx: u32, delta: u16, index_slot: u16 },
            ReturnVoid { frame_size: u16 },
            ReturnOne { frame_size: u16, result_slot: u16 },
            Return { arity: u16, frame_size: u16, result_slot: u16 },
            InitLocals { k0: u16, k1: u16, k2: u16 },
            Term,
            Data { imm0: u64, imm1: u64, imm2: u64 },
            Fused { imm0: u64, imm1: u64, imm2: u64, target_imm: u8 },
        }

        impl From<&CoreOpKind> for IrOpKind {
            fn from(kind: &CoreOpKind) -> Self {
                match kind {
                    $( CoreOpKind::$name $( { $($field),* } )? => IrOpKind::$name $( { $($field : *$field),* } )?, )*
                }
            }
        }
    };
}

for_each_core_op!(define_resolve_kind);

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_ir_resolve.rs"));

pub fn resolve_handler_for_op(kind: &LirOpKind, variant: u8) -> OpHandler {
    let lifted = lift_kind(kind);
    resolve_handler(&lifted, variant)
}

pub fn handler_variant(op: &LirOp) -> u8 {
    let (pop, push) = op.kind.stack_effect();
    if pop == 0 && push > 0 {
        push_variant(op.window.0)
    } else {
        current_variant(op.window.0)
    }
}

fn lift_kind(kind: &LirOpKind) -> IrOpKind {
    match kind {
        LirOpKind::Core(kind) => kind.into(),
        LirOpKind::InitLocals { l0, l1, l2 } => IrOpKind::InitLocals {
            k0: l0.0,
            k1: l1.0,
            k2: l2.0,
        },
        LirOpKind::CacheTransfer(SpillArtifact::CacheTransfer(transfer)) => {
            let slot = transfer.frame.end().0.saturating_sub(1);
            match transfer.direction {
                CacheTransferDirection::Spill => IrOpKind::Spill {
                    slot,
                    count: transfer.frame.count as u8,
                },
                CacheTransferDirection::Fill => IrOpKind::Fill {
                    slot,
                    count: transfer.frame.count as u8,
                },
            }
        }
        LirOpKind::LocalGetHot { reg } => IrOpKind::LocalGetHot { reg: *reg },
        LirOpKind::LocalSetHot { reg } => IrOpKind::LocalSetHot { reg: *reg },
        LirOpKind::LocalTeeHot { reg } => IrOpKind::LocalTeeHot { reg: *reg },
        LirOpKind::LocalGetFrame { frame_slot } => IrOpKind::LocalGetFrame { idx: frame_slot.0 },
        LirOpKind::LocalSetFrame { frame_slot } => IrOpKind::LocalSetFrame { idx: frame_slot.0 },
        LirOpKind::LocalTeeFrame { frame_slot } => IrOpKind::LocalTeeFrame { idx: frame_slot.0 },
        LirOpKind::Marker(LirMarkerKind::Block { .. }) => IrOpKind::Block,
        LirOpKind::Marker(LirMarkerKind::Loop { .. }) => IrOpKind::Loop,
        LirOpKind::Marker(LirMarkerKind::If { .. }) => IrOpKind::If,
        LirOpKind::Marker(LirMarkerKind::Else) => IrOpKind::Else,
        LirOpKind::Marker(LirMarkerKind::End) => IrOpKind::End,
        LirOpKind::Branch {
            kind: crate::vm::plan::plan::PlannedBranchKind::Br,
            fixup,
            ..
        } => IrOpKind::Br {
            has_fixup: fixup.is_some(),
        },
        LirOpKind::Branch {
            kind: crate::vm::plan::plan::PlannedBranchKind::BrIf,
            fixup,
            ..
        } => {
            if fixup.is_none() {
                IrOpKind::BrIfSimple
            } else {
                IrOpKind::BrIf { has_fixup: true }
            }
        }
        LirOpKind::BrTable { entries, .. } => IrOpKind::BrTable {
            entry_count: entries.len() as u32,
            data_slot_count: entries.len() as u32,
        },
        LirOpKind::CallExternal { func_idx, args, .. } => IrOpKind::CallExternal {
            func_idx: *func_idx,
            delta: args.start.0,
        },
        LirOpKind::CallInternal { callee, args, .. } => IrOpKind::CallInternal {
            callee: *callee as u64,
            delta: args.start.0,
        },
        LirOpKind::CallIndirect {
            type_idx,
            table_idx,
            args,
            index_slot,
            ..
        } => IrOpKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            delta: args.start.0,
            index_slot: index_slot.0,
        },
        LirOpKind::Return { results: None } => IrOpKind::ReturnVoid { frame_size: 0 },
        LirOpKind::Return {
            results: Some(results),
        } if results.count == 1 => IrOpKind::ReturnOne {
            frame_size: 0,
            result_slot: results.start.0,
        },
        LirOpKind::Return {
            results: Some(results),
        } => IrOpKind::Return {
            arity: results.count,
            frame_size: 0,
            result_slot: results.start.0,
        },
    }
}

#[inline]
fn current_variant(window: u8) -> u8 {
    (((window as usize + 3) % 4) + 1) as u8
}

#[inline]
fn push_variant(window: u8) -> u8 {
    ((window as usize % 4) + 1) as u8
}
