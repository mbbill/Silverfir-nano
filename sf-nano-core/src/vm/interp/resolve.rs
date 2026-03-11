//! Resolve interpreter ops to handler entry points.

use crate::vm::{
    interp::{
        handler_lookup,
        handlers::{full_set::*, OpHandler},
        TOS_REGISTER_COUNT,
    },
    lir::leaf::LirLeafOp,
    wasm::primitive_op::for_each_primitive_op,
};

type Handler = OpHandler;

macro_rules! define_resolve_kind {
    ($(
        $name:ident $( { $($field:ident : $ty:ty),* $(,)? } )? => ($pops:expr, $pushes:expr),
    )* ) => {
        #[derive(Clone, Debug)]
        pub(crate) enum IrOpKind {
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
            BrTable { entry_count: u32, data_slot_count: u32 },
            CallExternal { func_idx: u32, delta: u16 },
            CallInternal { callee: u64, delta: u16 },
            CallIndirect { type_idx: u32, table_idx: u32, delta: u16, index_slot: u16 },
            ReturnVoid { frame_size: u16 },
            ReturnOne { frame_size: u16, result_slot: u16 },
            Return { arity: u16, frame_size: u16, result_slot: u16 },
            // Retained for the future hot-local-register / fusion path.
            InitLocals { k0: u16, k1: u16, k2: u16 },
            Term,
            Data { imm0: u64, imm1: u64, imm2: u64 },
            Fused { imm0: u64, imm1: u64, imm2: u64, target_imm: u8 },
        }

        impl From<&LirLeafOp> for IrOpKind {
            fn from(kind: &LirLeafOp) -> Self {
                match kind {
                    $( LirLeafOp::$name $( { $($field),* } )? => Self::$name $( { $($field : *$field),* } )?, )*
                }
            }
        }

        pub(crate) fn leaf_stack_effect(kind: &LirLeafOp) -> (u8, u8) {
            match kind {
                $( LirLeafOp::$name $( { $($field: _),* } )? => ($pops, $pushes), )*
            }
        }
    };
}

for_each_primitive_op!(define_resolve_kind);

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_ir_resolve.rs"));

#[inline]
pub(crate) fn resolve_handler_for_kind(kind: &IrOpKind, variant: u8) -> OpHandler {
    resolve_handler(kind, variant)
}

#[inline]
pub(crate) fn variant_for_depth(depth: usize) -> u8 {
    debug_assert!(depth > 0);
    (((depth - 1) % TOS_REGISTER_COUNT) + 1) as u8
}

#[inline]
pub(crate) fn variant_for_leaf(op: &LirLeafOp, depth_before: usize) -> u8 {
    let (pop, push) = leaf_stack_effect(op);
    if pop == 0 && push > 0 {
        variant_for_depth(depth_before + push as usize)
    } else if depth_before == 0 {
        0
    } else {
        variant_for_depth(depth_before)
    }
}
