//! Backend-lowered IR types for the fast/fusion/native pipeline.
//!
//! `IrOp` is the instruction representation produced after backend planning has
//! already resolved stack management details such as rotating-window state,
//! spill/fill, and hot-local placement.
//!
//! This is not the long-term semantic frontend IR. It is the current lowered IR
//! consumed by backend resolution and finalization.

use alloc::vec::Vec;

pub use crate::vm::wasm::common::{BrTableEntry, OpIndex, SlotRef};
use crate::vm::wasm::{CoreOpKind, core_op::for_each_core_op};

/// A single resolved IR instruction.
///
/// Produced by the lowering pass after all stack management is resolved.
/// Consumed by the finalizer and JIT group compiler.
#[derive(Debug, Clone)]
pub struct IrOp {
    /// What this op does (semantic).
    pub kind: IrOpKind,
    /// Rotating T-window offset before this op executes.
    ///
    /// This is `stack_height % TOS_REGISTER_COUNT`, not a live-count. The
    /// frontend guarantees the rotating cache is valid; backends use this only
    /// to select concrete T registers for the op's stack effect.
    pub window: u8,
    /// Primary explicit operand/result slot for cold/helper-style ops.
    ///
    /// This is computed during lowering so backends do not need to reconstruct
    /// operand locations from abstract stack height.
    pub frame_slot: Option<SlotRef>,
    /// Next IR index (linear fallthrough).
    pub fallthrough: Option<OpIndex>,
    /// Branch/else target (IR index).
    pub alt_target: Option<OpIndex>,
    /// Whether this instruction encodes a target field that needs pointer patching.
    pub has_target: bool,
}

/// Backend-lowered opcode — still handler/backend agnostic, but no longer
/// purely semantic.
///
/// Hot-local placement, `InitLocals`, and explicit spill/fill ops are already
/// committed at this stage.
#[derive(Debug, Clone)]
pub enum IrOpKind {
    // =========================================================================
    // i32 binary (pop 2, push 1)
    // =========================================================================
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,

    // =========================================================================
    // i64 binary (pop 2, push 1)
    // =========================================================================
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,

    // =========================================================================
    // f32 binary (pop 2, push 1)
    // =========================================================================
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,

    // =========================================================================
    // f64 binary (pop 2, push 1)
    // =========================================================================
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,

    // =========================================================================
    // i32 comparisons (pop 2, push 1)
    // =========================================================================
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,

    // =========================================================================
    // i64 comparisons (pop 2, push 1)
    // =========================================================================
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,

    // =========================================================================
    // f32 comparisons (pop 2, push 1)
    // =========================================================================
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,

    // =========================================================================
    // f64 comparisons (pop 2, push 1)
    // =========================================================================
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,

    // =========================================================================
    // i32 unary (pop 1, push 1)
    // =========================================================================
    I32Eqz,
    I32Clz,
    I32Ctz,
    I32Popcnt,

    // =========================================================================
    // i64 unary (pop 1, push 1)
    // =========================================================================
    I64Eqz,
    I64Clz,
    I64Ctz,
    I64Popcnt,

    // =========================================================================
    // f32 unary (pop 1, push 1)
    // =========================================================================
    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F32Sqrt,

    // =========================================================================
    // f64 unary (pop 1, push 1)
    // =========================================================================
    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Sqrt,

    // =========================================================================
    // Conversions (pop 1, push 1)
    // =========================================================================
    I32WrapI64,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64ExtendI32S,
    I64ExtendI32U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F32DemoteF64,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F64PromoteF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,

    // Sign extension (pop 1, push 1)
    I32Extend8S,
    I32Extend16S,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,

    // Saturating truncation (pop 1, push 1)
    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,

    // =========================================================================
    // Constants (push 1)
    // =========================================================================
    I32Const { value: u32 },
    I64Const { value: u64 },
    F32Const { value: u32 },   // stored as bits
    F64Const { value: u64 },   // stored as bits

    // =========================================================================
    // Locals — hot vs frame already resolved during lowering
    // =========================================================================
    LocalGetHot { reg: u8 },      // 0=l0, 1=l1, 2=l2
    LocalSetHot { reg: u8 },
    LocalTeeHot { reg: u8 },
    LocalGetFrame { idx: u16 },   // remapped frame index
    LocalSetFrame { idx: u16 },
    LocalTeeFrame { idx: u16 },

    // =========================================================================
    // TOS management — inserted during lowering
    // =========================================================================
    Spill { slot: u16, count: u8 },
    Fill { slot: u16, count: u8 },

    // =========================================================================
    // Memory loads (pop 1, push 1)
    // =========================================================================
    I32Load { offset: u32, memidx: u32 },
    I64Load { offset: u32, memidx: u32 },
    F32Load { offset: u32, memidx: u32 },
    F64Load { offset: u32, memidx: u32 },
    I32Load8S { offset: u32, memidx: u32 },
    I32Load8U { offset: u32, memidx: u32 },
    I32Load16S { offset: u32, memidx: u32 },
    I32Load16U { offset: u32, memidx: u32 },
    I64Load8S { offset: u32, memidx: u32 },
    I64Load8U { offset: u32, memidx: u32 },
    I64Load16S { offset: u32, memidx: u32 },
    I64Load16U { offset: u32, memidx: u32 },
    I64Load32S { offset: u32, memidx: u32 },
    I64Load32U { offset: u32, memidx: u32 },

    // =========================================================================
    // Memory stores (pop 2, push 0)
    // =========================================================================
    I32Store { offset: u32, memidx: u32 },
    I64Store { offset: u32, memidx: u32 },
    F32Store { offset: u32, memidx: u32 },
    F64Store { offset: u32, memidx: u32 },
    I32Store8 { offset: u32, memidx: u32 },
    I32Store16 { offset: u32, memidx: u32 },
    I64Store8 { offset: u32, memidx: u32 },
    I64Store16 { offset: u32, memidx: u32 },
    I64Store32 { offset: u32, memidx: u32 },

    // =========================================================================
    // Globals
    // =========================================================================
    GlobalGet { idx: u32 },   // push 1
    GlobalSet { idx: u32 },   // pop 1

    // =========================================================================
    // Memory size/grow
    // =========================================================================
    MemorySize { mem_idx: u32 },  // push 1
    MemoryGrow { mem_idx: u32 },  // pop 1, push 1

    // =========================================================================
    // Bulk memory (ternary: pop 3, except DataDrop: pop 0)
    // =========================================================================
    MemoryFill { imm0: u32, imm1: u32 },
    MemoryCopy { imm0: u32, imm1: u32 },
    MemoryInit { imm0: u32, imm1: u32 },
    DataDrop { data_idx: u32 },

    // =========================================================================
    // Tables
    // =========================================================================
    TableGet { table_idx: u32 },              // pop 1, push 1
    TableSet { table_idx: u32 },              // pop 2
    TableSize { table_idx: u32 },             // push 1
    TableGrow { table_idx: u32 },             // pop 2, push 1
    TableFill { imm0: u32, imm1: u32 },       // pop 3
    TableCopy { imm0: u32, imm1: u32 },       // pop 3
    TableInit { imm0: u32, imm1: u32 },       // pop 3
    ElemDrop { elem_idx: u32 },               // pop 0

    // =========================================================================
    // References
    // =========================================================================
    RefNull,                        // push 1
    RefIsNull,                      // pop 1, push 1
    RefFunc { func_idx: u32 },      // push 1

    // =========================================================================
    // Stack ops
    // =========================================================================
    Drop,                           // pop 1
    Select,                         // pop 3, push 1

    // =========================================================================
    // Control flow
    // =========================================================================
    Block,   // structural, removed by finalizer
    Loop,    // structural, removed by finalizer
    If,      // pop 1 (condition)
    Else,    // unconditional jump
    End,     // structural, removed by finalizer
    Br { stack_drop: u32, arity: u16 },
    BrIfSimple,  // pop 1
    BrIf { stack_drop: u32, arity: u16 },  // pop 1
    BrTable { entries: Vec<BrTableEntry>, entry_count: u32, data_slot_count: u32 }, // pop 1

    // =========================================================================
    // Calls
    // =========================================================================
    CallExternal { func_idx: u32, delta: SlotRef },
    CallInternal { callee: u64, delta: SlotRef },
    CallIndirect { type_idx: u32, table_idx: u32, delta: SlotRef },

    // =========================================================================
    // Returns
    // =========================================================================
    ReturnVoid { frame_size: u16 },
    ReturnOne { frame_size: u16 },
    Return { arity: u16, frame_size: u16 },
    Unreachable,

    // =========================================================================
    // Prologue
    // =========================================================================
    InitLocals { k0: u16, k1: u16, k2: u16 },

    // =========================================================================
    // NOP (structural, removed by finalizer)
    // =========================================================================
    Nop,

    // =========================================================================
    // Terminal / pseudo
    // =========================================================================
    Term,
    Data { imm0: u64, imm1: u64, imm2: u64 },

    // =========================================================================
    // Fused (pre-encoded by fusion backend)
    // =========================================================================
    /// Pre-encoded fused instruction. `target_imm` indicates which slot (0/1/2)
    /// holds the branch target pointer, or 0xFF if none.
    Fused { imm0: u64, imm1: u64, imm2: u64, target_imm: u8 },
}

pub const TOS_REGISTER_COUNT: usize = 4;

#[inline]
pub const fn window_from_height(height: usize) -> u8 {
    (height % TOS_REGISTER_COUNT) as u8
}

#[inline]
pub const fn current_variant_from_window(window: u8) -> u8 {
    (((window as usize + TOS_REGISTER_COUNT - 1) % TOS_REGISTER_COUNT) + 1) as u8
}

#[inline]
pub const fn push_variant_from_window(window: u8) -> u8 {
    (window as usize % TOS_REGISTER_COUNT + 1) as u8
}

macro_rules! impl_from_core_op {
    ($(
        $name:ident $( { $($field:ident : $ty:ty),* $(,)? } )? => ($pops:expr, $pushes:expr),
    )* ) => {
        impl From<CoreOpKind> for IrOpKind {
            fn from(kind: CoreOpKind) -> Self {
                match kind {
                    $( CoreOpKind::$name $( { $($field),* } )? => IrOpKind::$name $( { $($field),* } )?, )*
                }
            }
        }
    };
}

for_each_core_op!(impl_from_core_op);

/// Canonical stack effect: (pops, pushes) for every `IrOpKind`.
///
/// Single source of truth — replaces `op_stack_effect()` in group.rs,
/// implicit knowledge in dispatch.rs, and `get_pop_push()` in op_classify.rs.
pub fn stack_effect(kind: &IrOpKind) -> (u8, u8) {
    use IrOpKind::*;
    match kind {
        // Binary ops + comparisons: pop 2, push 1
        I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU |
        I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |
        I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU |
        I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr |
        F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign |
        F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign |
        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
        F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge |
        F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => (2, 1),

        // Unary + conversions: pop 1, push 1
        I32Eqz | I32Clz | I32Ctz | I32Popcnt |
        I64Eqz | I64Clz | I64Ctz | I64Popcnt |
        F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt |
        F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt |
        I32WrapI64 |
        I32TruncF32S | I32TruncF32U | I32TruncF64S | I32TruncF64U |
        I64ExtendI32S | I64ExtendI32U |
        I64TruncF32S | I64TruncF32U | I64TruncF64S | I64TruncF64U |
        F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U | F32DemoteF64 |
        F64ConvertI32S | F64ConvertI32U | F64ConvertI64S | F64ConvertI64U | F64PromoteF32 |
        I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 | F64ReinterpretI64 |
        I32Extend8S | I32Extend16S | I64Extend8S | I64Extend16S | I64Extend32S |
        I32TruncSatF32S | I32TruncSatF32U | I32TruncSatF64S | I32TruncSatF64U |
        I64TruncSatF32S | I64TruncSatF32U | I64TruncSatF64S | I64TruncSatF64U |
        RefIsNull => (1, 1),

        // Memory loads: pop 1, push 1
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } => (1, 1),

        // Memory stores: pop 2, push 0
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => (2, 0),

        // Constants: push 1
        I32Const { .. } | I64Const { .. } | F32Const { .. } | F64Const { .. } => (0, 1),

        // Local get (hot or frame): push 1
        LocalGetHot { .. } | LocalGetFrame { .. } => (0, 1),
        // Local set (hot or frame): pop 1
        LocalSetHot { .. } | LocalSetFrame { .. } => (1, 0),
        // Local tee: pop 0, push 0 (reads TOS, writes local, no height change)
        LocalTeeHot { .. } | LocalTeeFrame { .. } => (0, 0),

        // Spill/fill: no stack effect (TOS register management)
        Spill { .. } | Fill { .. } => (0, 0),

        // Globals, memory size, table size, references: push 1
        GlobalGet { .. } | MemorySize { .. } | TableSize { .. } |
        RefNull | RefFunc { .. } => (0, 1),
        // Global set: pop 1
        GlobalSet { .. } => (1, 0),

        // Memory grow: pop 1 (pages), push 1 (result)
        MemoryGrow { .. } => (1, 1),

        // Table ops
        TableGet { .. } => (1, 1),
        TableSet { .. } => (2, 0),
        TableGrow { .. } => (2, 1),

        // Ternary bulk ops: pop 3
        MemoryFill { .. } | MemoryCopy { .. } | MemoryInit { .. } |
        TableFill { .. } | TableCopy { .. } | TableInit { .. } => (3, 0),

        // Drop-like (no stack operands): pop 0
        DataDrop { .. } | ElemDrop { .. } => (0, 0),

        // Stack ops
        Drop => (1, 0),
        Select => (3, 1),

        // Control flow: condition pop
        If | BrIfSimple | BrIf { .. } | BrTable { .. } => (1, 0),
        // Unconditional branch / structural: no pop
        Br { .. } | Else => (0, 0),

        // Calls — variable, not tracked here. Actual effect applied by lowering
        // via stack.apply_call().
        CallExternal { .. } | CallInternal { .. } | CallIndirect { .. } => (0, 0),

        // Returns — terminal, no stack effect tracked
        ReturnVoid { .. } | ReturnOne { .. } | Return { .. } | Unreachable => (0, 0),

        // Structural / pseudo — no stack effect
        Block | Loop | End | Nop | InitLocals { .. } | Term | Data { .. } | Fused { .. } => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binop_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::I32Add), (2, 1));
        assert_eq!(stack_effect(&IrOpKind::I64Mul), (2, 1));
        assert_eq!(stack_effect(&IrOpKind::F64Div), (2, 1));
    }

    #[test]
    fn test_unop_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::I32Eqz), (1, 1));
        assert_eq!(stack_effect(&IrOpKind::F64Sqrt), (1, 1));
        assert_eq!(stack_effect(&IrOpKind::I32WrapI64), (1, 1));
    }

    #[test]
    fn test_const_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::I32Const { value: 42 }), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::I64Const { value: 0 }), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::F32Const { value: 0 }), (0, 1));
    }

    #[test]
    fn test_local_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::LocalGetHot { reg: 0 }), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::LocalSetHot { reg: 1 }), (1, 0));
        assert_eq!(stack_effect(&IrOpKind::LocalTeeHot { reg: 0 }), (0, 0));
        assert_eq!(stack_effect(&IrOpKind::LocalGetFrame { idx: 5 }), (0, 1));
    }

    #[test]
    fn test_memory_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::I32Load { offset: 0, memidx: 0 }), (1, 1));
        assert_eq!(stack_effect(&IrOpKind::I32Store { offset: 0, memidx: 0 }), (2, 0));
        assert_eq!(stack_effect(&IrOpKind::MemorySize { mem_idx: 0 }), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::MemoryGrow { mem_idx: 0 }), (1, 1));
    }

    #[test]
    fn test_spill_fill_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::Spill { slot: 0, count: 1 }), (0, 0));
        assert_eq!(stack_effect(&IrOpKind::Fill { slot: 0, count: 1 }), (0, 0));
    }

    #[test]
    fn test_table_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::TableGet { table_idx: 0 }), (1, 1));
        assert_eq!(stack_effect(&IrOpKind::TableSet { table_idx: 0 }), (2, 0));
        assert_eq!(stack_effect(&IrOpKind::TableSize { table_idx: 0 }), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::TableGrow { table_idx: 0 }), (2, 1));
    }

    #[test]
    fn test_select_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::Select), (3, 1));
        assert_eq!(stack_effect(&IrOpKind::Drop), (1, 0));
    }

    #[test]
    fn test_control_flow_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::If), (1, 0));
        assert_eq!(stack_effect(&IrOpKind::BrIfSimple), (1, 0));
        assert_eq!(stack_effect(&IrOpKind::BrIf { stack_drop: 0, arity: 0 }), (1, 0));
        assert_eq!(stack_effect(&IrOpKind::BrTable { entries: Vec::new(), entry_count: 0, data_slot_count: 0 }), (1, 0));
        assert_eq!(stack_effect(&IrOpKind::Br { stack_drop: 0, arity: 0 }), (0, 0));
        assert_eq!(stack_effect(&IrOpKind::Else), (0, 0));
        assert_eq!(stack_effect(&IrOpKind::Block), (0, 0));
    }

    #[test]
    fn test_slot_ref_resolution() {
        assert_eq!(SlotRef::absolute(7).resolve(32), 7);
        assert_eq!(SlotRef::operand_relative(3).resolve(32), 35);
    }

    #[test]
    fn test_bulk_memory_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::MemoryFill { imm0: 0, imm1: 0 }), (3, 0));
        assert_eq!(stack_effect(&IrOpKind::MemoryCopy { imm0: 0, imm1: 0 }), (3, 0));
        assert_eq!(stack_effect(&IrOpKind::DataDrop { data_idx: 0 }), (0, 0));
    }

    #[test]
    fn test_reference_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::RefNull), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::RefIsNull), (1, 1));
        assert_eq!(stack_effect(&IrOpKind::RefFunc { func_idx: 0 }), (0, 1));
    }
}
