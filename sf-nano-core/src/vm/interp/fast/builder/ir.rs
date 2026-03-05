//! Neutral IR types for the unified interpreter/fusion/JIT pipeline.
//!
//! `IrOp` is the single instruction representation produced by lowering (ir_lower.rs)
//! and consumed by all backends (interpreter, fusion, JIT) via ir_backend.rs.
//! Stack management (variants, spill/fill, hot locals) is resolved once during lowering.

use alloc::vec::Vec;

/// A single resolved IR instruction.
///
/// Produced by the lowering pass after all stack management is resolved.
/// Consumed by the unified backend to produce `TempInst` for the finalizer.
#[derive(Debug, Clone)]
pub struct IrOp {
    /// What this op does (semantic).
    pub kind: IrOpKind,
    /// D-variant (1-4), resolved by lowering. 0 = N/A (structural ops).
    pub variant: u8,
    /// Stack height before this op executes.
    pub pre_height: u16,
    /// Next IR index (linear fallthrough).
    pub fallthrough: Option<usize>,
    /// Branch/else target (IR index).
    pub alt_target: Option<usize>,
    /// Whether this instruction encodes a target field that needs pointer patching.
    pub has_target: bool,
}

/// Purely semantic opcode — no handler pointers, no encoding concerns.
///
/// Every variant that `dispatch.rs` handles is represented here. Hot locals are
/// distinguished from frame locals. Spill/fill are first-class ops.
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
    F32Const { value: u32 },
    F64Const { value: u64 },

    // =========================================================================
    // Locals — hot vs frame already resolved during lowering
    // =========================================================================
    LocalGetHot { reg: u8 },      // l0=0, l1=1, l2=2
    LocalSetHot { reg: u8 },
    LocalTeeHot { reg: u8 },
    LocalGetFrame { idx: u16 },   // remapped frame index
    LocalSetFrame { idx: u16 },
    LocalTeeFrame { idx: u16 },

    // =========================================================================
    // TOS management — explicit, inserted during lowering
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
    // Memory size/grow (push 1 / pop 1 push 1)
    // =========================================================================
    MemorySize { mem_idx: u32 },
    MemoryGrow { mem_idx: u32 },

    // =========================================================================
    // Bulk memory (pop 3, push 0)
    // =========================================================================
    MemoryFill { mem_idx: u32 },
    MemoryCopy { dst_idx: u32, src_idx: u32 },
    MemoryInit { mem_idx: u32, data_idx: u32 },
    DataDrop { data_idx: u32 },

    // =========================================================================
    // Globals
    // =========================================================================
    GlobalGet { idx: u32 },
    GlobalSet { idx: u32 },

    // =========================================================================
    // Tables
    // =========================================================================
    TableGet { table_idx: u32 },
    TableSet { table_idx: u32 },
    TableSize { table_idx: u32 },
    TableGrow { table_idx: u32 },
    TableFill { table_idx: u32 },
    TableCopy { dst_idx: u32, src_idx: u32 },
    TableInit { table_idx: u32, elem_idx: u32 },
    ElemDrop { elem_idx: u32 },

    // =========================================================================
    // References
    // =========================================================================
    RefNull,
    RefIsNull,
    RefFunc { func_idx: u32 },

    // =========================================================================
    // Stack ops
    // =========================================================================
    Drop,
    Select,

    // =========================================================================
    // Control flow
    // =========================================================================
    Block,
    Loop,
    If,
    Else,
    End,
    Br {
        stack_drop: usize,
        arity: usize,
        height: usize,
        operand_base_offset: u32,
    },
    BrIfSimple,
    BrIf {
        stack_drop: usize,
        arity: usize,
        height: usize,
        operand_base_offset: u32,
    },
    BrTable {
        entries: Vec<BrTableEntry>,
        height: usize,
        operand_base_offset: u32,
    },

    // =========================================================================
    // Calls
    // =========================================================================
    CallExternal { func_idx: u32, delta: usize },
    CallInternal { callee: u64, delta: usize },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        delta: usize,
        operand_base_offset: u32,
        height: u16,
    },

    // =========================================================================
    // Returns
    // =========================================================================
    ReturnVoid { frame_size: usize },
    ReturnOne {
        frame_size: usize,
        operand_base_offset: u32,
        height: usize,
    },
    Return {
        arity: usize,
        frame_size: usize,
        operand_base_offset: u32,
        height: usize,
    },
    Unreachable,

    // =========================================================================
    // Prologue
    // =========================================================================
    InitLocals { k0: u16, k1: u16, k2: u16 },

    // =========================================================================
    // Structural (removed during finalization)
    // =========================================================================
    Nop,

    // =========================================================================
    // Terminal / pseudo
    // =========================================================================
    Term,
    Data { imm0: u64, imm1: u64, imm2: u64 },
}

/// Entry for br_table: target info for each label.
#[derive(Debug, Clone)]
pub struct BrTableEntry {
    /// Target instruction index (None for forward refs).
    pub target_idx: Option<usize>,
    /// Stack offset for this branch.
    pub stack_offset: usize,
    /// Branch arity.
    pub arity: usize,
}

/// Canonical stack effect: (pops, pushes) for every `IrOpKind`.
///
/// Single source of truth — replaces `op_stack_effect()` in group.rs,
/// implicit knowledge in dispatch.rs, and `get_pop_push()` in op_classify.rs.
pub fn stack_effect(kind: &IrOpKind) -> (u8, u8) {
    use IrOpKind::*;
    match kind {
        // Binary ops: pop 2, push 1
        I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU |
        I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |
        I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU |
        I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr |
        F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign |
        F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign => (2, 1),

        // Comparisons: pop 2, push 1
        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
        F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge |
        F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => (2, 1),

        // Unary / conversions: pop 1, push 1
        I32Eqz | I32Clz | I32Ctz | I32Popcnt |
        I64Eqz | I64Clz | I64Ctz | I64Popcnt |
        F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt |
        F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt |
        I32WrapI64 | I32TruncF32S | I32TruncF32U | I32TruncF64S | I32TruncF64U |
        I64ExtendI32S | I64ExtendI32U |
        I64TruncF32S | I64TruncF32U | I64TruncF64S | I64TruncF64U |
        F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U | F32DemoteF64 |
        F64ConvertI32S | F64ConvertI32U | F64ConvertI64S | F64ConvertI64U | F64PromoteF32 |
        I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 | F64ReinterpretI64 |
        I32Extend8S | I32Extend16S | I64Extend8S | I64Extend16S | I64Extend32S |
        I32TruncSatF32S | I32TruncSatF32U | I32TruncSatF64S | I32TruncSatF64U |
        I64TruncSatF32S | I64TruncSatF32U | I64TruncSatF64S | I64TruncSatF64U |
        RefIsNull => (1, 1),

        // Constants: push 1
        I32Const { .. } | I64Const { .. } | F32Const { .. } | F64Const { .. } => (0, 1),

        // Local get / hot: push 1
        LocalGetHot { .. } | LocalGetFrame { .. } => (0, 1),
        // Local set / hot: pop 1
        LocalSetHot { .. } | LocalSetFrame { .. } => (1, 0),
        // Local tee: read top, write local (no height change)
        LocalTeeHot { .. } | LocalTeeFrame { .. } => (0, 0),

        // Spill/fill: no net stack effect (TOS management)
        Spill { .. } | Fill { .. } => (0, 0),

        // Memory loads: pop 1, push 1
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } => (1, 1),

        // Memory stores: pop 2, push 0
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => (2, 0),

        // Memory size: push 1
        MemorySize { .. } => (0, 1),
        // Memory grow: pop 1, push 1
        MemoryGrow { .. } => (1, 1),

        // Bulk memory: pop 3, push 0
        MemoryFill { .. } | MemoryCopy { .. } | MemoryInit { .. } => (3, 0),
        // Data drop: no stack effect
        DataDrop { .. } => (0, 0),

        // Globals
        GlobalGet { .. } => (0, 1),
        GlobalSet { .. } => (1, 0),

        // Tables
        TableGet { .. } => (1, 1),   // pop idx, push val
        TableSet { .. } => (2, 0),   // pop val + idx
        TableSize { .. } => (0, 1),  // push size
        TableGrow { .. } => (2, 1),  // pop val + n, push result
        TableFill { .. } | TableCopy { .. } | TableInit { .. } => (3, 0),
        ElemDrop { .. } => (0, 0),

        // References
        RefNull => (0, 1),
        RefFunc { .. } => (0, 1),

        // Stack ops
        Drop => (1, 0),
        Select => (3, 1),

        // Control flow — stack effects are complex and context-dependent.
        // These return (0,0) because the actual stack manipulation is handled
        // by the lowering pass, not by the backend.
        Block | Loop | If | Else | End => (0, 0),
        Br { .. } | BrIf { .. } | BrIfSimple | BrTable { .. } => (0, 0),

        // Calls — stack effects are computed from function signatures at call site.
        CallExternal { .. } | CallInternal { .. } | CallIndirect { .. } => (0, 0),

        // Returns — terminal
        ReturnVoid { .. } | ReturnOne { .. } | Return { .. } | Unreachable => (0, 0),

        // Prologue / structural / terminal
        InitLocals { .. } | Nop | Term | Data { .. } => (0, 0),
    }
}

/// Whether an IR op can be JIT-compiled into ARM64 native code.
///
/// Replaces `is_jit_able()` in group.rs. Height checks are not done here —
/// they are a property of the op's position in the IR stream, not the op itself.
/// The JIT grouper checks height when building groups.
pub fn is_jit_able(kind: &IrOpKind) -> bool {
    use IrOpKind::*;
    matches!(
        kind,
        // i32 binary
        I32Add | I32Sub | I32Mul |
        I32And | I32Or | I32Xor |
        I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |

        // i64 binary
        I64Add | I64Sub | I64Mul |
        I64And | I64Or | I64Xor |
        I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr |

        // i32 comparisons
        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU |
        I32LeS | I32LeU | I32GeS | I32GeU |

        // i64 comparisons
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU |
        I64LeS | I64LeU | I64GeS | I64GeU |

        // i32 unary
        I32Eqz | I32Clz | I32Ctz |

        // i64 unary
        I64Eqz | I64Clz | I64Ctz |

        // Constants
        I32Const { .. } | I64Const { .. } |

        // Locals (hot and frame)
        LocalGetHot { .. } | LocalGetFrame { .. } |
        LocalSetHot { .. } | LocalSetFrame { .. } |
        LocalTeeHot { .. } |

        // Drop
        Drop |

        // Memory loads
        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } |

        // Memory stores
        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. }
    )
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
    fn test_is_jit_able_arithmetic() {
        assert!(is_jit_able(&IrOpKind::I32Add));
        assert!(is_jit_able(&IrOpKind::I64Mul));
        assert!(is_jit_able(&IrOpKind::I32Eqz));
        assert!(is_jit_able(&IrOpKind::I32Const { value: 42 }));
        assert!(is_jit_able(&IrOpKind::LocalGetHot { reg: 0 }));
        assert!(is_jit_able(&IrOpKind::Drop));
    }

    #[test]
    fn test_is_jit_able_memory() {
        assert!(is_jit_able(&IrOpKind::I32Load { offset: 0, memidx: 0 }));
        assert!(is_jit_able(&IrOpKind::I32Store { offset: 0, memidx: 0 }));
        assert!(is_jit_able(&IrOpKind::I64Load32U { offset: 4, memidx: 0 }));
    }

    #[test]
    fn test_is_not_jit_able() {
        // Div/rem not JIT-able (can trap)
        assert!(!is_jit_able(&IrOpKind::I32DivS));
        assert!(!is_jit_able(&IrOpKind::I64RemU));
        // Float ops not JIT-able
        assert!(!is_jit_able(&IrOpKind::F32Add));
        assert!(!is_jit_able(&IrOpKind::F64Sqrt));
        // Control flow not JIT-able
        assert!(!is_jit_able(&IrOpKind::Block));
        assert!(!is_jit_able(&IrOpKind::Br { stack_drop: 0, arity: 0, height: 0, operand_base_offset: 0 }));
        // Calls not JIT-able
        assert!(!is_jit_able(&IrOpKind::CallExternal { func_idx: 0, delta: 0 }));
        // Globals not JIT-able
        assert!(!is_jit_able(&IrOpKind::GlobalGet { idx: 0 }));
        // Select not JIT-able
        assert!(!is_jit_able(&IrOpKind::Select));
    }

    #[test]
    fn test_bulk_memory_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::MemoryFill { mem_idx: 0 }), (3, 0));
        assert_eq!(stack_effect(&IrOpKind::MemoryCopy { dst_idx: 0, src_idx: 0 }), (3, 0));
        assert_eq!(stack_effect(&IrOpKind::DataDrop { data_idx: 0 }), (0, 0));
    }

    #[test]
    fn test_reference_stack_effect() {
        assert_eq!(stack_effect(&IrOpKind::RefNull), (0, 1));
        assert_eq!(stack_effect(&IrOpKind::RefIsNull), (1, 1));
        assert_eq!(stack_effect(&IrOpKind::RefFunc { func_idx: 0 }), (0, 1));
    }
}
