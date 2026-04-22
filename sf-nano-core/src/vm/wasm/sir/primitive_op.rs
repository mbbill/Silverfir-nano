//! Canonical leaf Wasm op vocabulary shared across semantic/planning/SSA-IR.
//!
//! This stays intentionally backend-agnostic and excludes function-body-only
//! structure such as locals, calls, returns, structured control, and branch
//! targets. The semantic layer wraps these leaf ops in a larger function-body
//! IR instead of duplicating the common non-structural Wasm operation set.

use crate::value_type::ValueType;

macro_rules! for_each_primitive_op {
    ($m:ident) => {
        $m! {
            I32Add => (2, 1),
            I32Sub => (2, 1),
            I32Mul => (2, 1),
            I32DivS => (2, 1),
            I32DivU => (2, 1),
            I32RemS => (2, 1),
            I32RemU => (2, 1),
            I32And => (2, 1),
            I32Or => (2, 1),
            I32Xor => (2, 1),
            I32Shl => (2, 1),
            I32ShrS => (2, 1),
            I32ShrU => (2, 1),
            I32Rotl => (2, 1),
            I32Rotr => (2, 1),
            I64Add => (2, 1),
            I64Sub => (2, 1),
            I64Mul => (2, 1),
            I64DivS => (2, 1),
            I64DivU => (2, 1),
            I64RemS => (2, 1),
            I64RemU => (2, 1),
            I64And => (2, 1),
            I64Or => (2, 1),
            I64Xor => (2, 1),
            I64Shl => (2, 1),
            I64ShrS => (2, 1),
            I64ShrU => (2, 1),
            I64Rotl => (2, 1),
            I64Rotr => (2, 1),
            F32Add => (2, 1),
            F32Sub => (2, 1),
            F32Mul => (2, 1),
            F32Div => (2, 1),
            F32Min => (2, 1),
            F32Max => (2, 1),
            F32Copysign => (2, 1),
            F64Add => (2, 1),
            F64Sub => (2, 1),
            F64Mul => (2, 1),
            F64Div => (2, 1),
            F64Min => (2, 1),
            F64Max => (2, 1),
            F64Copysign => (2, 1),
            I32Eq => (2, 1),
            I32Ne => (2, 1),
            I32LtS => (2, 1),
            I32LtU => (2, 1),
            I32GtS => (2, 1),
            I32GtU => (2, 1),
            I32LeS => (2, 1),
            I32LeU => (2, 1),
            I32GeS => (2, 1),
            I32GeU => (2, 1),
            I64Eq => (2, 1),
            I64Ne => (2, 1),
            I64LtS => (2, 1),
            I64LtU => (2, 1),
            I64GtS => (2, 1),
            I64GtU => (2, 1),
            I64LeS => (2, 1),
            I64LeU => (2, 1),
            I64GeS => (2, 1),
            I64GeU => (2, 1),
            F32Eq => (2, 1),
            F32Ne => (2, 1),
            F32Lt => (2, 1),
            F32Gt => (2, 1),
            F32Le => (2, 1),
            F32Ge => (2, 1),
            F64Eq => (2, 1),
            F64Ne => (2, 1),
            F64Lt => (2, 1),
            F64Gt => (2, 1),
            F64Le => (2, 1),
            F64Ge => (2, 1),
            I32Eqz => (1, 1),
            I32Clz => (1, 1),
            I32Ctz => (1, 1),
            I32Popcnt => (1, 1),
            I64Eqz => (1, 1),
            I64Clz => (1, 1),
            I64Ctz => (1, 1),
            I64Popcnt => (1, 1),
            F32Abs => (1, 1),
            F32Neg => (1, 1),
            F32Ceil => (1, 1),
            F32Floor => (1, 1),
            F32Trunc => (1, 1),
            F32Nearest => (1, 1),
            F32Sqrt => (1, 1),
            F64Abs => (1, 1),
            F64Neg => (1, 1),
            F64Ceil => (1, 1),
            F64Floor => (1, 1),
            F64Trunc => (1, 1),
            F64Nearest => (1, 1),
            F64Sqrt => (1, 1),
            I32WrapI64 => (1, 1),
            I32TruncF32S => (1, 1),
            I32TruncF32U => (1, 1),
            I32TruncF64S => (1, 1),
            I32TruncF64U => (1, 1),
            I64ExtendI32S => (1, 1),
            I64ExtendI32U => (1, 1),
            I64TruncF32S => (1, 1),
            I64TruncF32U => (1, 1),
            I64TruncF64S => (1, 1),
            I64TruncF64U => (1, 1),
            F32ConvertI32S => (1, 1),
            F32ConvertI32U => (1, 1),
            F32ConvertI64S => (1, 1),
            F32ConvertI64U => (1, 1),
            F32DemoteF64 => (1, 1),
            F64ConvertI32S => (1, 1),
            F64ConvertI32U => (1, 1),
            F64ConvertI64S => (1, 1),
            F64ConvertI64U => (1, 1),
            F64PromoteF32 => (1, 1),
            I32ReinterpretF32 => (1, 1),
            I64ReinterpretF64 => (1, 1),
            F32ReinterpretI32 => (1, 1),
            F64ReinterpretI64 => (1, 1),
            I32Extend8S => (1, 1),
            I32Extend16S => (1, 1),
            I64Extend8S => (1, 1),
            I64Extend16S => (1, 1),
            I64Extend32S => (1, 1),
            I32TruncSatF32S => (1, 1),
            I32TruncSatF32U => (1, 1),
            I32TruncSatF64S => (1, 1),
            I32TruncSatF64U => (1, 1),
            I64TruncSatF32S => (1, 1),
            I64TruncSatF32U => (1, 1),
            I64TruncSatF64S => (1, 1),
            I64TruncSatF64U => (1, 1),
            I32Const { value: u32 } => (0, 1),
            I64Const { value: u64 } => (0, 1),
            F32Const { value: u32 } => (0, 1),
            F64Const { value: u64 } => (0, 1),
            V128Const { value: [u8; 16] } => (0, 1),
            V128Not => (1, 1),
            V128And => (2, 1),
            V128AndNot => (2, 1),
            V128Or => (2, 1),
            V128Xor => (2, 1),
            V128Bitselect => (3, 1),
            V128AnyTrue => (1, 1),
            I8x16AllTrue => (1, 1),
            I16x8AllTrue => (1, 1),
            I32x4AllTrue => (1, 1),
            I64x2AllTrue => (1, 1),
            I8x16Bitmask => (1, 1),
            I16x8Bitmask => (1, 1),
            I32x4Bitmask => (1, 1),
            I64x2Bitmask => (1, 1),
            I8x16Splat => (1, 1),
            I16x8Splat => (1, 1),
            I32x4Splat => (1, 1),
            I64x2Splat => (1, 1),
            F32x4Splat => (1, 1),
            F64x2Splat => (1, 1),
            I8x16ExtractLaneS { lane: u8 } => (1, 1),
            I8x16ExtractLaneU { lane: u8 } => (1, 1),
            I16x8ExtractLaneS { lane: u8 } => (1, 1),
            I16x8ExtractLaneU { lane: u8 } => (1, 1),
            I32x4ExtractLane { lane: u8 } => (1, 1),
            I64x2ExtractLane { lane: u8 } => (1, 1),
            F32x4ExtractLane { lane: u8 } => (1, 1),
            F64x2ExtractLane { lane: u8 } => (1, 1),
            SimdReplaceLane { opcode: u32, lane: u8 } => (2, 1),
            I8x16Shuffle { lanes: [u8; 16] } => (2, 1),
            I32x4Add => (2, 1),
            I64x2Add => (2, 1),
            SimdUnaryV128 { opcode: u32 } => (1, 1),
            SimdBinaryV128 { opcode: u32 } => (2, 1),
            SimdTernaryV128 { opcode: u32 } => (3, 1),
            SimdShiftV128 { opcode: u32 } => (2, 1),
            SimdMemLoadV128 { opcode: u32, offset: u32, memidx: u32 } => (1, 1),
            SimdMemLoadLaneV128 { opcode: u32, lane: u8, offset: u32, memidx: u32 } => (2, 1),
            SimdMemStoreLaneV128 { opcode: u32, lane: u8, offset: u32, memidx: u32 } => (2, 0),
            V128Load { offset: u32, memidx: u32 } => (1, 1),
            V128Store { offset: u32, memidx: u32 } => (2, 0),
            I32Load { offset: u32, memidx: u32 } => (1, 1),
            I64Load { offset: u32, memidx: u32 } => (1, 1),
            F32Load { offset: u32, memidx: u32 } => (1, 1),
            F64Load { offset: u32, memidx: u32 } => (1, 1),
            I32Load8S { offset: u32, memidx: u32 } => (1, 1),
            I32Load8U { offset: u32, memidx: u32 } => (1, 1),
            I32Load16S { offset: u32, memidx: u32 } => (1, 1),
            I32Load16U { offset: u32, memidx: u32 } => (1, 1),
            I64Load8S { offset: u32, memidx: u32 } => (1, 1),
            I64Load8U { offset: u32, memidx: u32 } => (1, 1),
            I64Load16S { offset: u32, memidx: u32 } => (1, 1),
            I64Load16U { offset: u32, memidx: u32 } => (1, 1),
            I64Load32S { offset: u32, memidx: u32 } => (1, 1),
            I64Load32U { offset: u32, memidx: u32 } => (1, 1),
            I32Store { offset: u32, memidx: u32 } => (2, 0),
            I64Store { offset: u32, memidx: u32 } => (2, 0),
            F32Store { offset: u32, memidx: u32 } => (2, 0),
            F64Store { offset: u32, memidx: u32 } => (2, 0),
            I32Store8 { offset: u32, memidx: u32 } => (2, 0),
            I32Store16 { offset: u32, memidx: u32 } => (2, 0),
            I64Store8 { offset: u32, memidx: u32 } => (2, 0),
            I64Store16 { offset: u32, memidx: u32 } => (2, 0),
            I64Store32 { offset: u32, memidx: u32 } => (2, 0),
            GlobalGet { idx: u32 } => (0, 1),
            GlobalSet { idx: u32 } => (1, 0),
            MemorySize { mem_idx: u32 } => (0, 1),
            MemoryGrow { mem_idx: u32 } => (1, 1),
            MemoryFill { imm0: u32, imm1: u32 } => (3, 0),
            MemoryCopy { imm0: u32, imm1: u32 } => (3, 0),
            MemoryInit { imm0: u32, imm1: u32 } => (3, 0),
            DataDrop { data_idx: u32 } => (0, 0),
            TableGet { table_idx: u32 } => (1, 1),
            TableSet { table_idx: u32 } => (2, 0),
            TableSize { table_idx: u32 } => (0, 1),
            TableGrow { table_idx: u32 } => (2, 1),
            TableFill { imm0: u32, imm1: u32 } => (3, 0),
            TableCopy { imm0: u32, imm1: u32 } => (3, 0),
            TableInit { imm0: u32, imm1: u32 } => (3, 0),
            ElemDrop { elem_idx: u32 } => (0, 0),
            RefNull => (0, 1),
            RefIsNull => (1, 1),
            RefFunc { func_idx: u32 } => (0, 1),
            EhAllocExnRef { tag_idx: u32 } => (0, 1),
            RefAsNonNull => (1, 1),
            RefEq => (2, 1),
            RefI31 => (1, 1),
            I31GetS => (1, 1),
            I31GetU => (1, 1),
            AnyConvertExtern => (1, 1),
            ExternConvertAny => (1, 1),
            RefTest { ref_type: crate::value_type::RefType } => (1, 1),
            RefCast { ref_type: crate::value_type::RefType } => (1, 1),
            StructNew { type_idx: u32, field_count: u32 } => (*field_count as usize, 1),
            StructNewDefault { type_idx: u32 } => (0, 1),
            StructGet { type_idx: u32, field_idx: u32 } => (1, 1),
            StructGetS { type_idx: u32, field_idx: u32 } => (1, 1),
            StructGetU { type_idx: u32, field_idx: u32 } => (1, 1),
            StructSet { type_idx: u32, field_idx: u32 } => (2, 0),
            ArrayNew { type_idx: u32 } => (2, 1),
            ArrayNewDefault { type_idx: u32 } => (1, 1),
            ArrayNewFixed { type_idx: u32, count: u32 } => (*count as usize, 1),
            ArrayNewData { type_idx: u32, data_idx: u32 } => (2, 1),
            ArrayNewElem { type_idx: u32, elem_idx: u32 } => (2, 1),
            ArrayGet { type_idx: u32 } => (2, 1),
            ArrayGetS { type_idx: u32 } => (2, 1),
            ArrayGetU { type_idx: u32 } => (2, 1),
            ArraySet { type_idx: u32 } => (3, 0),
            ArrayFill { type_idx: u32 } => (4, 0),
            ArrayCopy { dst_type_idx: u32, src_type_idx: u32 } => (5, 0),
            ArrayInitData { type_idx: u32, data_idx: u32 } => (4, 0),
            ArrayInitElem { type_idx: u32, elem_idx: u32 } => (4, 0),
            ArrayLen => (1, 1),
            Drop => (1, 0),
            Select => (3, 1),
            Nop => (0, 0),
            Unreachable => (0, 0),
        }
    };
}

pub(crate) use for_each_primitive_op;
macro_rules! define_primitive_ops {
    ($(
        $name:ident $( { $($field:ident : $ty:ty),* $(,)? } )? => ($pops:expr, $pushes:expr),
    )* ) => {
        /// Pure reusable Wasm leaf operations shared across compiler layers.
        ///
        /// Keep this limited to non-structural ops with stable stack effects.
        /// Function-body semantics such as locals, calls, returns, structured
        /// control, and branch metadata live in `SemanticOpKind`, which embeds
        /// `PrimitiveOpKind` instead of expanding this enum into a whole-function IR.
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub(crate) enum PrimitiveOpKind {
            $( $name $( { $($field : $ty),* } )?, )*
        }

        #[allow(unused_variables)]
        #[inline]
        pub(crate) fn stack_effect(kind: &PrimitiveOpKind) -> (usize, usize) {
            match kind {
                $( PrimitiveOpKind::$name $( { $($field),* } )? => ($pops, $pushes), )*
            }
        }
    };
}

for_each_primitive_op!(define_primitive_ops);

/// Returns the Wasm value type of the single result produced by a primitive op.
///
/// Returns `None` for ops that produce zero results (stores, drops, etc.) or
/// ops whose result type depends on context (Select, GlobalGet).
#[inline]
pub(crate) fn result_type(kind: &PrimitiveOpKind) -> Option<ValueType> {
    Some(match kind {
        // i32 arithmetic
        PrimitiveOpKind::I32Add
        | PrimitiveOpKind::I32Sub
        | PrimitiveOpKind::I32Mul
        | PrimitiveOpKind::I32DivS
        | PrimitiveOpKind::I32DivU
        | PrimitiveOpKind::I32RemS
        | PrimitiveOpKind::I32RemU
        | PrimitiveOpKind::I32And
        | PrimitiveOpKind::I32Or
        | PrimitiveOpKind::I32Xor
        | PrimitiveOpKind::I32Shl
        | PrimitiveOpKind::I32ShrS
        | PrimitiveOpKind::I32ShrU
        | PrimitiveOpKind::I32Rotl
        | PrimitiveOpKind::I32Rotr => ValueType::I32,

        // i64 arithmetic
        PrimitiveOpKind::I64Add
        | PrimitiveOpKind::I64Sub
        | PrimitiveOpKind::I64Mul
        | PrimitiveOpKind::I64DivS
        | PrimitiveOpKind::I64DivU
        | PrimitiveOpKind::I64RemS
        | PrimitiveOpKind::I64RemU
        | PrimitiveOpKind::I64And
        | PrimitiveOpKind::I64Or
        | PrimitiveOpKind::I64Xor
        | PrimitiveOpKind::I64Shl
        | PrimitiveOpKind::I64ShrS
        | PrimitiveOpKind::I64ShrU
        | PrimitiveOpKind::I64Rotl
        | PrimitiveOpKind::I64Rotr => ValueType::I64,

        // f32 arithmetic
        PrimitiveOpKind::F32Add
        | PrimitiveOpKind::F32Sub
        | PrimitiveOpKind::F32Mul
        | PrimitiveOpKind::F32Div
        | PrimitiveOpKind::F32Min
        | PrimitiveOpKind::F32Max
        | PrimitiveOpKind::F32Copysign => ValueType::F32,

        // f64 arithmetic
        PrimitiveOpKind::F64Add
        | PrimitiveOpKind::F64Sub
        | PrimitiveOpKind::F64Mul
        | PrimitiveOpKind::F64Div
        | PrimitiveOpKind::F64Min
        | PrimitiveOpKind::F64Max
        | PrimitiveOpKind::F64Copysign => ValueType::F64,

        // All comparisons produce i32
        PrimitiveOpKind::I32Eq
        | PrimitiveOpKind::I32Ne
        | PrimitiveOpKind::I32LtS
        | PrimitiveOpKind::I32LtU
        | PrimitiveOpKind::I32GtS
        | PrimitiveOpKind::I32GtU
        | PrimitiveOpKind::I32LeS
        | PrimitiveOpKind::I32LeU
        | PrimitiveOpKind::I32GeS
        | PrimitiveOpKind::I32GeU
        | PrimitiveOpKind::I64Eq
        | PrimitiveOpKind::I64Ne
        | PrimitiveOpKind::I64LtS
        | PrimitiveOpKind::I64LtU
        | PrimitiveOpKind::I64GtS
        | PrimitiveOpKind::I64GtU
        | PrimitiveOpKind::I64LeS
        | PrimitiveOpKind::I64LeU
        | PrimitiveOpKind::I64GeS
        | PrimitiveOpKind::I64GeU
        | PrimitiveOpKind::F32Eq
        | PrimitiveOpKind::F32Ne
        | PrimitiveOpKind::F32Lt
        | PrimitiveOpKind::F32Gt
        | PrimitiveOpKind::F32Le
        | PrimitiveOpKind::F32Ge
        | PrimitiveOpKind::F64Eq
        | PrimitiveOpKind::F64Ne
        | PrimitiveOpKind::F64Lt
        | PrimitiveOpKind::F64Gt
        | PrimitiveOpKind::F64Le
        | PrimitiveOpKind::F64Ge
        | PrimitiveOpKind::I32Eqz
        | PrimitiveOpKind::I64Eqz => ValueType::I32,

        // i32 unary
        PrimitiveOpKind::I32Clz | PrimitiveOpKind::I32Ctz | PrimitiveOpKind::I32Popcnt => {
            ValueType::I32
        }

        // i64 unary
        PrimitiveOpKind::I64Clz | PrimitiveOpKind::I64Ctz | PrimitiveOpKind::I64Popcnt => {
            ValueType::I64
        }

        // f32 unary
        PrimitiveOpKind::F32Abs
        | PrimitiveOpKind::F32Neg
        | PrimitiveOpKind::F32Ceil
        | PrimitiveOpKind::F32Floor
        | PrimitiveOpKind::F32Trunc
        | PrimitiveOpKind::F32Nearest
        | PrimitiveOpKind::F32Sqrt => ValueType::F32,

        // f64 unary
        PrimitiveOpKind::F64Abs
        | PrimitiveOpKind::F64Neg
        | PrimitiveOpKind::F64Ceil
        | PrimitiveOpKind::F64Floor
        | PrimitiveOpKind::F64Trunc
        | PrimitiveOpKind::F64Nearest
        | PrimitiveOpKind::F64Sqrt => ValueType::F64,

        // Conversions — result type is the target type
        PrimitiveOpKind::I32WrapI64
        | PrimitiveOpKind::I32TruncF32S
        | PrimitiveOpKind::I32TruncF32U
        | PrimitiveOpKind::I32TruncF64S
        | PrimitiveOpKind::I32TruncF64U
        | PrimitiveOpKind::I32TruncSatF32S
        | PrimitiveOpKind::I32TruncSatF32U
        | PrimitiveOpKind::I32TruncSatF64S
        | PrimitiveOpKind::I32TruncSatF64U
        | PrimitiveOpKind::I32ReinterpretF32 => ValueType::I32,

        PrimitiveOpKind::I64ExtendI32S
        | PrimitiveOpKind::I64ExtendI32U
        | PrimitiveOpKind::I64TruncF32S
        | PrimitiveOpKind::I64TruncF32U
        | PrimitiveOpKind::I64TruncF64S
        | PrimitiveOpKind::I64TruncF64U
        | PrimitiveOpKind::I64TruncSatF32S
        | PrimitiveOpKind::I64TruncSatF32U
        | PrimitiveOpKind::I64TruncSatF64S
        | PrimitiveOpKind::I64TruncSatF64U
        | PrimitiveOpKind::I64ReinterpretF64 => ValueType::I64,

        PrimitiveOpKind::F32ConvertI32S
        | PrimitiveOpKind::F32ConvertI32U
        | PrimitiveOpKind::F32ConvertI64S
        | PrimitiveOpKind::F32ConvertI64U
        | PrimitiveOpKind::F32DemoteF64
        | PrimitiveOpKind::F32ReinterpretI32 => ValueType::F32,

        PrimitiveOpKind::F64ConvertI32S
        | PrimitiveOpKind::F64ConvertI32U
        | PrimitiveOpKind::F64ConvertI64S
        | PrimitiveOpKind::F64ConvertI64U
        | PrimitiveOpKind::F64PromoteF32
        | PrimitiveOpKind::F64ReinterpretI64 => ValueType::F64,

        // Extensions
        PrimitiveOpKind::I32Extend8S | PrimitiveOpKind::I32Extend16S => ValueType::I32,
        PrimitiveOpKind::I64Extend8S
        | PrimitiveOpKind::I64Extend16S
        | PrimitiveOpKind::I64Extend32S => ValueType::I64,

        // Constants
        PrimitiveOpKind::I32Const { .. } => ValueType::I32,
        PrimitiveOpKind::I64Const { .. } => ValueType::I64,
        PrimitiveOpKind::F32Const { .. } => ValueType::F32,
        PrimitiveOpKind::F64Const { .. } => ValueType::F64,
        PrimitiveOpKind::V128Const { .. }
        | PrimitiveOpKind::V128Not
        | PrimitiveOpKind::V128And
        | PrimitiveOpKind::V128AndNot
        | PrimitiveOpKind::V128Or
        | PrimitiveOpKind::V128Xor
        | PrimitiveOpKind::V128Bitselect
        | PrimitiveOpKind::I8x16Splat
        | PrimitiveOpKind::I16x8Splat
        | PrimitiveOpKind::I32x4Splat
        | PrimitiveOpKind::I64x2Splat
        | PrimitiveOpKind::F32x4Splat
        | PrimitiveOpKind::F64x2Splat
        | PrimitiveOpKind::SimdReplaceLane { .. }
        | PrimitiveOpKind::I8x16Shuffle { .. }
        | PrimitiveOpKind::I32x4Add
        | PrimitiveOpKind::I64x2Add
        | PrimitiveOpKind::SimdUnaryV128 { .. }
        | PrimitiveOpKind::SimdBinaryV128 { .. }
        | PrimitiveOpKind::SimdTernaryV128 { .. }
        | PrimitiveOpKind::SimdShiftV128 { .. }
        | PrimitiveOpKind::SimdMemLoadV128 { .. }
        | PrimitiveOpKind::SimdMemLoadLaneV128 { .. }
        | PrimitiveOpKind::V128Load { .. } => ValueType::V128,
        PrimitiveOpKind::I8x16ExtractLaneS { .. }
        | PrimitiveOpKind::I8x16ExtractLaneU { .. }
        | PrimitiveOpKind::I16x8ExtractLaneS { .. }
        | PrimitiveOpKind::I16x8ExtractLaneU { .. }
        | PrimitiveOpKind::I32x4ExtractLane { .. } => ValueType::I32,
        PrimitiveOpKind::I64x2ExtractLane { .. } => ValueType::I64,
        PrimitiveOpKind::F32x4ExtractLane { .. } => ValueType::F32,
        PrimitiveOpKind::F64x2ExtractLane { .. } => ValueType::F64,
        PrimitiveOpKind::V128AnyTrue
        | PrimitiveOpKind::I8x16AllTrue
        | PrimitiveOpKind::I16x8AllTrue
        | PrimitiveOpKind::I32x4AllTrue
        | PrimitiveOpKind::I64x2AllTrue
        | PrimitiveOpKind::I8x16Bitmask
        | PrimitiveOpKind::I16x8Bitmask
        | PrimitiveOpKind::I32x4Bitmask
        | PrimitiveOpKind::I64x2Bitmask => ValueType::I32,

        // Loads — result type matches the load type
        PrimitiveOpKind::I32Load { .. }
        | PrimitiveOpKind::I32Load8S { .. }
        | PrimitiveOpKind::I32Load8U { .. }
        | PrimitiveOpKind::I32Load16S { .. }
        | PrimitiveOpKind::I32Load16U { .. } => ValueType::I32,

        PrimitiveOpKind::I64Load { .. }
        | PrimitiveOpKind::I64Load8S { .. }
        | PrimitiveOpKind::I64Load8U { .. }
        | PrimitiveOpKind::I64Load16S { .. }
        | PrimitiveOpKind::I64Load16U { .. }
        | PrimitiveOpKind::I64Load32S { .. }
        | PrimitiveOpKind::I64Load32U { .. } => ValueType::I64,

        PrimitiveOpKind::F32Load { .. } => ValueType::F32,
        PrimitiveOpKind::F64Load { .. } => ValueType::F64,

        // Stores — no result
        PrimitiveOpKind::V128Store { .. }
        | PrimitiveOpKind::SimdMemStoreLaneV128 { .. }
        | PrimitiveOpKind::I32Store { .. }
        | PrimitiveOpKind::I64Store { .. }
        | PrimitiveOpKind::F32Store { .. }
        | PrimitiveOpKind::F64Store { .. }
        | PrimitiveOpKind::I32Store8 { .. }
        | PrimitiveOpKind::I32Store16 { .. }
        | PrimitiveOpKind::I64Store8 { .. }
        | PrimitiveOpKind::I64Store16 { .. }
        | PrimitiveOpKind::I64Store32 { .. } => return None,

        // Globals — type depends on the global, not deterministic here
        PrimitiveOpKind::GlobalGet { .. } => return None,
        PrimitiveOpKind::GlobalSet { .. } => return None,

        // Memory/table size/grow depend on the selected memory/table index type.
        PrimitiveOpKind::MemorySize { .. } | PrimitiveOpKind::MemoryGrow { .. } => return None,

        // Memory/table bulk ops — no result or boundary
        PrimitiveOpKind::MemoryFill { .. }
        | PrimitiveOpKind::MemoryCopy { .. }
        | PrimitiveOpKind::MemoryInit { .. }
        | PrimitiveOpKind::DataDrop { .. }
        | PrimitiveOpKind::TableFill { .. }
        | PrimitiveOpKind::TableCopy { .. }
        | PrimitiveOpKind::TableInit { .. }
        | PrimitiveOpKind::ElemDrop { .. }
        | PrimitiveOpKind::TableSet { .. } => return None,

        // Table get/size/grow — context-dependent
        PrimitiveOpKind::TableGet { .. }
        | PrimitiveOpKind::TableSize { .. }
        | PrimitiveOpKind::TableGrow { .. } => return None,

        // References
        PrimitiveOpKind::RefNull => return None,
        PrimitiveOpKind::RefIsNull => ValueType::I32,
        PrimitiveOpKind::RefFunc { .. } => return None,
        PrimitiveOpKind::EhAllocExnRef { .. } => ValueType::ref_exn(),
        PrimitiveOpKind::RefAsNonNull => ValueType::anyref(),
        PrimitiveOpKind::RefEq => ValueType::I32,
        PrimitiveOpKind::RefI31 => ValueType::i31ref(),
        PrimitiveOpKind::I31GetS | PrimitiveOpKind::I31GetU => ValueType::I32,
        PrimitiveOpKind::AnyConvertExtern => ValueType::anyref(),
        PrimitiveOpKind::ExternConvertAny => ValueType::externref(),
        PrimitiveOpKind::RefTest { .. } => ValueType::I32,
        PrimitiveOpKind::RefCast { .. } => return None,
        PrimitiveOpKind::StructNew { .. } => return None,
        PrimitiveOpKind::StructNewDefault { .. } => return None,
        PrimitiveOpKind::StructGet { .. }
        | PrimitiveOpKind::StructGetS { .. }
        | PrimitiveOpKind::StructGetU { .. } => return None,
        PrimitiveOpKind::StructSet { .. } => return None,
        PrimitiveOpKind::ArrayNew { .. } => return None,
        PrimitiveOpKind::ArrayNewDefault { .. } => return None,
        PrimitiveOpKind::ArrayNewFixed { .. } => return None,
        PrimitiveOpKind::ArrayNewData { .. } => return None,
        PrimitiveOpKind::ArrayNewElem { .. } => return None,
        PrimitiveOpKind::ArrayGet { .. }
        | PrimitiveOpKind::ArrayGetS { .. }
        | PrimitiveOpKind::ArrayGetU { .. } => return None,
        PrimitiveOpKind::ArraySet { .. } => return None,
        PrimitiveOpKind::ArrayFill { .. } => return None,
        PrimitiveOpKind::ArrayCopy { .. } => return None,
        PrimitiveOpKind::ArrayInitData { .. } => return None,
        PrimitiveOpKind::ArrayInitElem { .. } => return None,
        PrimitiveOpKind::ArrayLen => ValueType::I32,

        // Stack/control
        PrimitiveOpKind::Drop => return None,
        PrimitiveOpKind::Select => return None,
        PrimitiveOpKind::Nop => return None,
        PrimitiveOpKind::Unreachable => return None,
    })
}
