//! x86_64 backend: instruction lowering methods for X86_64Backend.

use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::{
            MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp,
            MachineFloatWidth, MachineIndexExtend, MachineInst, MachineInstKind,
            MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension,
            MachineMemWidth, MachineReg, MachineShiftOp, MachineSign, MachineStorageType,
            MachineTrapKind, MachineValue, MACHINE_FP_REG,
        },
        jit::runtime::preserved,
    },
};

use super::{
    abi::{C_ARG0, C_ARG1},
    backend::{FpLiteralFixup, X86_64Backend},
    callconv,
    enc::{self, Cc},
    fusion::map_int_cond,
    helpers::x86_64_saturating_trunc,
    reg::X86Reg,
};
use crate::vm::jit::arch::common::helpers::convert_result_float_width;

use crate::vm::jit::machine::machine_ir::MachineAddr;

/// Map a `MachineConvertOp` to the u32 op code consumed by the runtime
/// trunc/saturating-trunc helpers. Returns `u32::MAX` for ops that do not
/// go through the helper path.
pub(super) fn convert_op_code(op: MachineConvertOp) -> u32 {
    match op {
        MachineConvertOp::I32TruncF32S => 0,
        MachineConvertOp::I32TruncF32U => 1,
        MachineConvertOp::I32TruncF64S => 2,
        MachineConvertOp::I32TruncF64U => 3,
        MachineConvertOp::I64TruncF32S => 4,
        MachineConvertOp::I64TruncF32U => 5,
        MachineConvertOp::I64TruncF64S => 6,
        MachineConvertOp::I64TruncF64U => 7,
        MachineConvertOp::I32TruncSatF32S => 8,
        MachineConvertOp::I32TruncSatF32U => 9,
        MachineConvertOp::I32TruncSatF64S => 10,
        MachineConvertOp::I32TruncSatF64U => 11,
        MachineConvertOp::I64TruncSatF32S => 12,
        MachineConvertOp::I64TruncSatF32U => 13,
        MachineConvertOp::I64TruncSatF64S => 14,
        MachineConvertOp::I64TruncSatF64U => 15,
        _ => u32::MAX,
    }
}

impl<'a> X86_64Backend<'a> {
    pub(super) fn lower_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move {
                ty: MachineStorageType::V128,
                ..
            }
            | MachineInstKind::V128Const { .. }
            | MachineInstKind::V128FromRaw { .. }
            | MachineInstKind::V128ToRaw { .. }
            | MachineInstKind::SimdUnary { .. }
            | MachineInstKind::SimdBinary { .. }
            | MachineInstKind::SimdTernary { .. }
            | MachineInstKind::SimdShift { .. }
            | MachineInstKind::SimdExtractLane { .. }
            | MachineInstKind::SimdReplaceLane { .. }
            | MachineInstKind::SimdShuffle { .. }
            | MachineInstKind::SimdLoad { .. }
            | MachineInstKind::SimdStore { .. }
            | MachineInstKind::SimdLoadLane { .. }
            | MachineInstKind::SimdStoreLane { .. } => self.lower_simd_inst_dispatch(inst),
            MachineInstKind::Move { dst, src, ty, .. } => self.lower_move(*ty, *dst, *src),
            MachineInstKind::FloatConst { width, dst, bits } => {
                self.lower_float_const(*width, *dst, *bits)
            }
            MachineInstKind::Load {
                dst,
                addr,
                width,
                extension,
                ..
            } => self.lower_load(*dst, *addr, *width, *extension),
            MachineInstKind::Store {
                ty: _,
                addr,
                width,
                src,
            } => self.lower_store(*addr, *width, *src),
            MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            } => self.lower_int_unary(*width, *op, *dst, *src),
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.lower_int_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::Int64PairBinary { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairBinary; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairUnary { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairUnary; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairDivRem { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairDivRem; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairShift { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairShift; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64MulFromSignExt32 { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64MulFromSignExt32; this peephole only fires on 32-bit GP backends".into(),
            )),
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => self.lower_int_compare(*width, *kind, *sign, *dst, *lhs, *rhs),
            MachineInstKind::Select {
                ty: MachineStorageType::V128,
                dst,
                on_true,
                on_false,
                cond,
                ..
            } => self.lower_select_v128(*dst, *on_true, *on_false, *cond),
            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
                ..
            } => self.lower_select(*ty, *dst, *on_true, *on_false, *cond),
            MachineInstKind::TrapIf { kind, cond } => {
                let trap_label = self.core.ensure_trap_label(*kind);
                self.lower_branch_if(cond, trap_label)
            }
            MachineInstKind::CallRuntime(call) => {
                self.lower_call_runtime(call.metadata.0 as usize)
            }
            MachineInstKind::EhThrow { tag_idx, args } => self.lower_preserved_no_result(
                preserved::op::EH_THROW,
                *tag_idx,
                0,
                MachineValue::Reg(MACHINE_FP_REG),
                MachineValue::Imm64(args.start.0 as u64),
                MachineValue::Imm64(args.count as u64),
            ),
            MachineInstKind::EhThrowRef { exnref_slot } => self.lower_preserved_no_result(
                preserved::op::EH_THROW_REF,
                0,
                0,
                MachineValue::Reg(MACHINE_FP_REG),
                MachineValue::Imm64(exnref_slot.0 as u64),
                MachineValue::Imm64(0),
            ),
            MachineInstKind::EhAllocExnRef { tag_idx, dst } => self.lower_preserved_result(
                preserved::op::EH_ALLOC_EXN_REF,
                *tag_idx,
                0,
                MachineValue::Reg(MACHINE_FP_REG),
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::FloatUnary {
                width,
                op,
                dst,
                src,
            } => self.lower_float_unary(*width, *op, *dst, *src),
            MachineInstKind::FloatBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => self.lower_float_binary(*width, *op, *dst, *lhs, *rhs),
            MachineInstKind::FloatCompare {
                width,
                kind,
                dst,
                lhs,
                rhs,
            } => self.lower_float_compare(*width, *kind, *dst, *lhs, *rhs),
            MachineInstKind::Convert { op, dst, src } => self.lower_convert(*op, *dst, *src),
            MachineInstKind::ConvertI64PairToFloat { .. } => Err(WasmError::internal(
                "x86_64 backend received ConvertI64PairToFloat; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::Int64PairCompare { .. } => Err(WasmError::internal(
                "x86_64 backend received Int64PairCompare; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ConvertFloatToI64Pair { .. } => Err(WasmError::internal(
                "x86_64 backend received ConvertFloatToI64Pair; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ReinterpretF64ToI64Pair { .. } => Err(WasmError::internal(
                "x86_64 backend received ReinterpretF64ToI64Pair; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::ReinterpretI64PairToF64 { .. } => Err(WasmError::internal(
                "x86_64 backend received ReinterpretI64PairToF64; 32-bit legalized MachineIR should not reach x86_64 codegen".into(),
            )),
            MachineInstKind::IndexedLoad {
                dst,
                base,
                index,
                index_extend,
                offset,
                width,
                extension,
            } => {
                self.lower_indexed_load(*dst, *base, *index, *index_extend, *offset, *width, *extension)
            }
            MachineInstKind::IndexedStore {
                base,
                index,
                index_extend,
                offset,
                width,
                src,
            } => {
                self.lower_indexed_store(*base, *index, *index_extend, *offset, *width, *src)
            }
            MachineInstKind::BitfieldExtractU { width, dst, src, lsb, bits } => {
                self.lower_bitfield_extract_u(*width, *dst, *src, *lsb, *bits)
            }
            MachineInstKind::IntBinaryShifted { width, op, dst, lhs, rhs, shift, amount } => {
                self.lower_int_binary_shifted(*width, *op, *dst, *lhs, *rhs, *shift, *amount)
            }
            MachineInstKind::TestBits { width, kind, dst, src, mask } => {
                self.lower_test_bits(*width, *kind, *dst, *src, *mask)
            }
            MachineInstKind::MemoryGrow { mem_idx, dst, delta } => {
                self.lower_memory_grow(*mem_idx, *dst, *delta)
            }
            MachineInstKind::MemoryFill { mem_idx, dest, val, len } => {
                self.lower_memory_fill(*mem_idx, *dest, *val, *len)
            }
            MachineInstKind::MemoryCopy { dst_mem, src_mem, dest, src, len } => {
                self.lower_memory_copy(*dst_mem, *src_mem, *dest, *src, *len)
            }
            MachineInstKind::MemoryInit { mem_idx, data_idx, dest, src, len } => {
                self.lower_memory_init(*mem_idx, *data_idx, *dest, *src, *len)
            }
            MachineInstKind::DataDrop { data_idx } => self.lower_data_drop(*data_idx),
            MachineInstKind::TableGrow { table_idx, dst, init_val, delta } => {
                self.lower_table_grow(*table_idx, *dst, *init_val, *delta)
            }
            MachineInstKind::TableFill { table_idx, start, val, len } => {
                self.lower_table_fill(*table_idx, *start, *val, *len)
            }
            MachineInstKind::TableCopy { dst_tbl, src_tbl, dest, src, len } => {
                self.lower_table_copy(*dst_tbl, *src_tbl, *dest, *src, *len)
            }
            MachineInstKind::TableInit { table_idx, elem_idx, dest, src, len } => {
                self.lower_table_init(*table_idx, *elem_idx, *dest, *src, *len)
            }
            MachineInstKind::ElemDrop { elem_idx } => self.lower_elem_drop(*elem_idx),
            MachineInstKind::RefFunc { func_idx, dst } => self.lower_preserved_result(
                preserved::op::REF_FUNC,
                *func_idx,
                0,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefAsNonNull { src, dst } => self.lower_preserved_result(
                preserved::op::REF_AS_NON_NULL,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefAbsolutize { src, dst } => self.lower_preserved_result(
                preserved::op::REF_ABSOLUTIZE,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefEq { lhs, rhs, dst } => self.lower_preserved_result(
                preserved::op::REF_EQ,
                0,
                0,
                *lhs,
                *rhs,
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefI31 { src, dst } => self.lower_preserved_result(
                preserved::op::REF_I31,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::I31GetS { src, dst } => self.lower_preserved_result(
                preserved::op::I31_GET_S,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::I31GetU { src, dst } => self.lower_preserved_result(
                preserved::op::I31_GET_U,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::AnyConvertExtern { src, dst } => self.lower_preserved_result(
                preserved::op::ANY_CONVERT_EXTERN,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::ExternConvertAny { src, dst } => self.lower_preserved_result(
                preserved::op::EXTERN_CONVERT_ANY,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefTest { ref_type, src, dst } => {
                let encoded = ref_type.encode_to_u64();
                self.lower_preserved_result(
                    preserved::op::REF_TEST,
                    encoded as u32,
                    (encoded >> 32) as u32,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )
            }
            MachineInstKind::RefCast { ref_type, src, dst } => {
                let encoded = ref_type.encode_to_u64();
                self.lower_preserved_result(
                    preserved::op::REF_CAST,
                    encoded as u32,
                    (encoded >> 32) as u32,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )
            }
            MachineInstKind::StructNew {
                type_idx,
                fields,
                dst,
            } => self.lower_struct_new(*type_idx, fields, *dst),
            MachineInstKind::StructNewDefault { type_idx, dst } => self.lower_preserved_result(
                preserved::op::STRUCT_NEW_DEFAULT,
                *type_idx,
                0,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::StructGet {
                type_idx,
                field_idx,
                signed,
                ty,
                src,
                dst,
                dst_hi,
            } => {
                if dst_hi.is_some() {
                    return Err(WasmError::internal(
                        "x86_64 backend received pair-valued struct.get".into(),
                    ));
                }
                let op_code = match signed {
                    None => preserved::op::STRUCT_GET,
                    Some(true) => preserved::op::STRUCT_GET_S,
                    Some(false) => preserved::op::STRUCT_GET_U,
                };
                self.lower_preserved_result(
                    op_code,
                    *type_idx,
                    *field_idx,
                    *src,
                    MachineValue::Imm64(0),
                    MachineValue::Imm64(0),
                    *ty,
                    *dst,
                )
            }
            MachineInstKind::StructSet {
                type_idx,
                field_idx,
                ref_src,
                value_lo,
                value_hi,
            } => {
                if value_hi.is_some() {
                    return Err(WasmError::internal(
                        "x86_64 backend received pair-valued struct.set".into(),
                    ));
                }
                self.lower_preserved_no_result(
                    preserved::op::STRUCT_SET,
                    *type_idx,
                    *field_idx,
                    *ref_src,
                    *value_lo,
                    MachineValue::Imm64(0),
                )
            }
            MachineInstKind::ArrayNew {
                type_idx,
                init_lo,
                init_hi,
                length,
                dst,
            } => {
                if init_hi.is_some() {
                    return Err(WasmError::internal(
                        "x86_64 backend received pair-valued array.new".into(),
                    ));
                }
                self.lower_preserved_result(
                    preserved::op::ARRAY_NEW,
                    *type_idx,
                    0,
                    *init_lo,
                    *length,
                    MachineValue::Imm64(0),
                    MachineStorageType::GpWord,
                    *dst,
                )
            }
            MachineInstKind::ArrayNewDefault {
                type_idx,
                length,
                dst,
            } => self.lower_preserved_result(
                preserved::op::ARRAY_NEW_DEFAULT,
                *type_idx,
                0,
                *length,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::ArrayNewFixed {
                type_idx,
                elements,
                dst,
            } => self.lower_array_new_fixed(*type_idx, elements, *dst),
            MachineInstKind::ArrayNewData {
                type_idx,
                data_idx,
                src,
                len,
                dst,
            } => self.lower_preserved_result_extended(
                preserved::op::ARRAY_NEW_DATA,
                &[
                    (preserved::io::IMM0, *type_idx),
                    (preserved::io::IMM1, *data_idx),
                ],
                &[
                    (preserved::io::ARG0, *src),
                    (preserved::io::ARG1, *len),
                ],
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::ArrayNewElem {
                type_idx,
                elem_idx,
                src,
                len,
                dst,
            } => self.lower_preserved_result_extended(
                preserved::op::ARRAY_NEW_ELEM,
                &[
                    (preserved::io::IMM0, *type_idx),
                    (preserved::io::IMM1, *elem_idx),
                ],
                &[
                    (preserved::io::ARG0, *src),
                    (preserved::io::ARG1, *len),
                ],
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::ArrayGet {
                type_idx,
                signed,
                ty,
                ref_src,
                index,
                dst,
                dst_hi,
            } => {
                if dst_hi.is_some() {
                    return Err(WasmError::internal(
                        "x86_64 backend received pair-valued array.get".into(),
                    ));
                }
                let op_code = match signed {
                    None => preserved::op::ARRAY_GET,
                    Some(true) => preserved::op::ARRAY_GET_S,
                    Some(false) => preserved::op::ARRAY_GET_U,
                };
                self.lower_preserved_result(
                    op_code,
                    *type_idx,
                    0,
                    *ref_src,
                    *index,
                    MachineValue::Imm64(0),
                    *ty,
                    *dst,
                )
            }
            MachineInstKind::ArraySet {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
            } => {
                if value_hi.is_some() {
                    return Err(WasmError::internal(
                        "x86_64 backend received pair-valued array.set".into(),
                    ));
                }
                self.lower_preserved_no_result(
                    preserved::op::ARRAY_SET,
                    *type_idx,
                    0,
                    *ref_src,
                    *index,
                    *value_lo,
                )
            }
            MachineInstKind::ArrayFill {
                type_idx,
                ref_src,
                index,
                value_lo,
                value_hi,
                len,
            } => {
                if value_hi.is_some() {
                    return Err(WasmError::internal(
                        "x86_64 backend received pair-valued array.fill".into(),
                    ));
                }
                self.lower_preserved_no_result_extended(
                    preserved::op::ARRAY_FILL,
                    &[(preserved::io::IMM0, *type_idx)],
                    &[
                        (preserved::io::ARG0, *ref_src),
                        (preserved::io::ARG1, *index),
                        (preserved::io::ARG2, *value_lo),
                        (preserved::io::ARG3, *len),
                    ],
                )
            }
            MachineInstKind::ArrayCopy {
                dst_type_idx,
                src_type_idx,
                dst_ref,
                dst_index,
                src_ref,
                src_index,
                len,
            } => self.lower_preserved_no_result_extended(
                preserved::op::ARRAY_COPY,
                &[
                    (preserved::io::IMM0, *dst_type_idx),
                    (preserved::io::IMM1, *src_type_idx),
                ],
                &[
                    (preserved::io::ARG0, *dst_ref),
                    (preserved::io::ARG1, *dst_index),
                    (preserved::io::ARG2, *src_ref),
                    (preserved::io::ARG3, *src_index),
                    (preserved::io::ARG4, *len),
                ],
            ),
            MachineInstKind::ArrayInitData {
                type_idx,
                data_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            } => self.lower_preserved_no_result_extended(
                preserved::op::ARRAY_INIT_DATA,
                &[
                    (preserved::io::IMM0, *type_idx),
                    (preserved::io::IMM1, *data_idx),
                ],
                &[
                    (preserved::io::ARG0, *ref_src),
                    (preserved::io::ARG1, *dst_index),
                    (preserved::io::ARG2, *src_index),
                    (preserved::io::ARG3, *len),
                ],
            ),
            MachineInstKind::ArrayInitElem {
                type_idx,
                elem_idx,
                ref_src,
                dst_index,
                src_index,
                len,
            } => self.lower_preserved_no_result_extended(
                preserved::op::ARRAY_INIT_ELEM,
                &[
                    (preserved::io::IMM0, *type_idx),
                    (preserved::io::IMM1, *elem_idx),
                ],
                &[
                    (preserved::io::ARG0, *ref_src),
                    (preserved::io::ARG1, *dst_index),
                    (preserved::io::ARG2, *src_index),
                    (preserved::io::ARG3, *len),
                ],
            ),
            MachineInstKind::ArrayLen { src, dst } => self.lower_preserved_result(
                preserved::op::ARRAY_LEN,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
        }
    }
    // ── Move / const ────────────────────────────────────────────────────────

    pub(super) fn lower_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            match src {
                MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    let src_width = self.core.fp_reg_width(src_reg)?;
                    if src_width != width {
                        return Err(WasmError::invalid(
                            "x86_64 typed float move width mismatch: dst expects , src is",
                        ));
                    }
                    if dst_fp != src_fp {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, src_fp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, src_fp)
                            }
                        };
                    }
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp)
                        }
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    // Load from the function's literal pool: one FP-domain
                    // load the core can hoist, instead of a GP
                    // materialization plus a domain-crossing move on the
                    // consumer's critical path.
                    let bits = match width {
                        MachineFloatWidth::F32 => u64::from(value as u32),
                        MachineFloatWidth::F64 => value,
                    };
                    let literal_index = self.intern_fp_literal(bits);
                    let disp_offset = match width {
                        MachineFloatWidth::F32 => enc::movss_rip(&mut self.core.text, dst_fp),
                        MachineFloatWidth::F64 => enc::movsd_rip(&mut self.core.text, dst_fp),
                    };
                    self.fp_literal_fixups.push(FpLiteralFixup {
                        disp_offset,
                        literal_index,
                    });
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
                    "x86_64 Move cannot consume reserved cache register",
                )),
            }
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            match src {
                MachineValue::Reg(src_reg) if self.core.is_fp_reg(src_reg) => {
                    let src_fp = self.map_fp_reg(src_reg)? as u8;
                    match self.core.fp_reg_width(src_reg)? {
                        MachineFloatWidth::F32 => {
                            enc::movd_r32_xmm(&mut self.core.text, dst_gp, src_fp)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_r64_xmm(&mut self.core.text, dst_gp, src_fp)
                        }
                    };
                    Ok(())
                }
                MachineValue::Reg(src_reg) => {
                    let src_gp = self.map_gp_reg(src_reg)?;
                    if dst_gp != src_gp {
                        self.emit_gp_move_ty(ty, dst_gp, src_gp)?;
                    }
                    Ok(())
                }
                MachineValue::Imm64(value) => {
                    self.materialize_u64(dst_gp, value);
                    Ok(())
                }
                MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
                    "x86_64 Move cannot consume reserved cache register",
                )),
            }
        }
    }

    pub(super) fn lower_float_const(
        &mut self,
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    ) -> Result<(), WasmError> {
        if !self.core.is_fp_reg(dst) {
            return Err(WasmError::invalid(
                "x86_64 FloatConst destination must be an FP register",
            ));
        }
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let imm = match width {
            MachineFloatWidth::F32 => u64::from(bits as u32),
            MachineFloatWidth::F64 => bits,
        };
        if imm == 0 {
            // XORPS/XORPD xmm, xmm → zero
            match width {
                MachineFloatWidth::F32 => enc::xorps(&mut self.core.text, dst_fp, dst_fp),
                MachineFloatWidth::F64 => enc::xorpd(&mut self.core.text, dst_fp, dst_fp),
            };
        } else {
            // Load from the function's literal pool: one FP-domain load the
            // core can hoist, instead of a GP materialization plus a
            // domain-crossing move on the consumer's critical path.
            let literal_index = self.intern_fp_literal(imm);
            let disp_offset = match width {
                MachineFloatWidth::F32 => enc::movss_rip(&mut self.core.text, dst_fp),
                MachineFloatWidth::F64 => enc::movsd_rip(&mut self.core.text, dst_fp),
            };
            self.fp_literal_fixups.push(FpLiteralFixup {
                disp_offset,
                literal_index,
            });
        }
        self.core.set_fp_reg_width(dst, width)?;
        Ok(())
    }

    // ── Load / Store ─────────────────────────────────────────────────────────

    pub(super) fn lower_load(
        &mut self,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        self.lower_load_from(dst, base, addr.offset, width, extension)
    }

    #[cfg(sf_has_guard_pages)]
    pub(super) fn emit_stack_probe(&mut self, addr: MachineAddr) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        enc::load_64(&mut self.core.text, X86Reg::RAX, base, addr.offset);
        Ok(())
    }

    /// Load from a physical base register + displacement.
    fn lower_load_from(
        &mut self,
        dst: MachineReg,
        base: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        // FP register destination
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            let tracked_width = match width {
                MachineMemWidth::U32 => MachineFloatWidth::F32,
                MachineMemWidth::U64 => MachineFloatWidth::F64,
                _ => return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support narrow integer loads into FP regs"
                        .into(),
                )),
            };
            match tracked_width {
                MachineFloatWidth::F32 => enc::movss_load(&mut self.core.text, dst_fp, base, disp),
                MachineFloatWidth::F64 => enc::movsd_load(&mut self.core.text, dst_fp, base, disp),
            };
            self.core.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }

        // GP register destination
        let dst_gp = self.map_gp_reg(dst)?;
        match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::load_u8(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::load_s8_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::load_u16(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::load_s16_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::load_32(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::load_s32_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::load_64(&mut self.core.text, dst_gp, base, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support sign-extending 64-bit loads".into(),
                ))
            }
        };
        Ok(())
    }

    /// Emit one fused `dst <- dst OP [mem]` instruction for a proven
    /// load + ALU pair. The loaded transient is never written; the ALU
    /// reads memory directly at the same program point, so trap order
    /// and flags behavior match the unfused sequence.
    pub(super) fn lower_load_alu(
        &mut self,
        fusion: &super::fusion::LoadAluFusion,
    ) -> Result<(), WasmError> {
        use super::fusion::LoadAluMem;
        let dst = self.map_gp_reg(fusion.dst)?;
        match &fusion.mem {
            LoadAluMem::Base(addr) => {
                let base = self.map_gp_reg(addr.base)?;
                if fusion.w32 {
                    enc::alu_rm_32(&mut self.core.text, fusion.op, dst, base, addr.offset);
                } else {
                    enc::alu_rm_64(&mut self.core.text, fusion.op, dst, base, addr.offset);
                }
            }
            LoadAluMem::Indexed {
                base,
                index,
                extend,
                offset,
            } => {
                let base_x86 = self.map_gp_reg(*base)?;
                let index_x86 = self.map_gp_reg(*index)?;
                // Scratch guards live to the end of the function; the
                // ZeroExtend32 path zero-extends into one of them.
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let idx = match extend {
                    MachineIndexExtend::None => index_x86,
                    MachineIndexExtend::ZeroExtend32 => {
                        let idx = if base_x86 == *scratch0 {
                            *scratch1
                        } else {
                            *scratch0
                        };
                        debug_assert!(
                            idx != dst,
                            "x86_64 load+ALU fusion scratch must not alias the ALU destination"
                        );
                        enc::mov_rr_32(&mut self.core.text, idx, index_x86);
                        idx
                    }
                };
                if fusion.w32 {
                    enc::alu_rm_32_idx(&mut self.core.text, fusion.op, dst, base_x86, idx, *offset);
                } else {
                    enc::alu_rm_64_idx(&mut self.core.text, fusion.op, dst, base_x86, idx, *offset);
                }
            }
        }
        if fusion.w32 {
            self.note_flags32(dst);
        }
        Ok(())
    }

    pub(super) fn lower_memory_update(
        &mut self,
        update: &super::fusion::MemoryUpdateFusion,
    ) -> Result<(), WasmError> {
        use super::fusion::LoadAluMem;
        // Flags describe memory now, not any physical result register.
        self.flags32 = None;
        match update.mem {
            LoadAluMem::Base(addr) => {
                let base = self.map_gp_reg(addr.base)?;
                enc::alu_mi(
                    &mut self.core.text,
                    update.op,
                    !update.w32,
                    base,
                    None,
                    addr.offset,
                    update.immediate,
                );
            }
            LoadAluMem::Indexed {
                base,
                index,
                extend,
                offset,
            } => {
                let base = self.map_gp_reg(base)?;
                let index = self.map_gp_reg(index)?;
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let index = match extend {
                    MachineIndexExtend::None => index,
                    MachineIndexExtend::ZeroExtend32 => {
                        let scratch = if base == *scratch0 {
                            *scratch1
                        } else {
                            *scratch0
                        };
                        enc::mov_rr_32(&mut self.core.text, scratch, index);
                        scratch
                    }
                };
                enc::alu_mi(
                    &mut self.core.text,
                    update.op,
                    !update.w32,
                    base,
                    Some(index),
                    offset,
                    update.immediate,
                );
            }
        }
        Ok(())
    }

    /// Load from [base + index + disp]; the indexed twin of
    /// [`Self::lower_load_from`].
    fn lower_load_from_idx(
        &mut self,
        dst: MachineReg,
        base: X86Reg,
        index: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        // FP register destination
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            let tracked_width = match width {
                MachineMemWidth::U32 => MachineFloatWidth::F32,
                MachineMemWidth::U64 => MachineFloatWidth::F64,
                _ => return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support narrow integer loads into FP regs"
                        .into(),
                )),
            };
            match tracked_width {
                MachineFloatWidth::F32 => {
                    enc::movss_load_idx(&mut self.core.text, dst_fp, base, index, disp)
                }
                MachineFloatWidth::F64 => {
                    enc::movsd_load_idx(&mut self.core.text, dst_fp, base, index, disp)
                }
            };
            self.core.set_fp_reg_width(dst, tracked_width)?;
            return Ok(());
        }

        // GP register destination
        let dst_gp = self.map_gp_reg(dst)?;
        match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::None)
            | (MachineMemWidth::U8, MachineLoadExtension::ZeroExtend) => {
                enc::load_u8_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => {
                enc::load_s8_64_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::None)
            | (MachineMemWidth::U16, MachineLoadExtension::ZeroExtend) => {
                enc::load_u16_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => {
                enc::load_s16_64_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::None)
            | (MachineMemWidth::U32, MachineLoadExtension::ZeroExtend) => {
                enc::load_32_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => {
                enc::load_s32_64_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::None)
            | (MachineMemWidth::U64, MachineLoadExtension::ZeroExtend) => {
                enc::load_64_idx(&mut self.core.text, dst_gp, base, index, disp);
            }
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "x86_64 MachineIR backend does not support sign-extending 64-bit loads".into(),
                ))
            }
        };
        Ok(())
    }

    /// Store to [base + index + disp]; the indexed twin of
    /// [`Self::lower_store_to_with_scratch`]. `materialize_scratch` must
    /// differ from both address registers.
    fn lower_store_to_idx(
        &mut self,
        base: X86Reg,
        index: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        src: MachineValue,
        materialize_scratch: X86Reg,
    ) -> Result<(), WasmError> {
        // FP register source
        if let MachineValue::Reg(src_reg) = src {
            if self.core.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)? as u8;
                match width {
                    MachineMemWidth::U32 => {
                        enc::movss_store_idx(&mut self.core.text, base, index, disp, src_fp)
                    }
                    MachineMemWidth::U64 => {
                        enc::movsd_store_idx(&mut self.core.text, base, index, disp, src_fp)
                    }
                    _ => {
                        return Err(WasmError::invalid(
                            "x86_64 MachineIR backend does not support narrow FP stores".into(),
                        ))
                    }
                };
                return Ok(());
            }
        }

        // Imm64(0) store → store_imm32_64_idx for U64
        if matches!(src, MachineValue::Imm64(0)) && width == MachineMemWidth::U64 {
            enc::store_imm32_64_idx(&mut self.core.text, base, index, disp, 0);
            return Ok(());
        }

        let src_gp = self.materialize_value(materialize_scratch, src)?;
        match width {
            MachineMemWidth::U8 => enc::store_8_idx(&mut self.core.text, base, index, disp, src_gp),
            MachineMemWidth::U16 => {
                enc::store_16_idx(&mut self.core.text, base, index, disp, src_gp)
            }
            MachineMemWidth::U32 => {
                enc::store_32_idx(&mut self.core.text, base, index, disp, src_gp)
            }
            MachineMemWidth::U64 => {
                enc::store_64_idx(&mut self.core.text, base, index, disp, src_gp)
            }
        };
        Ok(())
    }

    pub(super) fn lower_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        self.lower_store_to(base, addr.offset, width, src)
    }

    /// Store to a physical base register + displacement.
    fn lower_store_to(
        &mut self,
        base: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        self.lower_store_to_with_scratch(base, disp, width, src, *scratch0, *scratch1)
    }

    fn lower_store_to_with_scratch(
        &mut self,
        base: X86Reg,
        disp: i32,
        width: MachineMemWidth,
        src: MachineValue,
        scratch0: X86Reg,
        scratch1: X86Reg,
    ) -> Result<(), WasmError> {
        // FP register source
        if let MachineValue::Reg(src_reg) = src {
            if self.core.is_fp_reg(src_reg) {
                let src_fp = self.map_fp_reg(src_reg)? as u8;
                match width {
                    MachineMemWidth::U32 => {
                        enc::movss_store(&mut self.core.text, base, disp, src_fp)
                    }
                    MachineMemWidth::U64 => {
                        enc::movsd_store(&mut self.core.text, base, disp, src_fp)
                    }
                    _ => {
                        return Err(WasmError::invalid(
                            "x86_64 MachineIR backend does not support narrow FP stores".into(),
                        ))
                    }
                };
                return Ok(());
            }
        }

        // Imm64(0) store → store_imm32_64 for U64
        if matches!(src, MachineValue::Imm64(0)) && width == MachineMemWidth::U64 {
            enc::store_imm32_64(&mut self.core.text, base, disp, 0);
            return Ok(());
        }

        // Indexed stores compute the effective address into gp_scratch[0]
        // before calling lower_store_to(). Do not reuse that register to
        // materialize the source or the base address will be lost.
        let materialize_scratch = if base == scratch0 { scratch1 } else { scratch0 };
        let src_gp = self.materialize_value(materialize_scratch, src)?;
        // Snapshot after materialization: loading an immediate zero can use
        // XOR and invalidate the producer's ZF. The MOV store itself changes
        // neither flags nor any register, including the producer's value.
        let flags32 = self.current_flags32();
        match width {
            MachineMemWidth::U8 => enc::store_8(&mut self.core.text, base, disp, src_gp),
            MachineMemWidth::U16 => enc::store_16(&mut self.core.text, base, disp, src_gp),
            MachineMemWidth::U32 => enc::store_32(&mut self.core.text, base, disp, src_gp),
            MachineMemWidth::U64 => enc::store_64(&mut self.core.text, base, disp, src_gp),
        };
        if let Some(reg) = flags32 {
            self.note_flags32(reg);
        }
        Ok(())
    }

    // ── Integer unary ops ────────────────────────────────────────────────────

    pub(super) fn lower_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let scratch = self.gp_scratch.scoped_alloc().detach();
        let src = self.materialize_value(*scratch, src)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Clz) => {
                enc::lzcnt_rr_64(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Ctz) => {
                enc::tzcnt_rr_64(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Popcnt) => {
                enc::popcnt_rr_64(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r32_r8(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r32_r16(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                enc::movsx_r64_r8(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                enc::movsx_r64_r16(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => {
                enc::movsxd_r64_r32(&mut self.core.text, dst, src);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => {
                // i32.extend32_s is a nop (already 32-bit)
                if dst != src {
                    enc::mov_rr_32(&mut self.core.text, dst, src);
                }
            }
        }
        Ok(())
    }

    // ── Integer binary ops ───────────────────────────────────────────────────

    pub(super) fn lower_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;

        // x86 two-operand model: dst = dst OP src. So we need lhs in dst first.
        match op {
            MachineIntBinaryOp::Add
            | MachineIntBinaryOp::Sub
            | MachineIntBinaryOp::And
            | MachineIntBinaryOp::Or
            | MachineIntBinaryOp::Xor => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                // Try immediate form: dst = lhs OP imm32
                if let MachineValue::Imm64(imm_val) = rhs {
                    let imm = imm_val as i64 as i32;
                    if imm as i64 == imm_val as i64
                        || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                    {
                        let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                        if dst != lhs_gp && width == MachineIntWidth::I64 {
                            // Keep two-operand ADD/SUB when already coalesced.
                            // The i64 form benefits from replacing its copy
                            // plus arithmetic. Keep i32 immediate ADD/SUB:
                            // the shorter form regressed native x64 execution.
                            // SUB must not negate MIN into a signed disp32.
                            let displacement = match op {
                                MachineIntBinaryOp::Add => Some(imm),
                                MachineIntBinaryOp::Sub => imm.checked_neg(),
                                _ => None,
                            };
                            if let Some(displacement) = displacement {
                                enc::lea_offset(
                                    &mut self.core.text,
                                    width == MachineIntWidth::I64,
                                    dst,
                                    lhs_gp,
                                    displacement,
                                );
                                // LEA does not produce result flags. Its emitted
                                // bytes invalidate flags32's position stamp.
                                return Ok(());
                            }
                        }
                        if dst != lhs_gp {
                            self.emit_gp_move_width(width, dst, lhs_gp);
                        }
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                                enc::sub_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_ri_32(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_64(&mut self.core.text, dst, imm)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_ri_32(&mut self.core.text, dst, imm)
                            }
                            _ => unreachable!(),
                        };
                        if width == MachineIntWidth::I32 {
                            self.note_flags32(dst);
                        }
                        return Ok(());
                    }
                }
                let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                let rhs_gp = self.materialize_value(*scratch1, rhs)?;
                if op == MachineIntBinaryOp::Add && dst != lhs_gp && dst != rhs_gp {
                    enc::lea_sum(
                        &mut self.core.text,
                        width == MachineIntWidth::I64,
                        dst,
                        lhs_gp,
                        rhs_gp,
                    );
                    return Ok(());
                }
                // Handle aliasing: if dst == rhs_gp but dst != lhs_gp,
                // mov dst, lhs would clobber rhs before the operation.
                if dst == rhs_gp && dst != lhs_gp {
                    if op == MachineIntBinaryOp::Sub {
                        // Sub is not commutative: compute in scratch
                        self.emit_gp_move_width(width, *scratch0, lhs_gp);
                        match width {
                            MachineIntWidth::I64 => {
                                enc::sub_rr_64(&mut self.core.text, *scratch0, rhs_gp)
                            }
                            MachineIntWidth::I32 => {
                                enc::sub_rr_32(&mut self.core.text, *scratch0, rhs_gp)
                            }
                        };
                        self.emit_gp_move_width(width, dst, *scratch0);
                    } else {
                        // Commutative: swap operands — do dst = rhs OP lhs
                        match (width, op) {
                            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                                enc::add_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                                enc::add_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                                enc::and_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                                enc::and_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                                enc::or_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                                enc::or_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_64(&mut self.core.text, dst, lhs_gp)
                            }
                            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                                enc::xor_rr_32(&mut self.core.text, dst, lhs_gp)
                            }
                            _ => unreachable!(),
                        };
                    }
                } else {
                    if dst != lhs_gp {
                        self.emit_gp_move_width(width, dst, lhs_gp);
                    }
                    match (width, op) {
                        (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                            enc::add_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                            enc::add_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                            enc::sub_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                            enc::and_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                            enc::and_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                            enc::or_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                            enc::or_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_64(&mut self.core.text, dst, rhs_gp)
                        }
                        (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                            enc::xor_rr_32(&mut self.core.text, dst, rhs_gp)
                        }
                        _ => unreachable!(),
                    };
                }
                // Every path above leaves dst's 32-bit result flags in
                // EFLAGS: either the ALU op wrote dst last, or its result
                // was moved into dst and mov does not touch flags.
                if width == MachineIntWidth::I32 {
                    self.note_flags32(dst);
                }
                Ok(())
            }
            MachineIntBinaryOp::Mul => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                let rhs_gp = self.materialize_value(*scratch1, rhs)?;
                if dst == rhs_gp && dst != lhs_gp {
                    // IMUL is commutative: dst already has rhs, just mul by lhs
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.core.text, dst, lhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.core.text, dst, lhs_gp),
                    };
                } else {
                    if dst != lhs_gp {
                        self.emit_gp_move_width(width, dst, lhs_gp);
                    }
                    match width {
                        MachineIntWidth::I64 => enc::imul_rr_64(&mut self.core.text, dst, rhs_gp),
                        MachineIntWidth::I32 => enc::imul_rr_32(&mut self.core.text, dst, rhs_gp),
                    };
                }
                Ok(())
            }
            MachineIntBinaryOp::Shl
            | MachineIntBinaryOp::ShrS
            | MachineIntBinaryOp::ShrU
            | MachineIntBinaryOp::Rotl
            | MachineIntBinaryOp::Rotr => self.lower_shift_op(width, op, dst, lhs, rhs),
            MachineIntBinaryOp::DivS
            | MachineIntBinaryOp::DivU
            | MachineIntBinaryOp::RemS
            | MachineIntBinaryOp::RemU => self.lower_div_rem(width, op, dst, lhs, rhs),
        }
    }

    /// Emit shift/rotate: on x86_64, shift amount must be in CL.
    fn lower_shift_op(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // For immediate shift amounts, use the imm8 form (no RCX needed).
        if let MachineValue::Imm64(amount) = rhs {
            let scratch0 = self.gp_scratch.scoped_alloc().detach();
            let lhs_gp = self.materialize_value(*scratch0, lhs)?;
            if dst != lhs_gp {
                self.emit_gp_move_width(width, dst, lhs_gp);
            }
            let imm = (amount & 0x3F) as u8; // mask to 6 bits (x86 does this anyway)
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                    enc::shl_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                    enc::sar_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                    enc::shr_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                    enc::rol_imm_32(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_64(&mut self.core.text, dst, imm)
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                    enc::ror_imm_32(&mut self.core.text, dst, imm)
                }
                _ => unreachable!(),
            };
            return Ok(());
        }
        // Variable shift: RCX is scratch-only on x86_64, so own it explicitly
        // rather than borrowing a dynamic register and restoring it later.
        let rcx = self.gp_scratch.claim_rcx().detach();
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let lhs_gp = self.materialize_value(*scratch0, lhs)?;
        let rhs_gp = self.materialize_value(*rcx, rhs)?;
        if rhs_gp == dst && dst != lhs_gp {
            if rhs_gp != *rcx {
                enc::mov_rr_64(&mut self.core.text, *rcx, rhs_gp);
            }
            if dst != lhs_gp {
                self.emit_gp_move_width(width, dst, lhs_gp);
            }
        } else {
            if dst != lhs_gp {
                self.emit_gp_move_width(width, dst, lhs_gp);
            }
            if rhs_gp != *rcx {
                enc::mov_rr_64(&mut self.core.text, *rcx, rhs_gp);
            }
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                enc::shl_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                enc::shl_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                enc::sar_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                enc::sar_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                enc::shr_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                enc::shr_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                enc::rol_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                enc::rol_cl_32(&mut self.core.text, dst)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                enc::ror_cl_64(&mut self.core.text, dst)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                enc::ror_cl_32(&mut self.core.text, dst)
            }
            _ => unreachable!(),
        };
        Ok(())
    }

    /// Emit div/rem: x86_64 uses RAX:RDX for dividend, result in RAX (quot) / RDX (rem).
    fn lower_div_rem(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: X86Reg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        // div/idiv implicitly uses RAX and RDX. On x86_64 these are scratch-only
        // backend registers, so claim ownership explicitly before touching them.
        let rax = self.gp_scratch.claim_rax().detach();
        let rdx = self.gp_scratch.claim_rdx().detach();
        let scratch0 = self.gp_scratch.scoped_alloc().detach();

        let lhs_gp = self.materialize_value(*rax, lhs)?;
        if lhs_gp != *rax {
            enc::mov_rr_64(&mut self.core.text, *rax, lhs_gp);
        }
        let rhs_gp = self.materialize_value(*scratch0, rhs)?;
        let divisor = if rhs_gp == *rax || rhs_gp == *rdx {
            enc::mov_rr_64(&mut self.core.text, *scratch0, rhs_gp);
            *scratch0
        } else {
            rhs_gp
        };

        // Division-by-zero check: divisor == 0 => trap
        enc::test_rr_64(&mut self.core.text, divisor, divisor);
        let div_zero_label = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_jcc(Cc::E, div_zero_label);

        match op {
            MachineIntBinaryOp::DivS => {
                // Signed overflow check: MIN / -1 => IntegerOverflow trap
                let not_min = self.core.new_label();
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.core.text, *rax, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(*rdx, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.core.text, *rax, *rdx);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                // Compare divisor against -1 using matching width
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, divisor, -1),
                };
                let overflow_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_jcc(Cc::E, overflow_label);
                self.core.bind_label(not_min);
                // Sign-extend RAX → RDX:RAX, then IDIV
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.core.text);
                        enc::idiv_rm_64(&mut self.core.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.core.text);
                        enc::idiv_rm_32(&mut self.core.text, divisor);
                    }
                };
            }
            MachineIntBinaryOp::RemS => {
                // MIN % -1 = 0 (no trap, just skip the div)
                let not_min = self.core.new_label();
                let done = self.core.new_label();
                match width {
                    MachineIntWidth::I32 => {
                        enc::cmp_ri_32(&mut self.core.text, *rax, i32::MIN);
                    }
                    MachineIntWidth::I64 => {
                        self.materialize_u64(*rdx, i64::MIN as u64);
                        enc::cmp_rr_64(&mut self.core.text, *rax, *rdx);
                    }
                };
                self.emit_jcc(Cc::NE, not_min);
                match width {
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, divisor, -1),
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, divisor, -1),
                };
                self.emit_jcc(Cc::NE, not_min);
                // MIN % -1 = 0
                enc::xor_rr_32(&mut self.core.text, *rdx, *rdx);
                self.emit_jmp(done);
                self.core.bind_label(not_min);
                match width {
                    MachineIntWidth::I64 => {
                        enc::cqo(&mut self.core.text);
                        enc::idiv_rm_64(&mut self.core.text, divisor);
                    }
                    MachineIntWidth::I32 => {
                        enc::cdq(&mut self.core.text);
                        enc::idiv_rm_32(&mut self.core.text, divisor);
                    }
                };
                self.core.bind_label(done);
            }
            MachineIntBinaryOp::DivU | MachineIntBinaryOp::RemU => {
                // Zero-extend: XOR RDX, RDX
                enc::xor_rr_32(&mut self.core.text, *rdx, *rdx);
                match width {
                    MachineIntWidth::I64 => enc::div_rm_64(&mut self.core.text, divisor),
                    MachineIntWidth::I32 => enc::div_rm_32(&mut self.core.text, divisor),
                };
            }
            _ => unreachable!(),
        }

        // Result: quotient in RAX, remainder in RDX
        let result_reg = match op {
            MachineIntBinaryOp::DivS | MachineIntBinaryOp::DivU => *rax,
            MachineIntBinaryOp::RemS | MachineIntBinaryOp::RemU => *rdx,
            _ => unreachable!(),
        };
        if dst != result_reg {
            self.emit_gp_move_width(width, dst, result_reg);
        }
        Ok(())
    }

    // ── Integer compare ──────────────────────────────────────────────────────

    pub(super) fn lower_int_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let cc = map_int_cond(kind, sign);
        // Preferred form: zero `dst` before the compare, then bare SETcc —
        // the xor is a zero idiom (renamed, off the flag/critical path),
        // kills SETcc's false dependency on the old register value, and
        // makes the movzx redundant. Requires `dst` to not carry a compare
        // operand.
        if !self.value_maps_to_gp(&lhs, dst) && !self.value_maps_to_gp(&rhs, dst) {
            enc::xor_rr_32(&mut self.core.text, dst, dst);
            self.lower_cmp_values(width, lhs, rhs)?;
            enc::setcc(&mut self.core.text, cc, dst);
            return Ok(());
        }
        self.lower_cmp_values(width, lhs, rhs)?;
        enc::setcc(&mut self.core.text, cc, dst);
        enc::movzx_r32_r8(&mut self.core.text, dst, dst);
        Ok(())
    }

    /// True when `value` is a register operand that maps to the physical
    /// GP register `phys`. Reserved cache registers are resolved through
    /// the same mapping; unmappable values conservatively report `true`
    /// so callers fall back to alias-safe forms.
    fn value_maps_to_gp(&self, value: &MachineValue, phys: X86Reg) -> bool {
        match value {
            MachineValue::Imm64(_) => false,
            MachineValue::Reg(reg) | MachineValue::ReservedReg(reg) => {
                self.map_gp_reg(*reg).map(|r| r == phys).unwrap_or(true)
            }
        }
    }

    pub(super) fn lower_cmp_values(
        &mut self,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        // Immediate form
        if let MachineValue::Imm64(imm_val) = rhs {
            let imm = imm_val as i64 as i32;
            if imm as i64 == imm_val as i64
                || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
            {
                let lhs_gp = self.materialize_value(*scratch0, lhs)?;
                match width {
                    MachineIntWidth::I64 => enc::cmp_ri_64(&mut self.core.text, lhs_gp, imm),
                    MachineIntWidth::I32 => enc::cmp_ri_32(&mut self.core.text, lhs_gp, imm),
                };
                return Ok(());
            }
        }
        let lhs_gp = self.materialize_value(*scratch0, lhs)?;
        let rhs_gp = self.materialize_value(*scratch1, rhs)?;
        match width {
            MachineIntWidth::I64 => enc::cmp_rr_64(&mut self.core.text, lhs_gp, rhs_gp),
            MachineIntWidth::I32 => enc::cmp_rr_32(&mut self.core.text, lhs_gp, rhs_gp),
        };
        Ok(())
    }

    /// Emit TEST (bitwise AND setting flags, discarding result).
    pub(super) fn lower_tst_values(
        &mut self,
        width: MachineIntWidth,
        src: MachineValue,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        let src_gp = self.materialize_value(*scratch0, src)?;
        match mask {
            MachineValue::Imm64(imm_val) => {
                let imm = imm_val as i64 as i32;
                if imm as i64 == imm_val as i64
                    || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                {
                    match width {
                        MachineIntWidth::I64 => enc::test_ri_64(&mut self.core.text, src_gp, imm),
                        MachineIntWidth::I32 => enc::test_ri_32(&mut self.core.text, src_gp, imm),
                    }
                } else {
                    let mask_gp =
                        self.materialize_value(*scratch1, MachineValue::Imm64(imm_val))?;
                    match width {
                        MachineIntWidth::I64 => {
                            enc::test_rr_64(&mut self.core.text, src_gp, mask_gp)
                        }
                        MachineIntWidth::I32 => {
                            enc::test_rr_32(&mut self.core.text, src_gp, mask_gp)
                        }
                    }
                }
            }
            MachineValue::Reg(_) => {
                let mask_gp = self.materialize_value(*scratch1, mask)?;
                match width {
                    MachineIntWidth::I64 => enc::test_rr_64(&mut self.core.text, src_gp, mask_gp),
                    MachineIntWidth::I32 => enc::test_rr_32(&mut self.core.text, src_gp, mask_gp),
                }
            }
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "x86_64 TestBits cannot read reserved cache register",
                ));
            }
        }
        Ok(())
    }

    // ── Bitfield extract (decomposed to SHR + AND) ────────────────────────────

    fn lower_bitfield_extract_u(
        &mut self,
        width: MachineIntWidth,
        dst: MachineReg,
        src: MachineReg,
        lsb: u8,
        bits: u8,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.map_gp_reg(src)?;
        // dst = (src >> lsb) & ((1 << bits) - 1)
        if lsb != 0 && super::cpu::prefer_bextr() {
            let scratch = self.gp_scratch.scoped_alloc().detach();
            let control = self.materialize_value(
                *scratch,
                MachineValue::Imm64((u64::from(bits) << 8) | u64::from(lsb)),
            )?;
            enc::bextr_rrr(
                &mut self.core.text,
                width == MachineIntWidth::I64,
                dst,
                src,
                control,
            );
            return Ok(());
        }
        if dst != src {
            self.emit_gp_move_width(width, dst, src);
        }
        if lsb > 0 {
            match width {
                MachineIntWidth::I64 => enc::shr_imm_64(&mut self.core.text, dst, lsb),
                MachineIntWidth::I32 => enc::shr_imm_32(&mut self.core.text, dst, lsb),
            }
        }
        let mask = (1u64 << bits) - 1;
        match width {
            MachineIntWidth::I64 => {
                // `and r64, imm32` sign-extends its immediate, so only use the
                // immediate form when it encodes the exact 64-bit mask.
                let imm = mask as i64 as i32;
                if imm as i64 as u64 == mask {
                    enc::and_ri_64(&mut self.core.text, dst, imm);
                } else {
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    let mask_gp = self.materialize_value(*scratch, MachineValue::Imm64(mask))?;
                    enc::and_rr_64(&mut self.core.text, dst, mask_gp);
                }
            }
            MachineIntWidth::I32 => {
                let imm = mask as u32 as i32;
                enc::and_ri_32(&mut self.core.text, dst, imm);
            }
        }
        Ok(())
    }

    // ── Shifted-register binary (decomposed to shift + op) ──────────────────

    fn lower_int_binary_shifted(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineReg,
        shift: MachineShiftOp,
        amount: u8,
    ) -> Result<(), WasmError> {
        // Decompose: dst = lhs OP (rhs SHIFT amount)
        // Step 1: shift rhs into scratch
        let dst = self.map_gp_reg(dst)?;
        let lhs = self.map_gp_reg(lhs)?;
        let rhs = self.map_gp_reg(rhs)?;
        let scratch = self.gp_scratch.scoped_alloc().detach();
        if *scratch != rhs {
            self.emit_gp_move_width(width, *scratch, rhs);
        }
        match (width, shift) {
            (MachineIntWidth::I64, MachineShiftOp::Lsl) => {
                enc::shl_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Lsl) => {
                enc::shl_imm_32(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I64, MachineShiftOp::Lsr) => {
                enc::shr_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Lsr) => {
                enc::shr_imm_32(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I64, MachineShiftOp::Asr) => {
                enc::sar_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Asr) => {
                enc::sar_imm_32(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I64, MachineShiftOp::Ror) => {
                enc::ror_imm_64(&mut self.core.text, *scratch, amount)
            }
            (MachineIntWidth::I32, MachineShiftOp::Ror) => {
                enc::ror_imm_32(&mut self.core.text, *scratch, amount)
            }
        }
        // Step 2: dst = lhs OP scratch
        if dst != lhs {
            self.emit_gp_move_width(width, dst, lhs);
        }
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                enc::add_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                enc::add_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                enc::sub_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                enc::sub_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                enc::and_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                enc::and_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                enc::or_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                enc::or_rr_32(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                enc::xor_rr_64(&mut self.core.text, dst, *scratch)
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                enc::xor_rr_32(&mut self.core.text, dst, *scratch)
            }
            _ => return Err(WasmError::internal("IntBinaryShifted: unsupported op")),
        }
        Ok(())
    }

    // ── Test bits (TEST + SETCC) ────────────────────────────────────────────

    fn lower_test_bits(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        src: MachineReg,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src = self.map_gp_reg(src)?;
        // Same pre-zeroed SETcc form as `lower_int_compare`: valid when
        // `dst` carries neither TEST operand.
        let prezeroed = dst != src && !self.value_maps_to_gp(&mask, dst);
        if prezeroed {
            enc::xor_rr_32(&mut self.core.text, dst, dst);
        }
        let scratch = self.gp_scratch.scoped_alloc().detach();
        // Emit TEST to set flags.
        match mask {
            MachineValue::Imm64(imm_val) => {
                let imm = imm_val as i64 as i32;
                if imm as i64 == imm_val as i64
                    || (width == MachineIntWidth::I32 && imm_val as u32 as i32 == imm)
                {
                    match width {
                        MachineIntWidth::I64 => enc::test_ri_64(&mut self.core.text, src, imm),
                        MachineIntWidth::I32 => enc::test_ri_32(&mut self.core.text, src, imm),
                    }
                } else {
                    // Doesn't fit i32 — materialize and use register form.
                    self.materialize_u64(*scratch, imm_val);
                    match width {
                        MachineIntWidth::I64 => enc::test_rr_64(&mut self.core.text, src, *scratch),
                        MachineIntWidth::I32 => enc::test_rr_32(&mut self.core.text, src, *scratch),
                    }
                }
            }
            MachineValue::Reg(mask_reg) => {
                let mask_gp = self.map_gp_reg(mask_reg)?;
                match width {
                    MachineIntWidth::I64 => enc::test_rr_64(&mut self.core.text, src, mask_gp),
                    MachineIntWidth::I32 => enc::test_rr_32(&mut self.core.text, src, mask_gp),
                }
            }
            MachineValue::ReservedReg(_reg) => {
                return Err(WasmError::internal(
                    "x86_64 TestBits mask cannot be reserved cache register",
                ));
            }
        }
        // TST sets Z flag: Eq → ZF=1 (test was zero), Ne → ZF=0 (test was nonzero).
        let cc = match kind {
            MachineCompareKind::Eq => Cc::E,
            MachineCompareKind::Ne => Cc::NE,
            _ => return Err(WasmError::internal("TestBits: unsupported compare kind")),
        };
        enc::setcc(&mut self.core.text, cc, dst);
        if !prezeroed {
            enc::movzx_r32_r8(&mut self.core.text, dst, dst);
        }
        Ok(())
    }

    // ── Select ───────────────────────────────────────────────────────────────

    pub(super) fn lower_select(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            match cond {
                MachineValue::Imm64(value) => {
                    let selected = if value != 0 { on_true } else { on_false };
                    return self.lower_move(ty, dst, selected);
                }
                MachineValue::Reg(reg) => {
                    let cond_gp = self.map_gp_reg(reg)?;
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    let gp0 = self.gp_scratch.scoped_alloc().detach();
                    let gp1 = self.gp_scratch.scoped_alloc().detach();
                    let fp0 = self.fp_scratch.scoped_alloc().detach();
                    let fp1 = self.fp_scratch.scoped_alloc().detach();
                    let false_label = self.core.new_label();
                    let done = self.core.new_label();
                    // Wasm select conditions are i32 values; ignore any stale
                    // upper half that may remain in a GpWord carrier.
                    if !self.flags32_current(cond_gp) {
                        enc::test_rr_32(&mut self.core.text, cond_gp, cond_gp);
                    }
                    self.emit_jcc(Cc::E, false_label);
                    let true_fp = self.prepare_float_operand(width, on_true, *gp0, *fp0)?;
                    if dst_fp != true_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, true_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, true_fp as u8)
                            }
                        };
                    }
                    self.emit_jmp(done);
                    self.core.bind_label(false_label);
                    let false_fp = self.prepare_float_operand(width, on_false, *gp1, *fp1)?;
                    if dst_fp != false_fp as u8 {
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, false_fp as u8)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movaps_rr(&mut self.core.text, dst_fp, false_fp as u8)
                            }
                        };
                    }
                    self.core.bind_label(done);
                    self.core.set_fp_reg_width(dst, width)?;
                    Ok(())
                }
                MachineValue::ReservedReg(_reg) => Err(WasmError::internal(
                    "x86_64 Select condition cannot be reserved cache register",
                )),
            }
        } else {
            if let MachineValue::Imm64(value) = cond {
                let selected = if value != 0 { on_true } else { on_false };
                return self.lower_move(ty, dst, selected);
            }
            let dst = self.map_gp_reg(dst)?;
            // Materialize operands BEFORE testing the condition, because
            // materialize_value may clobber flags (e.g. xor reg,reg for zero).
            let scratch0 = self.gp_scratch.scoped_alloc().detach();
            let scratch1 = self.gp_scratch.scoped_alloc().detach();
            let true_reg = self.materialize_value(*scratch0, on_true)?;
            let false_reg = self.materialize_value(*scratch1, on_false)?;
            let cond_gp = match cond {
                MachineValue::Reg(reg) => self.map_gp_reg(reg)?,
                _ => unreachable!(),
            };
            // Wasm select conditions are i32 values; ignore any stale
            // upper half that may remain in a GpWord carrier.
            if !self.flags32_current(cond_gp) {
                enc::test_rr_32(&mut self.core.text, cond_gp, cond_gp);
            }
            if dst == true_reg && dst != false_reg {
                self.emit_gp_cmov_ty(ty, Cc::E, dst, false_reg)?;
            } else if dst == false_reg {
                self.emit_gp_cmov_ty(ty, Cc::NE, dst, true_reg)?;
            } else {
                self.emit_gp_move_ty(ty, dst, false_reg)?;
                self.emit_gp_cmov_ty(ty, Cc::NE, dst, true_reg)?;
            }
            Ok(())
        }
    }

    /// GP select consuming an already-live EFLAGS condition (fused
    /// IntCompare + Select). The caller emitted the compare; operand
    /// materialization here must not clobber flags, so immediates use
    /// `mov` forms only (`materialize_value_flags_safe`).
    pub(super) fn lower_fused_compare_select(
        &mut self,
        fusion: &super::fusion::IntCompareSelect,
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;

        // Pre-materialize one zero immediate while EFLAGS is still dead:
        // the xor zero idiom is shorter and dependency-breaking but
        // clobbers flags, so it must precede the compare. Only one
        // operand can be held across `lower_cmp_values`, which pins two
        // of the three pool scratches itself.
        let mut pre_guard = None;
        let mut pre_true: Option<X86Reg> = None;
        let mut pre_false: Option<X86Reg> = None;
        if on_true == MachineValue::Imm64(0) {
            let guard = self.gp_scratch.scoped_alloc().detach();
            enc::xor_rr_32(&mut self.core.text, *guard, *guard);
            pre_true = Some(*guard);
            pre_guard = Some(guard);
        } else if on_false == MachineValue::Imm64(0) {
            let guard = self.gp_scratch.scoped_alloc().detach();
            enc::xor_rr_32(&mut self.core.text, *guard, *guard);
            pre_false = Some(*guard);
            pre_guard = Some(guard);
        }

        self.lower_cmp_values(fusion.width, fusion.lhs, fusion.rhs)?;
        let cc = super::fusion::map_int_cond(fusion.kind, fusion.sign);

        // --- flags live from here to the CMOV ---
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let true_reg = match pre_true {
            Some(reg) => reg,
            None => self.materialize_value_flags_safe(*scratch0, on_true)?,
        };
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        let false_reg = match pre_false {
            Some(reg) => reg,
            None => self.materialize_value_flags_safe(*scratch1, on_false)?,
        };
        if dst == true_reg && dst != false_reg {
            self.emit_gp_cmov_ty(ty, cc.invert(), dst, false_reg)?;
        } else if dst == false_reg {
            self.emit_gp_cmov_ty(ty, cc, dst, true_reg)?;
        } else {
            self.emit_gp_move_ty(ty, dst, false_reg)?;
            self.emit_gp_cmov_ty(ty, cc, dst, true_reg)?;
        }
        drop(pre_guard);
        Ok(())
    }

    /// `materialize_value` variant that never touches EFLAGS: the zero
    /// immediate uses `mov r32, 0` instead of the xor zero idiom.
    fn materialize_value_flags_safe(
        &mut self,
        scratch: X86Reg,
        value: MachineValue,
    ) -> Result<X86Reg, WasmError> {
        if let MachineValue::Imm64(imm) = value {
            if imm <= u32::MAX as u64 {
                enc::mov_ri_32_no_flags(&mut self.core.text, scratch, imm as u32);
            } else {
                enc::mov_ri_64(&mut self.core.text, scratch, imm);
            }
            return Ok(scratch);
        }
        self.materialize_value(scratch, value)
    }

    // ── Convert (integer parts only — float conversions are Phase 4) ─────────

    pub(super) fn lower_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let src_gp = self.materialize_value(*gp0, src)?;
        match op {
            MachineConvertOp::I32WrapI64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zeroes the upper 32 bits
                enc::mov_rr_32(&mut self.core.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32S => {
                let dst_gp = self.map_gp_reg(dst)?;
                enc::movsxd_r64_r32(&mut self.core.text, dst_gp, src_gp);
            }
            MachineConvertOp::I64ExtendI32U => {
                let dst_gp = self.map_gp_reg(dst)?;
                // MOV r32, r32 zero-extends
                enc::mov_rr_32(&mut self.core.text, dst_gp, src_gp);
            }
            MachineConvertOp::I32ReinterpretF32 | MachineConvertOp::I64ReinterpretF64 => {
                let dst_gp = self.map_gp_reg(dst)?;
                if dst_gp != src_gp {
                    enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let width = dst_float_width.expect("float reinterpret width");
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp)
                        }
                    };
                    self.core.set_fp_reg_width(dst, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst)?;
                    if dst_gp != src_gp {
                        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
                    }
                }
            }
            // Float promotion / demotion
            MachineConvertOp::F64PromoteF32 => {
                let src_fp = self.prepare_float_operand(MachineFloatWidth::F32, src, *gp0, *fp0)?;
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                    enc::cvtss2sd(&mut self.core.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtss2sd(&mut self.core.text, *fp1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movq_r64_xmm(&mut self.core.text, dst_gp, *fp1 as u8);
                }
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp = self.prepare_float_operand(MachineFloatWidth::F64, src, *gp0, *fp0)?;
                if self.core.is_fp_reg(dst) {
                    let dst_fp = self.map_fp_reg(dst)? as u8;
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                    enc::cvtsd2ss(&mut self.core.text, dst_fp, src_fp as u8);
                } else {
                    enc::cvtsd2ss(&mut self.core.text, *fp1 as u8, src_fp as u8);
                    let dst_gp = self.map_gp_reg(dst)?;
                    enc::movd_r32_xmm(&mut self.core.text, dst_gp, *fp1 as u8);
                }
            }
            // Int -> Float conversions
            MachineConvertOp::F32ConvertI32S => {
                // CVTSI2SS xmm, r32
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::cvtsi2ss_r32(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI32U => {
                // Zero-extend to 64-bit first for unsigned interpretation
                enc::mov_rr_32(&mut self.core.text, *gp0, src_gp);
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, *gp0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64S => {
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F32ConvertI64U => {
                // x86_64 has no unsigned int-to-float instruction.
                // For values that fit in i64 (bit 63 = 0), use signed conversion.
                // For values with bit 63 set, shift right by 1, convert, then double.
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F32, *fp1 as u8)?;
                enc::test_rr_64(&mut self.core.text, src_gp, src_gp);
                let large = self.core.new_label();
                self.emit_jcc(Cc::S, large); // JS = sign flag set = bit 63 is 1
                                             // Small path: fits in i64
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, src_gp);
                let done = self.core.new_label();
                self.emit_jmp(done);
                // Large path: bit 63 set
                self.core.bind_label(large);
                enc::mov_rr_64(&mut self.core.text, *gp0, src_gp);
                enc::mov_rr_64(&mut self.core.text, *gp1, src_gp);
                enc::shr_imm_64(&mut self.core.text, *gp0, 1); // src >> 1
                enc::and_ri_32(&mut self.core.text, *gp1, 1); // src & 1 (preserve LSB)
                enc::or_rr_64(&mut self.core.text, *gp0, *gp1); // (src >> 1) | (src & 1)
                enc::cvtsi2ss_r64(&mut self.core.text, dst_fp, *gp0);
                enc::addss(&mut self.core.text, dst_fp, dst_fp); // double it
                self.core.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F32, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32S => {
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::cvtsi2sd_r32(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI32U => {
                enc::mov_rr_32(&mut self.core.text, *gp0, src_gp);
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, *gp0);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64S => {
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, src_gp);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            MachineConvertOp::F64ConvertI64U => {
                let dst_fp =
                    self.prepare_int_to_float_dst(dst, MachineFloatWidth::F64, *fp1 as u8)?;
                enc::test_rr_64(&mut self.core.text, src_gp, src_gp);
                let large = self.core.new_label();
                self.emit_jcc(Cc::S, large);
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, src_gp);
                let done = self.core.new_label();
                self.emit_jmp(done);
                self.core.bind_label(large);
                enc::mov_rr_64(&mut self.core.text, *gp0, src_gp);
                enc::mov_rr_64(&mut self.core.text, *gp1, src_gp);
                enc::shr_imm_64(&mut self.core.text, *gp0, 1);
                enc::and_ri_32(&mut self.core.text, *gp1, 1);
                enc::or_rr_64(&mut self.core.text, *gp0, *gp1);
                enc::cvtsi2sd_r64(&mut self.core.text, dst_fp, *gp0);
                enc::addsd(&mut self.core.text, dst_fp, dst_fp);
                self.core.bind_label(done);
                self.store_fp_result_if_gp(dst, MachineFloatWidth::F64, dst_fp)?;
            }
            // Trapping truncations: call Rust helper
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                if src_gp != *gp0 {
                    drop(gp0);
                }
                drop(gp1);
                drop(fp0);
                drop(fp1);
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_trapping_trunc(op, dst_gp, src_gp)?;
            }
            // Saturating truncations: call Rust helper
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => {
                if src_gp != *gp0 {
                    drop(gp0);
                }
                drop(gp1);
                drop(fp0);
                drop(fp1);
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_saturating_trunc(op, dst_gp, src_gp)?;
            }
        }
        Ok(())
    }

    // ── Float ops ─────────────────────────────────────────────────────────────

    pub(super) fn lower_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let gp_scratch = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let src_fp = self.prepare_float_operand(width, src, *gp_scratch, *fp0)?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            *fp1 as u8
        };
        match op {
            MachineFloatUnaryOp::Sqrt => {
                match width {
                    MachineFloatWidth::F32 => {
                        enc::sqrtss(&mut self.core.text, result_fp, src_fp as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::sqrtsd(&mut self.core.text, result_fp, src_fp as u8)
                    }
                };
            }
            MachineFloatUnaryOp::Ceil => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_CEIL,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_CEIL,
                    ),
                };
            }
            MachineFloatUnaryOp::Floor => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_FLOOR,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_FLOOR,
                    ),
                };
            }
            MachineFloatUnaryOp::Trunc => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_TRUNC,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_TRUNC,
                    ),
                };
            }
            MachineFloatUnaryOp::Nearest => {
                match width {
                    MachineFloatWidth::F32 => enc::roundss(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_NEAREST,
                    ),
                    MachineFloatWidth::F64 => enc::roundsd(
                        &mut self.core.text,
                        result_fp,
                        src_fp as u8,
                        enc::ROUND_NEAREST,
                    ),
                };
            }
            MachineFloatUnaryOp::Abs => {
                // Clear sign bit: AND with mask.
                let mask_xmm = if result_fp != *fp0 as u8 {
                    *fp0 as u8
                } else {
                    *fp1 as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                self.materialize_u64(*gp_scratch, mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp_scratch);
                enc::andpd(&mut self.core.text, result_fp, mask_xmm);
            }
            MachineFloatUnaryOp::Neg => {
                // Flip sign bit: XOR with mask.
                let mask_xmm = if result_fp != *fp0 as u8 {
                    *fp0 as u8
                } else {
                    *fp1 as u8
                };
                if result_fp != src_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, src_fp as u8)
                        }
                    };
                }
                let mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                self.materialize_u64(*gp_scratch, mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp_scratch);
                enc::xorpd(&mut self.core.text, result_fp, mask_xmm);
            }
        }
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, result_fp),
            };
        }
        Ok(())
    }

    pub(super) fn lower_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let lhs_fp = self.prepare_float_operand(width, lhs, *gp0, *fp0)?;
        let rhs_fp = self.prepare_float_operand(width, rhs, *gp1, *fp1)?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            // Keep GP-targeted float ops in XMM0. XMM1 is already the rhs
            // materialization scratch, so using it as the destination can
            // overwrite the rhs before the binary op runs.
            *fp0 as u8
        };
        match op {
            MachineFloatBinaryOp::Add
            | MachineFloatBinaryOp::Sub
            | MachineFloatBinaryOp::Mul
            | MachineFloatBinaryOp::Div => {
                // SSE two-operand: dst = dst OP src. Move lhs into result first.
                // If result == rhs and result != lhs, the move would clobber rhs.
                // Save rhs to scratch first in that case. Full-register MOVAPS
                // copies: the scalar move forms merge into the destination's
                // upper lanes, and that false dependency on the destination's
                // previous value serializes loop iterations.
                let actual_rhs = if result_fp == rhs_fp as u8
                    && matches!(op, MachineFloatBinaryOp::Add | MachineFloatBinaryOp::Mul)
                {
                    // Add and multiply can consume the rhs in place. This
                    // preserves rounding and signed zero; either operand's
                    // arithmetic NaN is permitted by Wasm.
                    lhs_fp as u8
                } else {
                    let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                        let scratch = *fp1 as u8;
                        enc::movaps_rr(&mut self.core.text, scratch, rhs_fp as u8);
                        scratch
                    } else {
                        rhs_fp as u8
                    };
                    if result_fp != lhs_fp as u8 {
                        enc::movaps_rr(&mut self.core.text, result_fp, lhs_fp as u8);
                    }
                    actual_rhs
                };
                match (width, op) {
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                        enc::addss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                        enc::addsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                        enc::subss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                        enc::subsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                        enc::mulss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                        enc::mulsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                        enc::divss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                        enc::divsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    _ => unreachable!(),
                };
            }
            MachineFloatBinaryOp::Min | MachineFloatBinaryOp::Max => {
                // Wasm fmin/fmax: if either operand is NaN, result is NaN.
                // x86_64 minsd/minss/maxsd/maxss: if either is NaN, returns
                // the SECOND (source) operand — and destroys the first
                // (destination) operand. We therefore (a) check for NaN
                // BEFORE the minsd/maxsd so the compare still sees the
                // original lhs register unclobbered, and (b) fall through
                // to an `addsd` on the NaN path so any NaN propagates.
                //
                // Previous version did the ucomisd AFTER the minsd, which
                // is wrong when `result_fp == lhs_fp`: the minsd clobbers
                // lhs_fp with the min result, so the NaN check compares
                // the WRONG operand and incorrectly takes the fast path.
                let is_min = matches!(op, MachineFloatBinaryOp::Min);

                // Guard: if result == rhs and result != lhs, save rhs to
                // a scratch so the later `movss result_fp, lhs_fp` (or the
                // inline minsd/addsd which both clobber dst) doesn't
                // destroy the rhs value before we use it.
                let actual_rhs = if result_fp == rhs_fp as u8 && result_fp != lhs_fp as u8 {
                    let scratch = *fp1 as u8;
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movaps_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movaps_rr(&mut self.core.text, scratch, rhs_fp as u8)
                        }
                    };
                    scratch
                } else {
                    rhs_fp as u8
                };
                // Move lhs into result_fp if needed. This must happen
                // BEFORE the ucomisd so the NaN check still sees the
                // original lhs_fp value (which we leave untouched in its
                // own register).
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                // NaN check on the ORIGINAL operands. If `result_fp ==
                // lhs_fp`, lhs_fp still holds the original lhs at this
                // point (we have not yet run the clobbering min/max).
                match width {
                    MachineFloatWidth::F32 => {
                        enc::ucomiss(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::ucomisd(&mut self.core.text, lhs_fp as u8, actual_rhs)
                    }
                };
                let nan_path = self.core.new_label();
                let equal_path = self.core.new_label();
                let done = self.core.new_label();
                // PF=1 means unordered (NaN) — jump to NaN path.
                self.emit_jcc(Cc::P, nan_path);
                // ZF=1 with PF=0 means equal, where minsd/maxsd return the
                // SECOND operand — wrong for a (-0.0, +0.0) pair, which
                // compares equal but must produce -0.0 for min and +0.0
                // for max. Bitwise OR (min) / AND (max) of the operands is
                // the identity when they are the same value and selects
                // the required zero sign when they differ only in sign.
                self.emit_jcc(Cc::E, equal_path);
                // Strictly-ordered fast path: minsd / maxsd are exact.
                match (width, is_min) {
                    (MachineFloatWidth::F32, true) => {
                        enc::minss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, true) => {
                        enc::minsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F32, false) => {
                        enc::maxss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    (MachineFloatWidth::F64, false) => {
                        enc::maxsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                };
                self.emit_jmp(done);
                // Equal path: result_fp already holds lhs; the full-width
                // register op is fine because only lane 0 is live for
                // scalars.
                self.core.bind_label(equal_path);
                if is_min {
                    enc::orpd(&mut self.core.text, result_fp, actual_rhs);
                } else {
                    enc::andpd(&mut self.core.text, result_fp, actual_rhs);
                }
                self.emit_jmp(done);
                // NaN path: result_fp already holds lhs, so addsd
                // propagates any NaN from either operand.
                self.core.bind_label(nan_path);
                match width {
                    MachineFloatWidth::F32 => {
                        enc::addss(&mut self.core.text, result_fp, actual_rhs)
                    }
                    MachineFloatWidth::F64 => {
                        enc::addsd(&mut self.core.text, result_fp, actual_rhs)
                    }
                };
                self.core.bind_label(done);
            }
            MachineFloatBinaryOp::Copysign => {
                // magnitude of lhs, sign of rhs.
                // Strategy: clear sign of lhs (abs), extract sign of rhs, OR them.
                // Use a mask scratch that doesn't conflict with result_fp or rhs_fp.
                let mask_xmm = if result_fp != *fp0 as u8 && rhs_fp as u8 != *fp0 as u8 {
                    *fp0 as u8
                } else {
                    *fp1 as u8
                };
                let sign_mask = match width {
                    MachineFloatWidth::F32 => 0x8000_0000u64,
                    MachineFloatWidth::F64 => 0x8000_0000_0000_0000u64,
                };
                let abs_mask = match width {
                    MachineFloatWidth::F32 => 0x7FFF_FFFFu64,
                    MachineFloatWidth::F64 => 0x7FFF_FFFF_FFFF_FFFFu64,
                };
                // result = lhs & abs_mask (clear sign bit)
                if result_fp != lhs_fp as u8 {
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movaps_rr(&mut self.core.text, result_fp, lhs_fp as u8)
                        }
                    };
                }
                self.materialize_u64(*gp0, abs_mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp0);
                enc::andpd(&mut self.core.text, result_fp, mask_xmm);
                // mask_xmm = rhs & sign_mask (extract sign bit)
                self.materialize_u64(*gp0, sign_mask);
                enc::movq_xmm_r64(&mut self.core.text, mask_xmm, *gp0);
                enc::andpd(&mut self.core.text, mask_xmm, rhs_fp as u8);
                // result |= mask_xmm
                enc::orpd(&mut self.core.text, result_fp, mask_xmm);
            }
        };
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, result_fp),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, result_fp),
            };
        }
        Ok(())
    }

    pub(super) fn lower_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let lhs_fp = self.prepare_float_operand(width, lhs, *gp0, *fp0)?;
        // Choose an rhs FP scratch that doesn't conflict with lhs. When lhs
        // already lives in a mapped FP register (not `fp0`), reuse `fp0` for
        // rhs to avoid clobbering a live FP SSA value.
        let rhs_fp_scratch = if lhs_fp != *fp0 as u32 { *fp0 } else { *fp1 };
        if matches!(rhs, MachineValue::Imm64(0)) {
            enc::xorpd(
                &mut self.core.text,
                rhs_fp_scratch as u8,
                rhs_fp_scratch as u8,
            );
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp_scratch as u8)
                }
            };
        } else {
            let rhs_fp = self.prepare_float_operand(width, rhs, *gp1, rhs_fp_scratch)?;
            match width {
                MachineFloatWidth::F32 => {
                    enc::ucomiss(&mut self.core.text, lhs_fp as u8, rhs_fp as u8)
                }
                MachineFloatWidth::F64 => {
                    enc::ucomisd(&mut self.core.text, lhs_fp as u8, rhs_fp as u8)
                }
            };
        }
        // Wasm float comparisons: unordered (NaN) => 0 for all except Ne.
        // UCOMISD sets: ZF=1,PF=1,CF=1 for unordered; ZF=1,PF=0,CF=0 for equal;
        // ZF=0,PF=0,CF=1 for less-than; ZF=0,PF=0,CF=0 for greater-than.
        match kind {
            MachineCompareKind::Eq => {
                // Ordered and equal: ZF=1 AND PF=0
                // SETE sets dst to ZF, SETNP sets tmp to !PF, AND them.
                enc::setcc(&mut self.core.text, Cc::E, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, *gp0);
                enc::and_rr_32(&mut self.core.text, dst_gp, *gp0);
                // Zero-extend the byte result to full register
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ne => {
                // Unordered OR not-equal: NE=1 OR PF=1
                enc::setcc(&mut self.core.text, Cc::NE, dst_gp);
                enc::setcc(&mut self.core.text, Cc::P, *gp0);
                enc::or_rr_32(&mut self.core.text, dst_gp, *gp0);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Lt => {
                // Ordered and less: CF=1 AND PF=0 → use JB (CF=1), but need !PF too
                // Actually, for UCOMISD: CF=1 for less-than AND for unordered.
                // So Lt = CF=1 AND PF=0 → SETB AND SETNP
                enc::setcc(&mut self.core.text, Cc::B, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, *gp0);
                enc::and_rr_32(&mut self.core.text, dst_gp, *gp0);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Gt => {
                // Ordered and greater: ZF=0, CF=0, PF=0 → JA (CF=0 AND ZF=0)
                // JA already excludes unordered (PF=1 implies CF=1), so SETA is correct.
                enc::setcc(&mut self.core.text, Cc::A, dst_gp);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Le => {
                // Ordered and less-or-equal: (CF=1 OR ZF=1) AND PF=0
                enc::setcc(&mut self.core.text, Cc::BE, dst_gp);
                enc::setcc(&mut self.core.text, Cc::NP, *gp0);
                enc::and_rr_32(&mut self.core.text, dst_gp, *gp0);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
            MachineCompareKind::Ge => {
                // Ordered and greater-or-equal: CF=0 AND PF=0 → JAE excludes unordered already
                // Actually JAE = !CF. Unordered sets CF=1, so JAE is 0 for unordered. Correct.
                enc::setcc(&mut self.core.text, Cc::AE, dst_gp);
                enc::movzx_r32_r8(&mut self.core.text, dst_gp, dst_gp);
            }
        }
        Ok(())
    }

    // ── Helper calls ──────────────────────────────────────────────────────────

    pub(super) fn lower_call_runtime(&mut self, const_idx: usize) -> Result<(), WasmError> {
        // Delegated to the control-flow module, which owns the matching
        // body_local_error_label propagation path.
        self.lower_call_runtime_term(const_idx)
    }

    // ── Float conversion helpers ────────────────────────────────────────────

    /// Integer conversion defines a scalar from scratch. When profitable for
    /// the CPU, clear its XMM destination to break the legacy CVTSI2SS/SD
    /// dependency on the previous value. No source lives in this FP register,
    /// and scalar carriers have no observable upper lanes. XORPS leaves
    /// integer EFLAGS intact.
    fn prepare_int_to_float_dst(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
        scratch_fp: u8,
    ) -> Result<u8, WasmError> {
        let dst_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)? as u8;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            scratch_fp
        };
        if super::cpu::clear_int_to_float_dst() {
            enc::xorps(&mut self.core.text, dst_fp, dst_fp);
        }
        Ok(dst_fp)
    }

    /// If dst is a GP register, move float result from XMM to GP.
    fn store_fp_result_if_gp(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
        fp_reg: u8,
    ) -> Result<(), WasmError> {
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            match width {
                MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, fp_reg),
                MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, fp_reg),
            };
        }
        Ok(())
    }

    /// Save caller-clobbered GP dynamic registers to the system stack before a
    /// C helper call.
    pub(super) fn save_caller_clobbered_gp_dynamic(&mut self) {
        // Push the caller-clobbered dynamic GP subset. `RAX`, `RCX`, and `RDX`
        // are backend-owned on x86_64, not dynamic.
        enc::push(&mut self.core.text, X86Reg::RSI);
        enc::push(&mut self.core.text, X86Reg::RDI);
        enc::push(&mut self.core.text, X86Reg::R8);
        enc::push(&mut self.core.text, X86Reg::R9);
        enc::push(&mut self.core.text, X86Reg::R10);
        enc::push(&mut self.core.text, X86Reg::R11);
    }

    /// Restore caller-clobbered GP dynamic registers from the system stack
    /// after a C helper call.
    pub(super) fn restore_caller_clobbered_gp_dynamic(&mut self) {
        enc::pop(&mut self.core.text, X86Reg::R11);
        enc::pop(&mut self.core.text, X86Reg::R10);
        enc::pop(&mut self.core.text, X86Reg::R9);
        enc::pop(&mut self.core.text, X86Reg::R8);
        enc::pop(&mut self.core.text, X86Reg::RDI);
        enc::pop(&mut self.core.text, X86Reg::RSI);
    }

    fn lower_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // The C trunc helper call is ABI-specific: SysV returns a two-field
        // `repr(C)` struct in RAX/RDX, while Win64 receives an out-pointer
        // as a fourth argument. `callconv::emit_trapping_trunc_call` owns
        // the whole save → arg-setup → call → restore → test → branch
        // sequence and leaves the 64-bit result in the supplied backend-owned
        // scratch register.
        #[cfg(sf_os_windows)]
        let result_scratch = X86Reg::RDX;
        #[cfg(not(sf_os_windows))]
        let result_scratch = *self
            .gp_scratch
            .try_claim_rcx()
            .or_else(|| self.gp_scratch.try_claim_rax())
            .expect("x86_64 trapping trunc needs RCX or RAX for the helper target")
            .detach();
        let error_label = self.core.body_local_error_label;
        callconv::emit_trapping_trunc_call(
            self,
            src,
            convert_op_code(op) as u64,
            result_scratch,
            error_label,
        );
        if dst != X86Reg::RDX {
            enc::mov_rr_64(&mut self.core.text, dst, X86Reg::RDX);
        }
        Ok(())
    }

    fn lower_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        // Keep the helper target out of ABI result registers. Win64 returns
        // the saturating result in RAX, so use backend-owned R11 for the
        // helper entry address.
        self.save_caller_clobbered_gp_dynamic();
        enc::mov_rr_64(&mut self.core.text, C_ARG0, src);
        self.materialize_u64(C_ARG1, convert_op_code(op) as u64);
        self.materialize_u64(
            X86Reg::R11,
            x86_64_saturating_trunc as *const () as usize as u64,
        );
        enc::call_reg(&mut self.core.text, X86Reg::R11);
        self.restore_caller_clobbered_gp_dynamic();
        if dst != X86Reg::RAX {
            enc::mov_rr_64(&mut self.core.text, dst, X86Reg::RAX);
        }
        Ok(())
    }

    /// Indexed load through one [base + index + disp] instruction. A
    /// ZeroExtend32 index is first zero-extended into a backend scratch —
    /// the MachineIR contract leaves its upper half undefined — while a
    /// full-width index feeds the SIB byte directly. Address arithmetic
    /// happens in the AGU, off the ALU dependency chain.
    pub(super) fn lower_indexed_load(
        &mut self,
        dst: MachineReg,
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base_x86 = self.map_gp_reg(base)?;
        let index_x86 = self.map_gp_reg(index)?;
        match index_extend {
            MachineIndexExtend::None => {
                self.lower_load_from_idx(dst, base_x86, index_x86, offset, width, extension)
            }
            MachineIndexExtend::ZeroExtend32 => {
                let scratch0 = self.gp_scratch.scoped_alloc().detach();
                let scratch1 = self.gp_scratch.scoped_alloc().detach();
                let idx = if base_x86 == *scratch0 {
                    *scratch1
                } else {
                    *scratch0
                };
                enc::mov_rr_32(&mut self.core.text, idx, index_x86);
                self.lower_load_from_idx(dst, base_x86, idx, offset, width, extension)
            }
        }
    }

    /// Indexed store; same addressing as [`Self::lower_indexed_load`].
    pub(super) fn lower_indexed_store(
        &mut self,
        base: MachineReg,
        index: MachineReg,
        index_extend: MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base_x86 = self.map_gp_reg(base)?;
        let index_x86 = self.map_gp_reg(index)?;
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        // Freed on drop at end of function, like scratch0/scratch1.
        let mut extra_scratch = None;
        let (idx, materialize_scratch) = match index_extend {
            MachineIndexExtend::None => {
                // The index register carries the address as-is; either
                // scratch may materialize the source, away from the base.
                let materialize = if base_x86 == *scratch0 {
                    *scratch1
                } else {
                    *scratch0
                };
                (index_x86, materialize)
            }
            MachineIndexExtend::ZeroExtend32 => {
                // One scratch holds the extended index; the other
                // materializes the source. Neither may be the base, and
                // the three-register pool minus at most one base overlap
                // always leaves two free.
                let (idx, materialize) = if base_x86 == *scratch0 {
                    let scratch2 = self.gp_scratch.scoped_alloc().detach();
                    let pair = (*scratch1, *scratch2);
                    extra_scratch = Some(scratch2);
                    pair
                } else if base_x86 == *scratch1 {
                    let scratch2 = self.gp_scratch.scoped_alloc().detach();
                    let pair = (*scratch0, *scratch2);
                    extra_scratch = Some(scratch2);
                    pair
                } else {
                    (*scratch0, *scratch1)
                };
                enc::mov_rr_32(&mut self.core.text, idx, index_x86);
                (idx, materialize)
            }
        };
        let result =
            self.lower_store_to_idx(base_x86, idx, offset, width, src, materialize_scratch);
        drop(extra_scratch);
        result
    }

    fn lower_memory_grow(
        &mut self,
        mem_idx: u32,
        dst: MachineReg,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, delta)?;
        self.emit_preserved_call_and_close(op::MEMORY_GROW, Some(dst_gp));
        Ok(())
    }

    fn lower_preserved_no_result(
        &mut self,
        op_code: u32,
        imm0: u32,
        imm1: u32,
        arg0: MachineValue,
        arg1: MachineValue,
        arg2: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::io as preserved_io;
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, imm0);
        self.emit_io_store_imm(preserved_io::IMM1, imm1);
        self.emit_io_store_value(preserved_io::ARG0, arg0)?;
        self.emit_io_store_value(preserved_io::ARG1, arg1)?;
        self.emit_io_store_value(preserved_io::ARG2, arg2)?;
        self.emit_preserved_call_and_close(op_code, None);
        Ok(())
    }

    fn lower_preserved_no_result_extended(
        &mut self,
        op_code: u32,
        imms: &[(usize, u32)],
        args: &[(usize, MachineValue)],
    ) -> Result<(), WasmError> {
        self.emit_preserved_io_open();
        for &(slot, imm) in imms {
            self.emit_io_store_imm(slot, imm);
        }
        for &(slot, value) in args {
            self.emit_io_store_value(slot, value)?;
        }
        self.emit_preserved_call_and_close(op_code, None);
        Ok(())
    }

    fn lower_preserved_result(
        &mut self,
        op_code: u32,
        imm0: u32,
        imm1: u32,
        arg0: MachineValue,
        arg1: MachineValue,
        arg2: MachineValue,
        ty: MachineStorageType,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        use preserved::io as preserved_io;

        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, imm0);
        self.emit_io_store_imm(preserved_io::IMM1, imm1);
        self.emit_io_store_value(preserved_io::ARG0, arg0)?;
        self.emit_io_store_value(preserved_io::ARG1, arg1)?;
        self.emit_io_store_value(preserved_io::ARG2, arg2)?;

        if let Some(width) = ty.float_width() {
            self.emit_preserved_call_and_close(op_code, Some(X86Reg::RCX));
            let dst_fp = self.map_fp_reg(dst)? as u8;
            match width {
                MachineFloatWidth::F32 => {
                    enc::movd_xmm_r32(&mut self.core.text, dst_fp, X86Reg::RCX)
                }
                MachineFloatWidth::F64 => {
                    enc::movq_xmm_r64(&mut self.core.text, dst_fp, X86Reg::RCX)
                }
            }
            self.core.set_fp_reg_width(dst, width)?;
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            self.emit_preserved_call_and_close(op_code, Some(dst_gp));
        }
        Ok(())
    }

    fn lower_preserved_result_extended(
        &mut self,
        op_code: u32,
        imms: &[(usize, u32)],
        args: &[(usize, MachineValue)],
        ty: MachineStorageType,
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.emit_preserved_io_open();
        for &(slot, imm) in imms {
            self.emit_io_store_imm(slot, imm);
        }
        for &(slot, value) in args {
            self.emit_io_store_value(slot, value)?;
        }
        if let Some(width) = ty.float_width() {
            self.emit_preserved_call_and_close(op_code, Some(X86Reg::RCX));
            let dst_fp = self.map_fp_reg(dst)? as u8;
            match width {
                MachineFloatWidth::F32 => {
                    enc::movd_xmm_r32(&mut self.core.text, dst_fp, X86Reg::RCX)
                }
                MachineFloatWidth::F64 => {
                    enc::movq_xmm_r64(&mut self.core.text, dst_fp, X86Reg::RCX)
                }
            }
            self.core.set_fp_reg_width(dst, width)?;
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            self.emit_preserved_call_and_close(op_code, Some(dst_gp));
        }
        Ok(())
    }

    fn lower_array_new_fixed(
        &mut self,
        type_idx: u32,
        elements: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.lower_payload_preserved_op(
            preserved::op::ARRAY_NEW_FIXED,
            type_idx,
            elements,
            dst,
            "x86_64 backend received pair-valued array.new_fixed",
        )
    }

    fn lower_struct_new(
        &mut self,
        type_idx: u32,
        fields: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.lower_payload_preserved_op(
            preserved::op::STRUCT_NEW,
            type_idx,
            fields,
            dst,
            "x86_64 backend received pair-valued struct.new",
        )
    }

    fn lower_payload_preserved_op(
        &mut self,
        op_code: u32,
        type_idx: u32,
        items: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
        pair_error: &'static str,
    ) -> Result<(), WasmError> {
        use preserved::io as preserved_io;
        let payload_bytes = ((items.len() as u32 * 8) + 15) & !15;
        self.emit_preserved_io_open_with_prefix(payload_bytes);
        for (index, (value_lo, value_hi)) in items.iter().enumerate() {
            if value_hi.is_some() {
                return Err(WasmError::internal(pair_error));
            }
            self.emit_io_store_value_at(0, index, *value_lo)?;
        }
        self.emit_io_store_imm_at(payload_bytes, preserved_io::IMM0, type_idx);
        self.emit_io_store_imm_at(payload_bytes, preserved_io::IMM1, items.len() as u32);
        if items.is_empty() {
            self.emit_io_store_imm_at(payload_bytes, preserved_io::ARG0, 0);
        } else {
            let scratch = self.gp_scratch.scoped_alloc().detach();
            enc::mov_rr_64(&mut self.core.text, *scratch, X86Reg::RSP);
            enc::store_64(
                &mut self.core.text,
                X86Reg::RSP,
                payload_bytes as i32 + preserved_io::ARG0 as i32 * 8,
                *scratch,
            );
        }
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_call_and_close_with_prefix(op_code, Some(dst_gp), payload_bytes);
        Ok(())
    }

    fn lower_memory_fill(
        &mut self,
        mem_idx: u32,
        dest: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, val)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::MEMORY_FILL, None);
        Ok(())
    }

    fn lower_memory_copy(
        &mut self,
        dst_mem: u32,
        src_mem: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_mem);
        self.emit_io_store_imm(preserved_io::IMM1, src_mem);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::MEMORY_COPY, None);
        Ok(())
    }

    fn lower_memory_init(
        &mut self,
        mem_idx: u32,
        data_idx: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_imm(preserved_io::IMM1, data_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::MEMORY_INIT, None);
        Ok(())
    }

    fn lower_data_drop(&mut self, data_idx: u32) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, data_idx);
        self.emit_preserved_call_and_close(op::DATA_DROP, None);
        Ok(())
    }

    fn lower_table_grow(
        &mut self,
        table_idx: u32,
        dst: MachineReg,
        init_val: MachineValue,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_value(preserved_io::ARG0, init_val)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, delta)?;
        self.emit_preserved_call_and_close(op::TABLE_GROW, Some(dst_gp));
        Ok(())
    }

    fn lower_table_fill(
        &mut self,
        table_idx: u32,
        start: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, start)?;
        self.emit_io_store_value(preserved_io::ARG1, val)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::TABLE_FILL, None);
        Ok(())
    }

    fn lower_table_copy(
        &mut self,
        dst_tbl: u32,
        src_tbl: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_tbl);
        self.emit_io_store_imm(preserved_io::IMM1, src_tbl);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::TABLE_COPY, None);
        Ok(())
    }

    fn lower_table_init(
        &mut self,
        table_idx: u32,
        elem_idx: u32,
        dest: MachineValue,
        src: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_imm(preserved_io::IMM1, elem_idx);
        self.emit_io_store_u32_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_u32_value(preserved_io::ARG1, src)?;
        self.emit_io_store_u32_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(op::TABLE_INIT, None);
        Ok(())
    }

    fn lower_elem_drop(&mut self, elem_idx: u32) -> Result<(), WasmError> {
        use preserved::{io as preserved_io, op};
        self.emit_preserved_io_open();
        self.emit_io_store_imm(preserved_io::IMM0, elem_idx);
        self.emit_preserved_call_and_close(op::ELEM_DROP, None);
        Ok(())
    }
}
