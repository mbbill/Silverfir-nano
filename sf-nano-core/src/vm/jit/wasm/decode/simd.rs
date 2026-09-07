//! SIMD decoding for targets with vector support.

use super::{decode_immediate_mismatch, DecodeContext, Immediate, PrimitiveOpKind, WasmError};
use crate::opcodes::OpcodeFD;

#[inline]
fn expect_v128_immediate(imm: &Immediate) -> Result<[u8; 16], WasmError> {
    match imm {
        Immediate::V128(value) => Ok(*value),
        _ => Err(decode_immediate_mismatch("decoder expected v128 immediate")),
    }
}

#[inline]
fn expect_lane_index_immediate(imm: &Immediate) -> Result<u8, WasmError> {
    match imm {
        Immediate::LaneIndex(lane) => Ok(*lane),
        _ => Err(decode_immediate_mismatch(
            "decoder expected SIMD lane immediate",
        )),
    }
}

#[inline]
fn expect_shuffle_mask_immediate(imm: &Immediate) -> Result<[u8; 16], WasmError> {
    match imm {
        Immediate::ShuffleMask(lanes) => Ok(*lanes),
        _ => Err(decode_immediate_mismatch(
            "decoder expected SIMD shuffle immediate",
        )),
    }
}

#[inline]
fn expect_memarg_lane_immediate(imm: &Immediate) -> Result<(u64, u32, u8), WasmError> {
    match imm {
        Immediate::MemArgLane {
            offset,
            memidx,
            lane,
            ..
        } => Ok((*offset, *memidx, *lane)),
        _ => Err(decode_immediate_mismatch(
            "decoder expected SIMD memarg-lane immediate",
        )),
    }
}

#[inline]
fn is_relaxed_simd_opcode(op: OpcodeFD) -> bool {
    use OpcodeFD::*;

    matches!(
        op,
        I8X16_RELAXED_SWIZZLE
            | I32X4_RELAXED_TRUNC_F32X4_S
            | I32X4_RELAXED_TRUNC_F32X4_U
            | I32X4_RELAXED_TRUNC_F64X2_S_ZERO
            | I32X4_RELAXED_TRUNC_F64X2_U_ZERO
            | I16X8_RELAXED_Q15MULR_S
            | I8X16_RELAXED_LANESELECT
            | I16X8_RELAXED_LANESELECT
            | I32X4_RELAXED_LANESELECT
            | I64X2_RELAXED_LANESELECT
            | F32X4_RELAXED_MIN
            | F32X4_RELAXED_MAX
            | F64X2_RELAXED_MIN
            | F64X2_RELAXED_MAX
            | I16X8_RELAXED_DOT_I8X16_I7X16_S
            | F32X4_RELAXED_MADD
            | F32X4_RELAXED_NMADD
            | F64X2_RELAXED_MADD
            | F64X2_RELAXED_NMADD
            | I32X4_RELAXED_DOT_I8X16_I7X16_ADD_S
    )
}

impl DecodeContext<'_> {
    pub(super) fn dispatch_simd(&mut self, op: OpcodeFD, imm: &Immediate) -> Result<(), WasmError> {
        match op {
            OpcodeFD::V128_CONST => self.handle_primitive(PrimitiveOpKind::V128Const {
                value: expect_v128_immediate(imm)?,
            }),
            OpcodeFD::V128_LOAD => {
                self.handle_load(imm, |offset, memidx| PrimitiveOpKind::V128Load {
                    offset,
                    memidx,
                });
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::V128_LOAD8X8_S
                        | OpcodeFD::V128_LOAD8X8_U
                        | OpcodeFD::V128_LOAD16X4_S
                        | OpcodeFD::V128_LOAD16X4_U
                        | OpcodeFD::V128_LOAD32X2_S
                        | OpcodeFD::V128_LOAD32X2_U
                        | OpcodeFD::V128_LOAD8_SPLAT
                        | OpcodeFD::V128_LOAD16_SPLAT
                        | OpcodeFD::V128_LOAD32_SPLAT
                        | OpcodeFD::V128_LOAD64_SPLAT
                        | OpcodeFD::V128_LOAD32_ZERO
                        | OpcodeFD::V128_LOAD64_ZERO
                ) =>
            {
                self.handle_load(imm, |offset, memidx| PrimitiveOpKind::SimdMemLoadV128 {
                    opcode: op_ext as u32,
                    offset,
                    memidx,
                });
            }
            OpcodeFD::V128_STORE => {
                self.handle_store(imm, |offset, memidx| PrimitiveOpKind::V128Store {
                    offset,
                    memidx,
                });
            }
            OpcodeFD::V128_NOT => self.handle_primitive(PrimitiveOpKind::V128Not),
            OpcodeFD::V128_AND => self.handle_primitive(PrimitiveOpKind::V128And),
            OpcodeFD::V128_ANDNOT => self.handle_primitive(PrimitiveOpKind::V128AndNot),
            OpcodeFD::V128_OR => self.handle_primitive(PrimitiveOpKind::V128Or),
            OpcodeFD::V128_XOR => self.handle_primitive(PrimitiveOpKind::V128Xor),
            OpcodeFD::V128_BITSELECT => self.handle_primitive(PrimitiveOpKind::V128Bitselect),
            OpcodeFD::V128_ANY_TRUE => self.handle_primitive(PrimitiveOpKind::V128AnyTrue),
            OpcodeFD::I8X16_ALL_TRUE => self.handle_primitive(PrimitiveOpKind::I8x16AllTrue),
            OpcodeFD::I16X8_ALL_TRUE => self.handle_primitive(PrimitiveOpKind::I16x8AllTrue),
            OpcodeFD::I32X4_ALL_TRUE => self.handle_primitive(PrimitiveOpKind::I32x4AllTrue),
            OpcodeFD::I64X2_ALL_TRUE => self.handle_primitive(PrimitiveOpKind::I64x2AllTrue),
            OpcodeFD::I8X16_BITMASK => self.handle_primitive(PrimitiveOpKind::I8x16Bitmask),
            OpcodeFD::I16X8_BITMASK => self.handle_primitive(PrimitiveOpKind::I16x8Bitmask),
            OpcodeFD::I32X4_BITMASK => self.handle_primitive(PrimitiveOpKind::I32x4Bitmask),
            OpcodeFD::I64X2_BITMASK => self.handle_primitive(PrimitiveOpKind::I64x2Bitmask),
            OpcodeFD::I8X16_SPLAT => self.handle_primitive(PrimitiveOpKind::I8x16Splat),
            OpcodeFD::I16X8_SPLAT => self.handle_primitive(PrimitiveOpKind::I16x8Splat),
            OpcodeFD::I32X4_SPLAT => self.handle_primitive(PrimitiveOpKind::I32x4Splat),
            OpcodeFD::I64X2_SPLAT => self.handle_primitive(PrimitiveOpKind::I64x2Splat),
            OpcodeFD::F32X4_SPLAT => self.handle_primitive(PrimitiveOpKind::F32x4Splat),
            OpcodeFD::F64X2_SPLAT => self.handle_primitive(PrimitiveOpKind::F64x2Splat),
            OpcodeFD::I8X16_EXTRACT_LANE_S => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I8x16ExtractLaneS { lane });
            }
            OpcodeFD::I8X16_EXTRACT_LANE_U => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I8x16ExtractLaneU { lane });
            }
            OpcodeFD::I16X8_EXTRACT_LANE_S => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I16x8ExtractLaneS { lane });
            }
            OpcodeFD::I16X8_EXTRACT_LANE_U => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I16x8ExtractLaneU { lane });
            }
            OpcodeFD::I32X4_EXTRACT_LANE => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I32x4ExtractLane { lane });
            }
            OpcodeFD::I64X2_EXTRACT_LANE => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I64x2ExtractLane { lane });
            }
            OpcodeFD::F32X4_EXTRACT_LANE => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::F32x4ExtractLane { lane });
            }
            OpcodeFD::F64X2_EXTRACT_LANE => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::F64x2ExtractLane { lane });
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_REPLACE_LANE
                        | OpcodeFD::I16X8_REPLACE_LANE
                        | OpcodeFD::I32X4_REPLACE_LANE
                        | OpcodeFD::I64X2_REPLACE_LANE
                        | OpcodeFD::F32X4_REPLACE_LANE
                        | OpcodeFD::F64X2_REPLACE_LANE
                ) =>
            {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::SimdReplaceLane {
                    opcode: op_ext as u32,
                    lane,
                });
            }
            OpcodeFD::I8X16_SHUFFLE => {
                let lanes = expect_shuffle_mask_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I8x16Shuffle { lanes });
            }
            OpcodeFD::I8X16_RELAXED_SWIZZLE => {
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 {
                    opcode: OpcodeFD::I8X16_SWIZZLE as u32,
                });
            }
            OpcodeFD::I32X4_RELAXED_TRUNC_F32X4_S => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F32X4_S as u32,
                });
            }
            OpcodeFD::I32X4_RELAXED_TRUNC_F32X4_U => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F32X4_U as u32,
                });
            }
            OpcodeFD::I32X4_RELAXED_TRUNC_F64X2_S_ZERO => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F64X2_S_ZERO as u32,
                });
            }
            OpcodeFD::I32X4_RELAXED_TRUNC_F64X2_U_ZERO => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F64X2_U_ZERO as u32,
                });
            }
            OpcodeFD::I16X8_RELAXED_Q15MULR_S => {
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 {
                    opcode: OpcodeFD::I16X8_Q15MULR_SAT_S as u32,
                });
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_RELAXED_LANESELECT
                        | OpcodeFD::I16X8_RELAXED_LANESELECT
                        | OpcodeFD::I32X4_RELAXED_LANESELECT
                        | OpcodeFD::I64X2_RELAXED_LANESELECT
                        | OpcodeFD::F32X4_RELAXED_MADD
                        | OpcodeFD::F32X4_RELAXED_NMADD
                        | OpcodeFD::F64X2_RELAXED_MADD
                        | OpcodeFD::F64X2_RELAXED_NMADD
                        | OpcodeFD::I32X4_RELAXED_DOT_I8X16_I7X16_ADD_S
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdTernaryV128 {
                    opcode: op_ext as u32,
                });
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::F32X4_RELAXED_MIN
                        | OpcodeFD::F32X4_RELAXED_MAX
                        | OpcodeFD::F64X2_RELAXED_MIN
                        | OpcodeFD::F64X2_RELAXED_MAX
                        | OpcodeFD::I16X8_RELAXED_DOT_I8X16_I7X16_S
                ) =>
            {
                let opcode = match op_ext {
                    OpcodeFD::F32X4_RELAXED_MIN => OpcodeFD::F32X4_MIN as u32,
                    OpcodeFD::F32X4_RELAXED_MAX => OpcodeFD::F32X4_MAX as u32,
                    OpcodeFD::F64X2_RELAXED_MIN => OpcodeFD::F64X2_MIN as u32,
                    OpcodeFD::F64X2_RELAXED_MAX => OpcodeFD::F64X2_MAX as u32,
                    _ => op_ext as u32,
                };
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 { opcode });
            }
            op_ext if is_relaxed_simd_opcode(op_ext) => {
                return Err(WasmError::invalid("unhandled relaxed SIMD opcode"));
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::V128_LOAD8_LANE
                        | OpcodeFD::V128_LOAD16_LANE
                        | OpcodeFD::V128_LOAD32_LANE
                        | OpcodeFD::V128_LOAD64_LANE
                ) =>
            {
                let (offset, memidx, lane) = expect_memarg_lane_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::SimdMemLoadLaneV128 {
                    opcode: op_ext as u32,
                    lane,
                    offset: u32::try_from(offset)
                        .map_err(|_| WasmError::invalid("simd memarg offset exceeds u32"))?,
                    memidx,
                });
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::V128_STORE8_LANE
                        | OpcodeFD::V128_STORE16_LANE
                        | OpcodeFD::V128_STORE32_LANE
                        | OpcodeFD::V128_STORE64_LANE
                ) =>
            {
                let (offset, memidx, lane) = expect_memarg_lane_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::SimdMemStoreLaneV128 {
                    opcode: op_ext as u32,
                    lane,
                    offset: u32::try_from(offset)
                        .map_err(|_| WasmError::invalid("simd memarg offset exceeds u32"))?,
                    memidx,
                });
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_SWIZZLE
                        | OpcodeFD::I8X16_NARROW_I16X8_S
                        | OpcodeFD::I8X16_NARROW_I16X8_U
                        | OpcodeFD::I8X16_MIN_S
                        | OpcodeFD::I8X16_MIN_U
                        | OpcodeFD::I8X16_MAX_S
                        | OpcodeFD::I8X16_MAX_U
                        | OpcodeFD::I8X16_AVGR_U
                        | OpcodeFD::I8X16_ADD
                        | OpcodeFD::I8X16_SUB
                        | OpcodeFD::I16X8_MIN_S
                        | OpcodeFD::I16X8_MIN_U
                        | OpcodeFD::I16X8_MAX_S
                        | OpcodeFD::I16X8_MAX_U
                        | OpcodeFD::I16X8_AVGR_U
                        | OpcodeFD::I16X8_Q15MULR_SAT_S
                        | OpcodeFD::I16X8_NARROW_I32X4_S
                        | OpcodeFD::I16X8_NARROW_I32X4_U
                        | OpcodeFD::I16X8_ADD
                        | OpcodeFD::I16X8_SUB
                        | OpcodeFD::I16X8_MUL
                        | OpcodeFD::I16X8_EXTMUL_LOW_I8X16_S
                        | OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_S
                        | OpcodeFD::I16X8_EXTMUL_LOW_I8X16_U
                        | OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_U
                        | OpcodeFD::I32X4_MIN_S
                        | OpcodeFD::I32X4_MIN_U
                        | OpcodeFD::I32X4_MAX_S
                        | OpcodeFD::I32X4_MAX_U
                        | OpcodeFD::I32X4_SUB
                        | OpcodeFD::I32X4_MUL
                        | OpcodeFD::I32X4_DOT_I16X8_S
                        | OpcodeFD::I32X4_EXTMUL_LOW_I16X8_S
                        | OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_S
                        | OpcodeFD::I32X4_EXTMUL_LOW_I16X8_U
                        | OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_U
                        | OpcodeFD::I64X2_SUB
                        | OpcodeFD::I64X2_MUL
                        | OpcodeFD::I64X2_EXTMUL_LOW_I32X4_S
                        | OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_S
                        | OpcodeFD::I64X2_EXTMUL_LOW_I32X4_U
                        | OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_U
                        | OpcodeFD::F64X2_ADD
                        | OpcodeFD::F64X2_SUB
                        | OpcodeFD::F64X2_MUL
                        | OpcodeFD::I8X16_ADD_SAT_S
                        | OpcodeFD::I8X16_ADD_SAT_U
                        | OpcodeFD::I16X8_ADD_SAT_S
                        | OpcodeFD::I16X8_ADD_SAT_U
                        | OpcodeFD::I8X16_SUB_SAT_S
                        | OpcodeFD::I8X16_SUB_SAT_U
                        | OpcodeFD::I16X8_SUB_SAT_S
                        | OpcodeFD::I16X8_SUB_SAT_U
                        | OpcodeFD::I8X16_EQ
                        | OpcodeFD::I8X16_NE
                        | OpcodeFD::I8X16_LT_S
                        | OpcodeFD::I8X16_LT_U
                        | OpcodeFD::I8X16_GT_S
                        | OpcodeFD::I8X16_GT_U
                        | OpcodeFD::I8X16_LE_S
                        | OpcodeFD::I8X16_LE_U
                        | OpcodeFD::I8X16_GE_S
                        | OpcodeFD::I8X16_GE_U
                        | OpcodeFD::I16X8_EQ
                        | OpcodeFD::I16X8_NE
                        | OpcodeFD::I16X8_LT_S
                        | OpcodeFD::I16X8_LT_U
                        | OpcodeFD::I16X8_GT_S
                        | OpcodeFD::I16X8_GT_U
                        | OpcodeFD::I16X8_LE_S
                        | OpcodeFD::I16X8_LE_U
                        | OpcodeFD::I16X8_GE_S
                        | OpcodeFD::I16X8_GE_U
                        | OpcodeFD::I32X4_EQ
                        | OpcodeFD::I32X4_NE
                        | OpcodeFD::I32X4_LT_S
                        | OpcodeFD::I32X4_LT_U
                        | OpcodeFD::I32X4_GT_S
                        | OpcodeFD::I32X4_GT_U
                        | OpcodeFD::I32X4_LE_S
                        | OpcodeFD::I32X4_LE_U
                        | OpcodeFD::I32X4_GE_S
                        | OpcodeFD::I32X4_GE_U
                        | OpcodeFD::F32X4_EQ
                        | OpcodeFD::F32X4_NE
                        | OpcodeFD::F32X4_LT
                        | OpcodeFD::F32X4_GT
                        | OpcodeFD::F32X4_LE
                        | OpcodeFD::F32X4_GE
                        | OpcodeFD::F32X4_ADD
                        | OpcodeFD::F32X4_SUB
                        | OpcodeFD::F32X4_MUL
                        | OpcodeFD::F32X4_MAX
                        | OpcodeFD::F32X4_PMIN
                        | OpcodeFD::F32X4_PMAX
                        | OpcodeFD::I64X2_EQ
                        | OpcodeFD::I64X2_NE
                        | OpcodeFD::I64X2_LT_S
                        | OpcodeFD::I64X2_GT_S
                        | OpcodeFD::I64X2_LE_S
                        | OpcodeFD::I64X2_GE_S
                        | OpcodeFD::F64X2_EQ
                        | OpcodeFD::F64X2_NE
                        | OpcodeFD::F64X2_LT
                        | OpcodeFD::F64X2_GT
                        | OpcodeFD::F64X2_LE
                        | OpcodeFD::F64X2_GE
                        | OpcodeFD::F32X4_MIN
                        | OpcodeFD::F64X2_PMIN
                        | OpcodeFD::F64X2_PMAX
                        | OpcodeFD::F64X2_MIN
                        | OpcodeFD::F64X2_MAX
                        | OpcodeFD::F32X4_DIV
                        | OpcodeFD::F64X2_DIV
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 {
                    opcode: op_ext as u32,
                })
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_SHL
                        | OpcodeFD::I8X16_SHR_S
                        | OpcodeFD::I8X16_SHR_U
                        | OpcodeFD::I16X8_SHL
                        | OpcodeFD::I16X8_SHR_S
                        | OpcodeFD::I16X8_SHR_U
                        | OpcodeFD::I32X4_SHL
                        | OpcodeFD::I32X4_SHR_S
                        | OpcodeFD::I32X4_SHR_U
                        | OpcodeFD::I64X2_SHL
                        | OpcodeFD::I64X2_SHR_S
                        | OpcodeFD::I64X2_SHR_U
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdShiftV128 {
                    opcode: op_ext as u32,
                })
            }
            op_ext
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_ABS
                        | OpcodeFD::I8X16_NEG
                        | OpcodeFD::I8X16_POPCNT
                        | OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_S
                        | OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_U
                        | OpcodeFD::I16X8_ABS
                        | OpcodeFD::I16X8_NEG
                        | OpcodeFD::I16X8_EXTEND_LOW_I8X16_S
                        | OpcodeFD::I16X8_EXTEND_HIGH_I8X16_S
                        | OpcodeFD::I16X8_EXTEND_LOW_I8X16_U
                        | OpcodeFD::I16X8_EXTEND_HIGH_I8X16_U
                        | OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_S
                        | OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_U
                        | OpcodeFD::I32X4_ABS
                        | OpcodeFD::I32X4_NEG
                        | OpcodeFD::I32X4_EXTEND_LOW_I16X8_S
                        | OpcodeFD::I32X4_EXTEND_HIGH_I16X8_S
                        | OpcodeFD::I32X4_EXTEND_LOW_I16X8_U
                        | OpcodeFD::I32X4_EXTEND_HIGH_I16X8_U
                        | OpcodeFD::I64X2_ABS
                        | OpcodeFD::I64X2_NEG
                        | OpcodeFD::I64X2_EXTEND_LOW_I32X4_S
                        | OpcodeFD::I64X2_EXTEND_HIGH_I32X4_S
                        | OpcodeFD::I64X2_EXTEND_LOW_I32X4_U
                        | OpcodeFD::I64X2_EXTEND_HIGH_I32X4_U
                        | OpcodeFD::F32X4_NEG
                        | OpcodeFD::F32X4_DEMOTE_F64X2_ZERO
                        | OpcodeFD::F32X4_SQRT
                        | OpcodeFD::F32X4_CEIL
                        | OpcodeFD::F32X4_FLOOR
                        | OpcodeFD::F32X4_TRUNC
                        | OpcodeFD::F32X4_NEAREST
                        | OpcodeFD::F64X2_ABS
                        | OpcodeFD::F64X2_NEG
                        | OpcodeFD::F64X2_SQRT
                        | OpcodeFD::F64X2_CEIL
                        | OpcodeFD::F64X2_FLOOR
                        | OpcodeFD::F64X2_TRUNC
                        | OpcodeFD::F64X2_NEAREST
                        | OpcodeFD::F32X4_ABS
                        | OpcodeFD::F32X4_CONVERT_I32X4_S
                        | OpcodeFD::F32X4_CONVERT_I32X4_U
                        | OpcodeFD::F64X2_PROMOTE_LOW_F32X4
                        | OpcodeFD::F64X2_CONVERT_LOW_I32X4_S
                        | OpcodeFD::F64X2_CONVERT_LOW_I32X4_U
                        | OpcodeFD::I32X4_TRUNC_SAT_F32X4_S
                        | OpcodeFD::I32X4_TRUNC_SAT_F32X4_U
                        | OpcodeFD::I32X4_TRUNC_SAT_F64X2_S_ZERO
                        | OpcodeFD::I32X4_TRUNC_SAT_F64X2_U_ZERO
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: op_ext as u32,
                })
            }
            OpcodeFD::I32X4_ADD => self.handle_primitive(PrimitiveOpKind::I32x4Add),
            OpcodeFD::I64X2_ADD => self.handle_primitive(PrimitiveOpKind::I64x2Add),

            _ => return Err(WasmError::invalid("unsupported semantic decode opcode")),
        }
        Ok(())
    }
}
