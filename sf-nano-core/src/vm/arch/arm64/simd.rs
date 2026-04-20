use super::{
    abi,
    backend::Arm64Backend,
    enc,
    reg::{Arm64FpReg, Arm64Reg},
};
use crate::{
    error::WasmError,
    opcodes::OpcodeFD,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineInst, MachineInstKind, MachineReg, MachineStorageType,
            MachineValue, MACHINE_CTX_REG,
        },
        runtime::preserved::{io as preserved_io, op as preserved_op, preserved_entry},
    },
};

const V128_PREFIX_BYTES: u32 = 16;

fn unsupported_simd_opcode(kind: &str, opcode: u32) -> WasmError {
    debug_assert!(
        false,
        "arm64 SIMD native codegen does not support {kind} opcode 0x{opcode:x}"
    );
    WasmError::internal("arm64 SIMD native codegen does not support this opcode")
}

fn expect_v128_reg(
    backend: &Arm64Backend<'_>,
    value: MachineValue,
    label: &'static str,
) -> Result<Arm64FpReg, WasmError> {
    match value {
        MachineValue::Reg(reg) => backend.map_fp_reg(reg),
        MachineValue::Imm64(_) | MachineValue::ReservedReg(_) => Err(WasmError::internal(label)),
    }
}

fn expect_scalar_gp_reg(
    backend: &Arm64Backend<'_>,
    value: MachineValue,
    label: &'static str,
) -> Result<Arm64Reg, WasmError> {
    match value {
        MachineValue::Reg(reg) => backend.map_gp_reg(reg),
        MachineValue::Imm64(_) | MachineValue::ReservedReg(_) => Err(WasmError::internal(label)),
    }
}

fn expect_scalar_fp_reg(
    backend: &Arm64Backend<'_>,
    value: MachineValue,
    label: &'static str,
) -> Result<Arm64FpReg, WasmError> {
    match value {
        MachineValue::Reg(reg) => backend.map_fp_reg(reg),
        MachineValue::Imm64(_) | MachineValue::ReservedReg(_) => Err(WasmError::internal(label)),
    }
}

fn encode_q_imm(addr: MachineAddr) -> Option<u32> {
    if addr.offset < 0 {
        return None;
    }
    let offset = addr.offset as u32;
    if offset > 65520 || offset & 0xf != 0 {
        return None;
    }
    Some(offset / 16)
}

fn encode_scaled_imm(offset: i32, unit_bytes: u32) -> Option<u32> {
    if offset < 0 {
        return None;
    }
    let offset = offset as u32;
    if unit_bytes == 0 || offset % unit_bytes != 0 {
        return None;
    }
    let imm12 = offset / unit_bytes;
    if imm12 > 4095 {
        return None;
    }
    Some(imm12)
}

impl<'a> Arm64Backend<'a> {
    pub(super) fn lower_simd_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move {
                ty: MachineStorageType::V128,
                dst,
                src,
                ..
            } => self.lower_move_v128(*dst, *src),
            MachineInstKind::V128Const { dst, bytes } => self.lower_v128_const_native(*dst, *bytes),
            MachineInstKind::V128FromRaw { dst, raw, .. } => self.lower_v128_from_raw(*dst, *raw),
            MachineInstKind::V128ToRaw { dst, src } => self.lower_v128_to_raw(*dst, *src),
            MachineInstKind::SimdUnary {
                opcode,
                dst_ty,
                dst,
                src,
            } => self.lower_simd_unary_native(*opcode, *dst_ty, *dst, *src),
            MachineInstKind::SimdBinary {
                opcode,
                dst,
                lhs,
                rhs,
                ..
            } => self.lower_simd_binary_native(*opcode, *dst, *lhs, *rhs),
            MachineInstKind::SimdShift {
                opcode,
                dst,
                vector,
                shift,
            } => self.lower_simd_shift_native(*opcode, *dst, *vector, *shift),
            MachineInstKind::SimdExtractLane {
                opcode,
                lane,
                dst_ty,
                dst,
                src,
            } => self.lower_simd_extract_lane_native(*opcode, *lane, *dst_ty, *dst, *src),
            MachineInstKind::SimdReplaceLane {
                opcode,
                lane,
                dst,
                vector,
                scalar,
            } => self.lower_simd_replace_lane_native(*opcode, *lane, *dst, *vector, *scalar),
            MachineInstKind::SimdShuffle {
                dst,
                lhs,
                rhs,
                lanes,
            } => self.lower_simd_shuffle_native(*dst, *lhs, *rhs, *lanes),
            MachineInstKind::SimdTernary {
                opcode,
                dst,
                a,
                b,
                c,
                ..
            } => self.lower_simd_ternary_native(*opcode, *dst, *a, *b, *c),
            MachineInstKind::SimdLoad { opcode, dst, addr } => {
                self.lower_simd_load_native(*opcode, *dst, *addr)
            }
            MachineInstKind::SimdLoadLane {
                opcode,
                lane,
                dst,
                addr,
                vector,
            } => self.lower_simd_load_lane_native(*opcode, *lane, *dst, *addr, *vector),
            MachineInstKind::SimdStore { opcode, addr, src } => {
                self.lower_simd_store_native(*opcode, *addr, *src)
            }
            MachineInstKind::SimdStoreLane {
                opcode,
                lane,
                addr,
                vector,
            } => self.lower_simd_store_lane_native(*opcode, *lane, *addr, *vector),
            _ => Err(WasmError::internal(
                "arm64 SIMD dispatch received a non-SIMD instruction",
            )),
        }
    }

    pub(super) fn lower_select_v128(
        &mut self,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let true_fp =
            expect_v128_reg(self, on_true, "arm64 v128 select needs a vector true input")?;
        let false_fp = expect_v128_reg(
            self,
            on_false,
            "arm64 v128 select needs a vector false input",
        )?;
        match cond {
            MachineValue::Imm64(value) => {
                let selected = if value != 0 { true_fp } else { false_fp };
                if dst_fp != selected {
                    self.core
                        .text
                        .emit_u32(enc::orr_16b(dst_fp, selected, selected));
                }
                Ok(())
            }
            MachineValue::Reg(reg) => {
                let cond_gp = self.map_gp_reg(reg)?;
                let false_label = self.core.new_label();
                let done_label = self.core.new_label();
                self.lower_cbz(cond_gp, false_label);
                if dst_fp != true_fp {
                    self.core
                        .text
                        .emit_u32(enc::orr_16b(dst_fp, true_fp, true_fp));
                }
                self.lower_b(done_label);
                self.core.bind_label(false_label);
                if dst_fp != false_fp {
                    self.core
                        .text
                        .emit_u32(enc::orr_16b(dst_fp, false_fp, false_fp));
                }
                self.core.bind_label(done_label);
                Ok(())
            }
            MachineValue::ReservedReg(_) => Err(WasmError::internal(
                "arm64 v128 select cannot consume reserved cache register as a condition",
            )),
        }
    }

    fn lower_move_v128(&mut self, dst: MachineReg, src: MachineValue) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let src_fp = expect_v128_reg(self, src, "arm64 v128 move needs a vector source register")?;
        if dst_fp != src_fp {
            self.core
                .text
                .emit_u32(enc::orr_16b(dst_fp, src_fp, src_fp));
        }
        Ok(())
    }

    fn lower_v128_const_native(
        &mut self,
        dst: MachineReg,
        bytes: [u8; 16],
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        self.materialize_v128_constant_into_fp(dst_fp, bytes);
        Ok(())
    }

    fn materialize_v128_constant_into_fp(&mut self, dst_fp: Arm64FpReg, bytes: [u8; 16]) {
        let low = u64::from_le_bytes(bytes[0..8].try_into().expect("slice length"));
        let high = u64::from_le_bytes(bytes[8..16].try_into().expect("slice length"));
        let gp0_idx = self.gp_scratch.alloc();
        let gp1_idx = self.gp_scratch.alloc();
        let gp0 = self.gp_scratch.reg(gp0_idx);
        let gp1 = self.gp_scratch.reg(gp1_idx);
        self.materialize_u64(gp0, low);
        self.materialize_u64(gp1, high);
        self.emit_adjust_stack_down(16);
        self.core
            .text
            .emit_u32(enc::stp_64(gp0, gp1, abi::stack_reg(), 0));
        self.core
            .text
            .emit_u32(enc::ldr_q(dst_fp, abi::stack_reg(), 0));
        self.emit_adjust_stack_up(16);
        self.gp_scratch.free_index(gp1_idx);
        self.gp_scratch.free_index(gp0_idx);
    }

    fn emit_preserved_call_for_v128_from_raw(&mut self, dst_fp: Arm64FpReg) {
        let call_scratch_idx = self.gp_scratch.alloc();
        let call_scratch = self.gp_scratch.reg(call_scratch_idx);

        // This is a storage-ABI bridge, not SIMD arithmetic lowering: live
        // v128 values stay in FP regs, while frame/global slots still carry a
        // 64-bit raw handle into the shared SIMD registry.
        self.core.text.emit_u32(enc::mov_reg_64(
            abi::C_ARG0,
            abi::map_fixed_reg(MACHINE_CTX_REG),
        ));
        super::inst::materialize_u64_into(
            &mut self.core.text,
            abi::C_ARG1,
            preserved_op::V128_FROM_RAW as u64,
        );
        self.core.text.emit_u32(enc::add_imm_64(
            abi::C_ARG2,
            abi::stack_reg(),
            V128_PREFIX_BYTES,
        ));
        super::inst::materialize_u64_into(
            &mut self.core.text,
            call_scratch,
            preserved_entry as usize as u64,
        );
        self.core.text.emit_u32(enc::blr(call_scratch));

        let error_path = self.core.new_label();
        self.lower_cbnz(abi::C_RET0, error_path);
        self.emit_restore_preserved_fp(V128_PREFIX_BYTES);
        self.emit_restore_preserved_gp(V128_PREFIX_BYTES);
        self.core
            .text
            .emit_u32(enc::ldr_q(dst_fp, abi::stack_reg(), 0));
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE + V128_PREFIX_BYTES);
        let done = self.core.new_label();
        self.lower_b(done);

        self.core.bind_label(error_path);
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE + V128_PREFIX_BYTES);
        let body_local_error_label = self.core.body_local_error_label;
        self.lower_b(body_local_error_label);

        self.core.bind_label(done);
        self.gp_scratch.free_index(call_scratch_idx);
    }

    fn lower_v128_from_raw(&mut self, dst: MachineReg, raw: MachineValue) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        self.emit_preserved_frame_open_with_prefix(V128_PREFIX_BYTES);
        self.emit_io_store_value_at((V128_PREFIX_BYTES / 8) as usize, preserved_io::ARG0, raw)?;
        self.emit_preserved_call_for_v128_from_raw(dst_fp);
        Ok(())
    }

    fn lower_v128_to_raw(&mut self, dst: MachineReg, src: MachineValue) -> Result<(), WasmError> {
        let src_fp = expect_v128_reg(
            self,
            src,
            "arm64 v128->raw conversion needs a vector source",
        )?;
        let dst_gp = self.map_gp_reg(dst)?;
        self.emit_preserved_frame_open_with_prefix(V128_PREFIX_BYTES);
        self.core
            .text
            .emit_u32(enc::str_q(src_fp, abi::stack_reg(), 0));

        let raw_idx = self.gp_scratch.alloc();
        let raw_tmp = self.gp_scratch.reg(raw_idx);
        self.emit_preserved_call_and_close_with_prefix(
            preserved_op::V128_TO_RAW,
            Some(raw_idx),
            V128_PREFIX_BYTES,
        );
        if dst_gp != raw_tmp {
            self.core.text.emit_u32(enc::mov_reg_64(dst_gp, raw_tmp));
        }
        self.gp_scratch.free_index(raw_idx);
        Ok(())
    }

    fn lower_simd_unary_native(
        &mut self,
        opcode: u32,
        dst_ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_NOT => {
                let src_fp = expect_v128_reg(self, src, "arm64 v128.not needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::V128_ANY_TRUE => {
                if !matches!(
                    dst_ty,
                    MachineStorageType::GpWord | MachineStorageType::GpI64
                ) {
                    return Err(WasmError::internal(
                        "arm64 v128.any_true expected a GP destination",
                    ));
                }
                let src_fp =
                    expect_v128_reg(self, src, "arm64 v128.any_true needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                let gp0 = self.gp_scratch.scoped_alloc().detach();
                let gp1 = self.gp_scratch.scoped_alloc().detach();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*gp0, src_fp, 0, 8));
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*gp1, src_fp, 1, 8));
                self.core.text.emit_u32(enc::orr_reg_64(*gp0, *gp0, *gp1));
                self.core.text.emit_u32(enc::cmp_zero_64(*gp0));
                self.core.text.emit_u32(enc::cset_64(dst_gp, enc::Cond::Ne));
                Ok(())
            }
            OpcodeFD::I8X16_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i8x16.all_true needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                let tmp = *self.fp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::cmeq_zero_16b(tmp, src_fp));
                self.core.text.emit_u32(enc::umaxv_16b(tmp, tmp));
                let scalar = *self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(scalar, tmp, 0, 1));
                self.core.text.emit_u32(enc::cmp_zero_32(scalar));
                self.core.text.emit_u32(enc::cset_32(dst_gp, enc::Cond::Eq));
                Ok(())
            }
            OpcodeFD::I16X8_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i16x8.all_true needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                let tmp = *self.fp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::cmeq_zero_8h(tmp, src_fp));
                self.core.text.emit_u32(enc::umaxv_8h(tmp, tmp));
                let scalar = *self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(scalar, tmp, 0, 2));
                self.core.text.emit_u32(enc::cmp_zero_32(scalar));
                self.core.text.emit_u32(enc::cset_32(dst_gp, enc::Cond::Eq));
                Ok(())
            }
            OpcodeFD::I32X4_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i32x4.all_true needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                let tmp = *self.fp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::cmeq_zero_4s(tmp, src_fp));
                self.core.text.emit_u32(enc::umaxv_4s(tmp, tmp));
                let scalar = *self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(scalar, tmp, 0, 4));
                self.core.text.emit_u32(enc::cmp_zero_32(scalar));
                self.core.text.emit_u32(enc::cset_32(dst_gp, enc::Cond::Eq));
                Ok(())
            }
            OpcodeFD::I64X2_ALL_TRUE => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i64x2.all_true needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                let lo = self.gp_scratch.scoped_alloc().detach();
                let hi = self.gp_scratch.scoped_alloc().detach();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*lo, src_fp, 0, 8));
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*hi, src_fp, 1, 8));
                let zero = self.core.new_label();
                let done = self.core.new_label();
                self.lower_cbz(*lo, zero);
                self.lower_cbz(*hi, zero);
                self.materialize_u64(dst_gp, 1);
                self.lower_b(done);
                self.core.bind_label(zero);
                self.materialize_u64(dst_gp, 0);
                self.core.bind_label(done);
                Ok(())
            }
            OpcodeFD::I8X16_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i8x16.bitmask needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_bitmask_from_v128(src_fp, dst_gp, 8)
            }
            OpcodeFD::I16X8_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i16x8.bitmask needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_bitmask_from_v128(src_fp, dst_gp, 16)
            }
            OpcodeFD::I32X4_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i32x4.bitmask needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_bitmask_from_v128(src_fp, dst_gp, 32)
            }
            OpcodeFD::I64X2_BITMASK => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i64x2.bitmask needs a vector source")?;
                let dst_gp = self.map_gp_reg(dst)?;
                self.lower_bitmask_from_v128(src_fp, dst_gp, 64)
            }
            OpcodeFD::I8X16_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)?;
                let src_gp = match src {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(self, src, "arm64 i8x16.splat needs a GP source")?,
                };
                self.core
                    .text
                    .emit_u32(enc::dup_16b_from_gp(dst_fp, src_gp));
                Ok(())
            }
            OpcodeFD::I16X8_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)?;
                let src_gp = match src {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(self, src, "arm64 i16x8.splat needs a GP source")?,
                };
                self.core.text.emit_u32(enc::dup_8h_from_gp(dst_fp, src_gp));
                Ok(())
            }
            OpcodeFD::I32X4_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)?;
                let src_gp = match src {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(self, src, "arm64 i32x4.splat needs a GP source")?,
                };
                self.core.text.emit_u32(enc::dup_4s_from_gp(dst_fp, src_gp));
                Ok(())
            }
            OpcodeFD::I64X2_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)?;
                let src_gp = match src {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(self, src, "arm64 i64x2.splat needs a GP source")?,
                };
                self.core.text.emit_u32(enc::dup_2d_from_gp(dst_fp, src_gp));
                Ok(())
            }
            OpcodeFD::F32X4_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)?;
                let src_fp = match src {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        let fp = *self.fp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        self.core.text.emit_u32(enc::fmov_s_from_gp(fp, gp));
                        fp
                    }
                    _ => expect_scalar_fp_reg(self, src, "arm64 f32x4.splat needs an FP source")?,
                };
                self.core
                    .text
                    .emit_u32(enc::dup_4s_from_lane0(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_SPLAT => {
                let dst_fp = self.map_fp_reg(dst)?;
                let src_fp = match src {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        let fp = *self.fp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        self.core.text.emit_u32(enc::fmov_d_from_gp(fp, gp));
                        fp
                    }
                    _ => expect_scalar_fp_reg(self, src, "arm64 f64x2.splat needs an FP source")?,
                };
                self.core
                    .text
                    .emit_u32(enc::dup_2d_from_lane0(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I8X16_POPCNT => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 i8x16.popcnt needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::cnt_16b(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i16x8.extadd_pairwise_i8x16_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::saddlp_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i16x8.extadd_pairwise_i8x16_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::uaddlp_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.extadd_pairwise_i16x8_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::saddlp_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.extadd_pairwise_i16x8_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::uaddlp_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTEND_LOW_I8X16_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i16x8.extend_low_i8x16_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::sshll_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTEND_HIGH_I8X16_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i16x8.extend_high_i8x16_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::sshll2_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTEND_LOW_I8X16_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i16x8.extend_low_i8x16_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ushll_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTEND_HIGH_I8X16_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i16x8.extend_high_i8x16_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ushll2_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTEND_LOW_I16X8_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.extend_low_i16x8_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::sshll_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTEND_HIGH_I16X8_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.extend_high_i16x8_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::sshll2_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTEND_LOW_I16X8_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.extend_low_i16x8_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ushll_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTEND_HIGH_I16X8_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.extend_high_i16x8_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ushll2_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTEND_LOW_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i64x2.extend_low_i32x4_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::sshll_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTEND_HIGH_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i64x2.extend_high_i32x4_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::sshll2_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTEND_LOW_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i64x2.extend_low_i32x4_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ushll_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTEND_HIGH_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i64x2.extend_high_i32x4_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ushll2_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_CONVERT_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 f32x4.convert_i32x4_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::scvtf_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_CONVERT_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 f32x4.convert_i32x4_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::ucvtf_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_CONVERT_LOW_I32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 f64x2.convert_low_i32x4_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::sshll_2d(*tmp, src_fp));
                self.core.text.emit_u32(enc::scvtf_2d(dst_fp, *tmp));
                Ok(())
            }
            OpcodeFD::F64X2_CONVERT_LOW_I32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 f64x2.convert_low_i32x4_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ushll_2d(*tmp, src_fp));
                self.core.text.emit_u32(enc::ucvtf_2d(dst_fp, *tmp));
                Ok(())
            }
            OpcodeFD::F32X4_DEMOTE_F64X2_ZERO => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 f32x4.demote_f64x2_zero needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                let src_tmp = if dst_fp == src_fp {
                    let tmp = self.fp_scratch.scoped_alloc().detach();
                    self.core.text.emit_u32(enc::orr_16b(*tmp, src_fp, src_fp));
                    Some(tmp)
                } else {
                    None
                };
                let src_fp = src_tmp.as_ref().map(|tmp| **tmp).unwrap_or(src_fp);
                self.core
                    .text
                    .emit_u32(enc::eor_16b(dst_fp, dst_fp, dst_fp));
                self.core.text.emit_u32(enc::fcvtn_2s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_PROMOTE_LOW_F32X4 => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 f64x2.promote_low_f32x4 needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtl_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_TRUNC_SAT_F32X4_S => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.trunc_sat_f32x4_s needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzs_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_TRUNC_SAT_F32X4_U => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.trunc_sat_f32x4_u needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fcvtzu_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_TRUNC_SAT_F64X2_S_ZERO => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.trunc_sat_f64x2_s_zero needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::fcvtzs_2d(*tmp, src_fp));
                self.core
                    .text
                    .emit_u32(enc::eor_16b(dst_fp, dst_fp, dst_fp));
                self.core.text.emit_u32(enc::sqxtn_2s(dst_fp, *tmp));
                Ok(())
            }
            OpcodeFD::I32X4_TRUNC_SAT_F64X2_U_ZERO => {
                let src_fp = expect_v128_reg(
                    self,
                    src,
                    "arm64 i32x4.trunc_sat_f64x2_u_zero needs a vector source",
                )?;
                let dst_fp = self.map_fp_reg(dst)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::fcvtzu_2d(*tmp, src_fp));
                self.core
                    .text
                    .emit_u32(enc::eor_16b(dst_fp, dst_fp, dst_fp));
                self.core.text.emit_u32(enc::uqxtn_2s(dst_fp, *tmp));
                Ok(())
            }
            OpcodeFD::I8X16_ABS => {
                let src_fp = expect_v128_reg(self, src, "arm64 i8x16.abs needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::abs_16b(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_ABS => {
                let src_fp = expect_v128_reg(self, src, "arm64 i16x8.abs needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::abs_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_ABS => {
                let src_fp = expect_v128_reg(self, src, "arm64 i32x4.abs needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::abs_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I64X2_ABS => {
                let src_fp = expect_v128_reg(self, src, "arm64 i64x2.abs needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::abs_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I8X16_NEG => {
                let src_fp = expect_v128_reg(self, src, "arm64 i8x16.neg needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::neg_16b(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I16X8_NEG => {
                let src_fp = expect_v128_reg(self, src, "arm64 i16x8.neg needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::neg_8h(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I32X4_NEG => {
                let src_fp = expect_v128_reg(self, src, "arm64 i32x4.neg needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::neg_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::I64X2_NEG => {
                let src_fp = expect_v128_reg(self, src, "arm64 i64x2.neg needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::neg_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_NEG => {
                let src_fp = expect_v128_reg(self, src, "arm64 f32x4.neg needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fneg_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_ABS => {
                let src_fp = expect_v128_reg(self, src, "arm64 f32x4.abs needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fabs_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_SQRT => {
                let src_fp = expect_v128_reg(self, src, "arm64 f32x4.sqrt needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fsqrt_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_CEIL => {
                let src_fp = expect_v128_reg(self, src, "arm64 f32x4.ceil needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintp_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_FLOOR => {
                let src_fp = expect_v128_reg(self, src, "arm64 f32x4.floor needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintm_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_TRUNC => {
                let src_fp = expect_v128_reg(self, src, "arm64 f32x4.trunc needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintz_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F32X4_NEAREST => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 f32x4.nearest needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintn_4s(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_NEG => {
                let src_fp = expect_v128_reg(self, src, "arm64 f64x2.neg needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fneg_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_ABS => {
                let src_fp = expect_v128_reg(self, src, "arm64 f64x2.abs needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fabs_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_SQRT => {
                let src_fp = expect_v128_reg(self, src, "arm64 f64x2.sqrt needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::fsqrt_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_CEIL => {
                let src_fp = expect_v128_reg(self, src, "arm64 f64x2.ceil needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintp_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_FLOOR => {
                let src_fp = expect_v128_reg(self, src, "arm64 f64x2.floor needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintm_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_TRUNC => {
                let src_fp = expect_v128_reg(self, src, "arm64 f64x2.trunc needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintz_2d(dst_fp, src_fp));
                Ok(())
            }
            OpcodeFD::F64X2_NEAREST => {
                let src_fp =
                    expect_v128_reg(self, src, "arm64 f64x2.nearest needs a vector source")?;
                let dst_fp = self.map_fp_reg(dst)?;
                self.core.text.emit_u32(enc::frintn_2d(dst_fp, src_fp));
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("unary", opcode)),
        }
    }

    fn lower_simd_binary_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let lhs_fp = expect_v128_reg(self, lhs, "arm64 SIMD binary needs a vector lhs")?;
        let rhs_fp = expect_v128_reg(self, rhs, "arm64 SIMD binary needs a vector rhs")?;
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_EQ => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_NE => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_16b(dst_fp, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, dst_fp));
                Ok(())
            }
            OpcodeFD::I8X16_LT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_16b(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_LT_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhi_16b(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_GT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_GT_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhi_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_LE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_16b(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_LE_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhs_16b(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_GE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_GE_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhs_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EQ => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_NE => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_8h(dst_fp, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, dst_fp));
                Ok(())
            }
            OpcodeFD::I16X8_LT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_8h(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_LT_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhi_8h(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_GT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_GT_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhi_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_LE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_8h(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_LE_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhs_8h(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_GE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_GE_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhs_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EQ => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_NE => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_4s(dst_fp, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, dst_fp));
                Ok(())
            }
            OpcodeFD::I32X4_LT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_4s(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_LT_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhi_4s(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_GT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_GT_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhi_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_LE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_4s(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_LE_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhs_4s(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_GE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_GE_U => {
                self.core
                    .text
                    .emit_u32(enc::cmhs_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EQ => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_NE => {
                self.core
                    .text
                    .emit_u32(enc::cmeq_2d(dst_fp, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, dst_fp));
                Ok(())
            }
            OpcodeFD::I64X2_LT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_2d(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_GT_S => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_LE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_2d(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_GE_S => {
                self.core
                    .text
                    .emit_u32(enc::cmge_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_EQ => {
                self.core
                    .text
                    .emit_u32(enc::fcmeq_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_NE => {
                self.core
                    .text
                    .emit_u32(enc::fcmeq_4s(dst_fp, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, dst_fp));
                Ok(())
            }
            OpcodeFD::F32X4_LT => {
                self.core
                    .text
                    .emit_u32(enc::fcmgt_4s(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_GT => {
                self.core
                    .text
                    .emit_u32(enc::fcmgt_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_LE => {
                self.core
                    .text
                    .emit_u32(enc::fcmge_4s(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_GE => {
                self.core
                    .text
                    .emit_u32(enc::fcmge_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_EQ => {
                self.core
                    .text
                    .emit_u32(enc::fcmeq_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_NE => {
                self.core
                    .text
                    .emit_u32(enc::fcmeq_2d(dst_fp, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::mvn_16b(dst_fp, dst_fp));
                Ok(())
            }
            OpcodeFD::F64X2_LT => {
                self.core
                    .text
                    .emit_u32(enc::fcmgt_2d(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_GT => {
                self.core
                    .text
                    .emit_u32(enc::fcmgt_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_LE => {
                self.core
                    .text
                    .emit_u32(enc::fcmge_2d(dst_fp, rhs_fp, lhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_GE => {
                self.core
                    .text
                    .emit_u32(enc::fcmge_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_ADD => {
                self.core
                    .text
                    .emit_u32(enc::add_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_SUB => {
                self.core
                    .text
                    .emit_u32(enc::sub_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_NARROW_I16X8_S => {
                let rhs_tmp = if dst_fp == rhs_fp {
                    let tmp = self.fp_scratch.scoped_alloc().detach();
                    self.core.text.emit_u32(enc::orr_16b(*tmp, rhs_fp, rhs_fp));
                    Some(tmp)
                } else {
                    None
                };
                let rhs_src = rhs_tmp.as_ref().map(|tmp| **tmp).unwrap_or(rhs_fp);
                self.core.text.emit_u32(enc::sqxtn_8b(dst_fp, lhs_fp));
                self.core.text.emit_u32(enc::sqxtn2_16b(dst_fp, rhs_src));
                Ok(())
            }
            OpcodeFD::I8X16_NARROW_I16X8_U => {
                let rhs_tmp = if dst_fp == rhs_fp {
                    let tmp = self.fp_scratch.scoped_alloc().detach();
                    self.core.text.emit_u32(enc::orr_16b(*tmp, rhs_fp, rhs_fp));
                    Some(tmp)
                } else {
                    None
                };
                let rhs_src = rhs_tmp.as_ref().map(|tmp| **tmp).unwrap_or(rhs_fp);
                self.core.text.emit_u32(enc::sqxtun_8b(dst_fp, lhs_fp));
                self.core.text.emit_u32(enc::sqxtun2_16b(dst_fp, rhs_src));
                Ok(())
            }
            OpcodeFD::I8X16_ADD_SAT_S => {
                self.core
                    .text
                    .emit_u32(enc::sqadd_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_ADD_SAT_U => {
                self.core
                    .text
                    .emit_u32(enc::uqadd_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_SUB_SAT_S => {
                self.core
                    .text
                    .emit_u32(enc::sqsub_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_SUB_SAT_U => {
                self.core
                    .text
                    .emit_u32(enc::uqsub_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_MIN_S => {
                self.core
                    .text
                    .emit_u32(enc::smin_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_MIN_U => {
                self.core
                    .text
                    .emit_u32(enc::umin_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_MAX_S => {
                self.core
                    .text
                    .emit_u32(enc::smax_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_MAX_U => {
                self.core
                    .text
                    .emit_u32(enc::umax_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_AVGR_U => {
                self.core
                    .text
                    .emit_u32(enc::urhadd_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_ADD => {
                self.core.text.emit_u32(enc::add_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_SUB => {
                self.core.text.emit_u32(enc::sub_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_NARROW_I32X4_S => {
                let rhs_tmp = if dst_fp == rhs_fp {
                    let tmp = self.fp_scratch.scoped_alloc().detach();
                    self.core.text.emit_u32(enc::orr_16b(*tmp, rhs_fp, rhs_fp));
                    Some(tmp)
                } else {
                    None
                };
                let rhs_src = rhs_tmp.as_ref().map(|tmp| **tmp).unwrap_or(rhs_fp);
                self.core.text.emit_u32(enc::sqxtn_4h(dst_fp, lhs_fp));
                self.core.text.emit_u32(enc::sqxtn2_8h(dst_fp, rhs_src));
                Ok(())
            }
            OpcodeFD::I16X8_NARROW_I32X4_U => {
                let rhs_tmp = if dst_fp == rhs_fp {
                    let tmp = self.fp_scratch.scoped_alloc().detach();
                    self.core.text.emit_u32(enc::orr_16b(*tmp, rhs_fp, rhs_fp));
                    Some(tmp)
                } else {
                    None
                };
                let rhs_src = rhs_tmp.as_ref().map(|tmp| **tmp).unwrap_or(rhs_fp);
                self.core.text.emit_u32(enc::sqxtun_4h(dst_fp, lhs_fp));
                self.core.text.emit_u32(enc::sqxtun2_8h(dst_fp, rhs_src));
                Ok(())
            }
            OpcodeFD::I16X8_MUL => {
                self.core.text.emit_u32(enc::mul_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_ADD_SAT_S => {
                self.core
                    .text
                    .emit_u32(enc::sqadd_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_ADD_SAT_U => {
                self.core
                    .text
                    .emit_u32(enc::uqadd_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_SUB_SAT_S => {
                self.core
                    .text
                    .emit_u32(enc::sqsub_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_SUB_SAT_U => {
                self.core
                    .text
                    .emit_u32(enc::uqsub_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_MIN_S => {
                self.core
                    .text
                    .emit_u32(enc::smin_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_MIN_U => {
                self.core
                    .text
                    .emit_u32(enc::umin_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_MAX_S => {
                self.core
                    .text
                    .emit_u32(enc::smax_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_MAX_U => {
                self.core
                    .text
                    .emit_u32(enc::umax_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_AVGR_U => {
                self.core
                    .text
                    .emit_u32(enc::urhadd_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_Q15MULR_SAT_S => {
                self.core
                    .text
                    .emit_u32(enc::sqrdmulh_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_RELAXED_DOT_I8X16_I7X16_S => {
                self.lower_i16x8_relaxed_dot_i8x16_i7x16_s(dst_fp, lhs_fp, rhs_fp)
            }
            OpcodeFD::I16X8_EXTMUL_LOW_I8X16_S => {
                self.core
                    .text
                    .emit_u32(enc::smull_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_S => {
                self.core
                    .text
                    .emit_u32(enc::smull2_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTMUL_LOW_I8X16_U => {
                self.core
                    .text
                    .emit_u32(enc::umull_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_U => {
                self.core
                    .text
                    .emit_u32(enc::umull2_8h(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::V128_AND => {
                self.core
                    .text
                    .emit_u32(enc::and_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::V128_ANDNOT => {
                self.core
                    .text
                    .emit_u32(enc::bic_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::V128_OR => {
                self.core
                    .text
                    .emit_u32(enc::orr_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::V128_XOR => {
                self.core
                    .text
                    .emit_u32(enc::eor_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I8X16_SWIZZLE => {
                self.core
                    .text
                    .emit_u32(enc::tbl_16b(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_ADD => {
                self.core.text.emit_u32(enc::add_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_SUB => {
                self.core.text.emit_u32(enc::sub_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_MUL => {
                self.core.text.emit_u32(enc::mul_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_MIN_S => {
                self.core
                    .text
                    .emit_u32(enc::smin_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_MIN_U => {
                self.core
                    .text
                    .emit_u32(enc::umin_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_MAX_S => {
                self.core
                    .text
                    .emit_u32(enc::smax_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_MAX_U => {
                self.core
                    .text
                    .emit_u32(enc::umax_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_DOT_I16X8_S => {
                let low = self.fp_scratch.scoped_alloc().detach();
                let high = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::smull_4s(*low, lhs_fp, rhs_fp));
                self.core
                    .text
                    .emit_u32(enc::smull2_4s(*high, lhs_fp, rhs_fp));
                self.core.text.emit_u32(enc::addp_4s(dst_fp, *low, *high));
                Ok(())
            }
            OpcodeFD::I32X4_EXTMUL_LOW_I16X8_S => {
                self.core
                    .text
                    .emit_u32(enc::smull_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_S => {
                self.core
                    .text
                    .emit_u32(enc::smull2_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTMUL_LOW_I16X8_U => {
                self.core
                    .text
                    .emit_u32(enc::umull_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_U => {
                self.core
                    .text
                    .emit_u32(enc::umull2_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_ADD => {
                self.core.text.emit_u32(enc::add_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_SUB => {
                self.core.text.emit_u32(enc::sub_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_MUL => self.lower_i64x2_mul_native(dst_fp, lhs_fp, rhs_fp),
            OpcodeFD::I64X2_EXTMUL_LOW_I32X4_S => {
                self.core
                    .text
                    .emit_u32(enc::smull_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_S => {
                self.core
                    .text
                    .emit_u32(enc::smull2_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTMUL_LOW_I32X4_U => {
                self.core
                    .text
                    .emit_u32(enc::umull_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_U => {
                self.core
                    .text
                    .emit_u32(enc::umull2_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_ADD => {
                self.core
                    .text
                    .emit_u32(enc::fadd_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_SUB => {
                self.core
                    .text
                    .emit_u32(enc::fsub_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_MUL => {
                self.core
                    .text
                    .emit_u32(enc::fmul_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_DIV => {
                self.core
                    .text
                    .emit_u32(enc::fdiv_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_MIN => {
                self.core
                    .text
                    .emit_u32(enc::fmin_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_MAX => {
                self.core
                    .text
                    .emit_u32(enc::fmax_4s(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F32X4_PMIN => {
                self.lower_f32x4_pseudo_minmax(dst_fp, lhs_fp, rhs_fp, false);
                Ok(())
            }
            OpcodeFD::F32X4_PMAX => {
                self.lower_f32x4_pseudo_minmax(dst_fp, lhs_fp, rhs_fp, true);
                Ok(())
            }
            OpcodeFD::F64X2_ADD => {
                self.core
                    .text
                    .emit_u32(enc::fadd_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_SUB => {
                self.core
                    .text
                    .emit_u32(enc::fsub_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_MUL => {
                self.core
                    .text
                    .emit_u32(enc::fmul_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_DIV => {
                self.core
                    .text
                    .emit_u32(enc::fdiv_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_MIN => {
                self.core
                    .text
                    .emit_u32(enc::fmin_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_MAX => {
                self.core
                    .text
                    .emit_u32(enc::fmax_2d(dst_fp, lhs_fp, rhs_fp));
                Ok(())
            }
            OpcodeFD::F64X2_PMIN => {
                self.lower_f64x2_pseudo_minmax(dst_fp, lhs_fp, rhs_fp, false);
                Ok(())
            }
            OpcodeFD::F64X2_PMAX => {
                self.lower_f64x2_pseudo_minmax(dst_fp, lhs_fp, rhs_fp, true);
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("binary", opcode)),
        }
    }

    fn lower_simd_shuffle_native(
        &mut self,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
        lanes: [u8; 16],
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let lhs_fp = expect_v128_reg(self, lhs, "arm64 shuffle needs a vector lhs")?;
        let rhs_fp = expect_v128_reg(self, rhs, "arm64 shuffle needs a vector rhs")?;
        let tmp0 = self.fp_scratch.scoped_alloc().detach();
        let tmp1 = self.fp_scratch.scoped_alloc().detach();
        let (table_lo, table_hi) = if tmp0.index() < tmp1.index() {
            (*tmp0, *tmp1)
        } else {
            (*tmp1, *tmp0)
        };
        debug_assert_eq!(table_hi.index(), table_lo.index() + 1);
        self.core
            .text
            .emit_u32(enc::orr_16b(table_lo, lhs_fp, lhs_fp));
        self.core
            .text
            .emit_u32(enc::orr_16b(table_hi, rhs_fp, rhs_fp));
        self.materialize_v128_constant_into_fp(dst_fp, lanes);
        self.core
            .text
            .emit_u32(enc::tbl_16b_2(dst_fp, table_lo, dst_fp));
        Ok(())
    }

    fn lower_f32x4_pseudo_minmax(
        &mut self,
        dst_fp: Arm64FpReg,
        lhs_fp: Arm64FpReg,
        rhs_fp: Arm64FpReg,
        is_max: bool,
    ) {
        let pick_rhs = self.fp_scratch.scoped_alloc().detach();
        let merge = self.fp_scratch.scoped_alloc().detach();
        let cmp = if is_max {
            enc::fcmgt_4s(*pick_rhs, rhs_fp, lhs_fp)
        } else {
            enc::fcmgt_4s(*pick_rhs, lhs_fp, rhs_fp)
        };
        self.core.text.emit_u32(cmp);
        self.core
            .text
            .emit_u32(enc::bic_16b(dst_fp, lhs_fp, *pick_rhs));
        self.core
            .text
            .emit_u32(enc::and_16b(*merge, rhs_fp, *pick_rhs));
        self.core
            .text
            .emit_u32(enc::orr_16b(dst_fp, dst_fp, *merge));
    }

    fn lower_f64x2_pseudo_minmax(
        &mut self,
        dst_fp: Arm64FpReg,
        lhs_fp: Arm64FpReg,
        rhs_fp: Arm64FpReg,
        is_max: bool,
    ) {
        let pick_rhs = self.fp_scratch.scoped_alloc().detach();
        let merge = self.fp_scratch.scoped_alloc().detach();
        let cmp = if is_max {
            enc::fcmgt_2d(*pick_rhs, rhs_fp, lhs_fp)
        } else {
            enc::fcmgt_2d(*pick_rhs, lhs_fp, rhs_fp)
        };
        self.core.text.emit_u32(cmp);
        self.core
            .text
            .emit_u32(enc::bic_16b(dst_fp, lhs_fp, *pick_rhs));
        self.core
            .text
            .emit_u32(enc::and_16b(*merge, rhs_fp, *pick_rhs));
        self.core
            .text
            .emit_u32(enc::orr_16b(dst_fp, dst_fp, *merge));
    }

    fn lower_simd_extract_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        _dst_ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = expect_v128_reg(self, src, "arm64 extract_lane needs a vector source")?;
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_EXTRACT_LANE_S => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core
                    .text
                    .emit_u32(enc::smov_lane_to_gp(dst_gp, src_fp, lane, 1));
                Ok(())
            }
            OpcodeFD::I8X16_EXTRACT_LANE_U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(dst_gp, src_fp, lane, 1));
                Ok(())
            }
            OpcodeFD::I16X8_EXTRACT_LANE_S => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core
                    .text
                    .emit_u32(enc::smov_lane_to_gp(dst_gp, src_fp, lane, 2));
                Ok(())
            }
            OpcodeFD::I16X8_EXTRACT_LANE_U => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(dst_gp, src_fp, lane, 2));
                Ok(())
            }
            OpcodeFD::I32X4_EXTRACT_LANE => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(dst_gp, src_fp, lane, 4));
                Ok(())
            }
            OpcodeFD::I64X2_EXTRACT_LANE => {
                let dst_gp = self.map_gp_reg(dst)?;
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(dst_gp, src_fp, lane, 8));
                Ok(())
            }
            OpcodeFD::F32X4_EXTRACT_LANE => {
                let dst_fp = self.map_fp_reg(dst)?;
                let raw = *self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(raw, src_fp, lane, 4));
                self.core.text.emit_u32(enc::fmov_s_from_gp(dst_fp, raw));
                Ok(())
            }
            OpcodeFD::F64X2_EXTRACT_LANE => {
                let dst_fp = self.map_fp_reg(dst)?;
                let raw = *self.gp_scratch.scoped_alloc();
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(raw, src_fp, lane, 8));
                self.core.text.emit_u32(enc::fmov_d_from_gp(dst_fp, raw));
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("extract_lane", opcode)),
        }
    }

    fn lower_simd_replace_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        dst: MachineReg,
        vector: MachineValue,
        scalar: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let vector_fp = expect_v128_reg(self, vector, "arm64 replace_lane needs a vector input")?;
        if dst_fp != vector_fp {
            self.core
                .text
                .emit_u32(enc::orr_16b(dst_fp, vector_fp, vector_fp));
        }

        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_REPLACE_LANE => {
                let scalar_gp = match scalar {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(
                        self,
                        scalar,
                        "arm64 i8x16.replace_lane needs a GP scalar",
                    )?,
                };
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, scalar_gp, 1));
                Ok(())
            }
            OpcodeFD::I16X8_REPLACE_LANE => {
                let scalar_gp = match scalar {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(
                        self,
                        scalar,
                        "arm64 i16x8.replace_lane needs a GP scalar",
                    )?,
                };
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, scalar_gp, 2));
                Ok(())
            }
            OpcodeFD::I32X4_REPLACE_LANE => {
                let scalar_gp = match scalar {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(
                        self,
                        scalar,
                        "arm64 i32x4.replace_lane needs a GP scalar",
                    )?,
                };
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, scalar_gp, 4));
                Ok(())
            }
            OpcodeFD::I64X2_REPLACE_LANE => {
                let scalar_gp = match scalar {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => expect_scalar_gp_reg(
                        self,
                        scalar,
                        "arm64 i64x2.replace_lane needs a GP scalar",
                    )?,
                };
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, scalar_gp, 8));
                Ok(())
            }
            OpcodeFD::F32X4_REPLACE_LANE => {
                let scalar_gp = match scalar {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => {
                        let scalar_fp = expect_scalar_fp_reg(
                            self,
                            scalar,
                            "arm64 f32x4.replace_lane needs an FP scalar",
                        )?;
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.core.text.emit_u32(enc::fmov_gp_from_s(gp, scalar_fp));
                        gp
                    }
                };
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, scalar_gp, 4));
                Ok(())
            }
            OpcodeFD::F64X2_REPLACE_LANE => {
                let scalar_gp = match scalar {
                    MachineValue::Imm64(value) => {
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.materialize_u64(gp, value);
                        gp
                    }
                    _ => {
                        let scalar_fp = expect_scalar_fp_reg(
                            self,
                            scalar,
                            "arm64 f64x2.replace_lane needs an FP scalar",
                        )?;
                        let gp = *self.gp_scratch.scoped_alloc();
                        self.core.text.emit_u32(enc::fmov_gp_from_d(gp, scalar_fp));
                        gp
                    }
                };
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, scalar_gp, 8));
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("replace_lane", opcode)),
        }
    }

    fn lower_simd_ternary_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        a: MachineValue,
        b: MachineValue,
        c: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let a_fp = expect_v128_reg(self, a, "arm64 SIMD ternary needs a vector a")?;
        let b_fp = expect_v128_reg(self, b, "arm64 SIMD ternary needs a vector b")?;
        let c_fp = expect_v128_reg(self, c, "arm64 SIMD ternary needs a vector c")?;
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_BITSELECT => {
                let tmp = self.fp_scratch.scoped_alloc();
                self.core.text.emit_u32(enc::bic_16b(*tmp, b_fp, c_fp));
                self.core.text.emit_u32(enc::and_16b(dst_fp, a_fp, c_fp));
                self.core.text.emit_u32(enc::orr_16b(dst_fp, dst_fp, *tmp));
                Ok(())
            }
            OpcodeFD::I8X16_RELAXED_LANESELECT
            | OpcodeFD::I16X8_RELAXED_LANESELECT
            | OpcodeFD::I32X4_RELAXED_LANESELECT
            | OpcodeFD::I64X2_RELAXED_LANESELECT => {
                self.lower_relaxed_laneselect(OpcodeFD::try_from(opcode)?, dst_fp, a_fp, b_fp, c_fp)
            }
            OpcodeFD::F32X4_RELAXED_MADD
            | OpcodeFD::F32X4_RELAXED_NMADD
            | OpcodeFD::F64X2_RELAXED_MADD
            | OpcodeFD::F64X2_RELAXED_NMADD => {
                self.lower_relaxed_madd_nmadd(OpcodeFD::try_from(opcode)?, dst_fp, a_fp, b_fp, c_fp)
            }
            OpcodeFD::I32X4_RELAXED_DOT_I8X16_I7X16_ADD_S => {
                self.lower_i32x4_relaxed_dot_i8x16_i7x16_add_s(dst_fp, a_fp, b_fp, c_fp)
            }
            _ => Err(unsupported_simd_opcode("ternary", opcode)),
        }
    }

    fn lower_i16x8_relaxed_dot_i8x16_i7x16_s(
        &mut self,
        dst_fp: Arm64FpReg,
        lhs_fp: Arm64FpReg,
        rhs_fp: Arm64FpReg,
    ) -> Result<(), WasmError> {
        let low = self.fp_scratch.scoped_alloc().detach();
        let high = self.fp_scratch.scoped_alloc().detach();
        self.core.text.emit_u32(enc::smull_8h(*low, lhs_fp, rhs_fp));
        self.core
            .text
            .emit_u32(enc::smull2_8h(*high, lhs_fp, rhs_fp));
        self.core.text.emit_u32(enc::addp_8h(dst_fp, *low, *high));
        Ok(())
    }

    fn lower_i32x4_relaxed_dot_i8x16_i7x16_add_s(
        &mut self,
        dst_fp: Arm64FpReg,
        lhs_fp: Arm64FpReg,
        rhs_fp: Arm64FpReg,
        add_fp: Arm64FpReg,
    ) -> Result<(), WasmError> {
        let dot = self.fp_scratch.scoped_alloc().detach();
        let high = self.fp_scratch.scoped_alloc().detach();
        self.core.text.emit_u32(enc::smull_8h(*dot, lhs_fp, rhs_fp));
        self.core
            .text
            .emit_u32(enc::smull2_8h(*high, lhs_fp, rhs_fp));
        self.core.text.emit_u32(enc::addp_8h(*dot, *dot, *high));
        self.core.text.emit_u32(enc::saddlp_4s(*high, *dot));
        self.core.text.emit_u32(enc::add_4s(dst_fp, *high, add_fp));
        Ok(())
    }

    fn lower_relaxed_laneselect(
        &mut self,
        opcode: OpcodeFD,
        dst_fp: Arm64FpReg,
        a_fp: Arm64FpReg,
        b_fp: Arm64FpReg,
        mask_fp: Arm64FpReg,
    ) -> Result<(), WasmError> {
        let tmp = self.fp_scratch.scoped_alloc().detach();
        let lane_mask = self.fp_scratch.scoped_alloc().detach();
        self.core.text.emit_u32(enc::eor_16b(*tmp, *tmp, *tmp));
        match opcode {
            OpcodeFD::I8X16_RELAXED_LANESELECT => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_16b(*lane_mask, *tmp, mask_fp));
            }
            OpcodeFD::I16X8_RELAXED_LANESELECT => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_8h(*lane_mask, *tmp, mask_fp));
            }
            OpcodeFD::I32X4_RELAXED_LANESELECT => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_4s(*lane_mask, *tmp, mask_fp));
            }
            OpcodeFD::I64X2_RELAXED_LANESELECT => {
                self.core
                    .text
                    .emit_u32(enc::cmgt_2d(*lane_mask, *tmp, mask_fp));
            }
            other => return Err(unsupported_simd_opcode("relaxed_laneselect", other as u32)),
        }
        self.core
            .text
            .emit_u32(enc::bic_16b(*tmp, b_fp, *lane_mask));
        self.core
            .text
            .emit_u32(enc::and_16b(dst_fp, a_fp, *lane_mask));
        self.core.text.emit_u32(enc::orr_16b(dst_fp, dst_fp, *tmp));
        Ok(())
    }

    fn lower_relaxed_madd_nmadd(
        &mut self,
        opcode: OpcodeFD,
        dst_fp: Arm64FpReg,
        a_fp: Arm64FpReg,
        b_fp: Arm64FpReg,
        c_fp: Arm64FpReg,
    ) -> Result<(), WasmError> {
        let product = self.fp_scratch.scoped_alloc().detach();
        match opcode {
            OpcodeFD::F32X4_RELAXED_MADD => {
                self.core.text.emit_u32(enc::fmul_4s(*product, a_fp, b_fp));
                self.core
                    .text
                    .emit_u32(enc::fadd_4s(dst_fp, *product, c_fp));
                Ok(())
            }
            OpcodeFD::F32X4_RELAXED_NMADD => {
                self.core.text.emit_u32(enc::fmul_4s(*product, a_fp, b_fp));
                self.core
                    .text
                    .emit_u32(enc::fsub_4s(dst_fp, c_fp, *product));
                Ok(())
            }
            OpcodeFD::F64X2_RELAXED_MADD => {
                self.core.text.emit_u32(enc::fmul_2d(*product, a_fp, b_fp));
                self.core
                    .text
                    .emit_u32(enc::fadd_2d(dst_fp, *product, c_fp));
                Ok(())
            }
            OpcodeFD::F64X2_RELAXED_NMADD => {
                self.core.text.emit_u32(enc::fmul_2d(*product, a_fp, b_fp));
                self.core
                    .text
                    .emit_u32(enc::fsub_2d(dst_fp, c_fp, *product));
                Ok(())
            }
            other => Err(unsupported_simd_opcode("relaxed_madd_nmadd", other as u32)),
        }
    }

    fn lower_simd_shift_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        vector: MachineValue,
        shift: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let src_fp = expect_v128_reg(self, vector, "arm64 SIMD shift needs a vector source")?;
        let (lane_bits, duplicate, shift_op, right_shift) = match OpcodeFD::try_from(opcode)? {
            OpcodeFD::I8X16_SHL => (8u32, ShiftDup::B8, ShiftOp::Unsigned, false),
            OpcodeFD::I8X16_SHR_S => (8u32, ShiftDup::B8, ShiftOp::Signed, true),
            OpcodeFD::I8X16_SHR_U => (8u32, ShiftDup::B8, ShiftOp::Unsigned, true),
            OpcodeFD::I16X8_SHL => (16u32, ShiftDup::H16, ShiftOp::Unsigned, false),
            OpcodeFD::I16X8_SHR_S => (16u32, ShiftDup::H16, ShiftOp::Signed, true),
            OpcodeFD::I16X8_SHR_U => (16u32, ShiftDup::H16, ShiftOp::Unsigned, true),
            OpcodeFD::I32X4_SHL => (32u32, ShiftDup::S32, ShiftOp::Unsigned, false),
            OpcodeFD::I32X4_SHR_S => (32u32, ShiftDup::S32, ShiftOp::Signed, true),
            OpcodeFD::I32X4_SHR_U => (32u32, ShiftDup::S32, ShiftOp::Unsigned, true),
            OpcodeFD::I64X2_SHL => (64u32, ShiftDup::D64, ShiftOp::Unsigned, false),
            OpcodeFD::I64X2_SHR_S => (64u32, ShiftDup::D64, ShiftOp::Signed, true),
            OpcodeFD::I64X2_SHR_U => (64u32, ShiftDup::D64, ShiftOp::Unsigned, true),
            _ => return Err(unsupported_simd_opcode("shift", opcode)),
        };

        let shift_gp = *self.gp_scratch.scoped_alloc();
        match shift {
            MachineValue::Imm64(value) => {
                self.materialize_u64(shift_gp, value & u64::from(lane_bits - 1));
            }
            _ => {
                let src_gp =
                    expect_scalar_gp_reg(self, shift, "arm64 SIMD shift needs a GP shift count")?;
                if shift_gp != src_gp {
                    self.core.text.emit_u32(enc::mov_reg_32(shift_gp, src_gp));
                }
                if let Some(and) = enc::and_imm_32(shift_gp, shift_gp, lane_bits - 1) {
                    self.core.text.emit_u32(and);
                } else {
                    return Err(WasmError::internal(
                        "arm64 SIMD shift mask was not encodable",
                    ));
                }
            }
        }

        if right_shift {
            self.core
                .text
                .emit_u32(enc::sub_reg_32(shift_gp, abi::zero_reg(), shift_gp));
        }

        let counts = *self.fp_scratch.scoped_alloc();
        match duplicate {
            ShiftDup::B8 => {
                self.core
                    .text
                    .emit_u32(enc::dup_16b_from_gp(counts, shift_gp));
            }
            ShiftDup::H16 => {
                self.core
                    .text
                    .emit_u32(enc::dup_8h_from_gp(counts, shift_gp));
            }
            ShiftDup::S32 => {
                self.core
                    .text
                    .emit_u32(enc::dup_4s_from_gp(counts, shift_gp));
            }
            ShiftDup::D64 => {
                self.core.text.emit_u32(enc::sxtw(shift_gp, shift_gp));
                self.core
                    .text
                    .emit_u32(enc::dup_2d_from_gp(counts, shift_gp));
            }
        }

        match (duplicate, shift_op) {
            (ShiftDup::B8, ShiftOp::Signed) => {
                self.core
                    .text
                    .emit_u32(enc::sshl_16b(dst_fp, src_fp, counts));
            }
            (ShiftDup::B8, ShiftOp::Unsigned) => {
                self.core
                    .text
                    .emit_u32(enc::ushl_16b(dst_fp, src_fp, counts));
            }
            (ShiftDup::H16, ShiftOp::Signed) => {
                self.core
                    .text
                    .emit_u32(enc::sshl_8h(dst_fp, src_fp, counts));
            }
            (ShiftDup::H16, ShiftOp::Unsigned) => {
                self.core
                    .text
                    .emit_u32(enc::ushl_8h(dst_fp, src_fp, counts));
            }
            (ShiftDup::S32, ShiftOp::Signed) => {
                self.core
                    .text
                    .emit_u32(enc::sshl_4s(dst_fp, src_fp, counts));
            }
            (ShiftDup::S32, ShiftOp::Unsigned) => {
                self.core
                    .text
                    .emit_u32(enc::ushl_4s(dst_fp, src_fp, counts));
            }
            (ShiftDup::D64, ShiftOp::Signed) => {
                self.core
                    .text
                    .emit_u32(enc::sshl_2d(dst_fp, src_fp, counts));
            }
            (ShiftDup::D64, ShiftOp::Unsigned) => {
                self.core
                    .text
                    .emit_u32(enc::ushl_2d(dst_fp, src_fp, counts));
            }
        }
        Ok(())
    }

    fn lower_simd_load_native(
        &mut self,
        opcode: u32,
        dst: MachineReg,
        addr: MachineAddr,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_LOAD => {
                if let Some(imm12) = encode_q_imm(addr) {
                    self.core
                        .text
                        .emit_u32(enc::ldr_q(dst_fp, self.map_gp_reg(addr.base)?, imm12));
                    return Ok(());
                }
                Err(WasmError::internal(
                    "arm64 v128.load currently requires an immediate-offset address",
                ))
            }
            OpcodeFD::V128_LOAD8X8_S | OpcodeFD::V128_LOAD8X8_U => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load8x8_* requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_d(*tmp, base, imm12));
                let op = match OpcodeFD::try_from(opcode)? {
                    OpcodeFD::V128_LOAD8X8_S => enc::sshll_8h(dst_fp, *tmp),
                    OpcodeFD::V128_LOAD8X8_U => enc::ushll_8h(dst_fp, *tmp),
                    _ => unreachable!(),
                };
                self.core.text.emit_u32(op);
                Ok(())
            }
            OpcodeFD::V128_LOAD16X4_S | OpcodeFD::V128_LOAD16X4_U => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load16x4_* requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_d(*tmp, base, imm12));
                let op = match OpcodeFD::try_from(opcode)? {
                    OpcodeFD::V128_LOAD16X4_S => enc::sshll_4s(dst_fp, *tmp),
                    OpcodeFD::V128_LOAD16X4_U => enc::ushll_4s(dst_fp, *tmp),
                    _ => unreachable!(),
                };
                self.core.text.emit_u32(op);
                Ok(())
            }
            OpcodeFD::V128_LOAD32X2_S | OpcodeFD::V128_LOAD32X2_U => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load32x2_* requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_d(*tmp, base, imm12));
                let op = match OpcodeFD::try_from(opcode)? {
                    OpcodeFD::V128_LOAD32X2_S => enc::sshll_2d(dst_fp, *tmp),
                    OpcodeFD::V128_LOAD32X2_U => enc::ushll_2d(dst_fp, *tmp),
                    _ => unreachable!(),
                };
                self.core.text.emit_u32(op);
                Ok(())
            }
            OpcodeFD::V128_LOAD8_SPLAT => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 1) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load8_splat requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let raw = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldrb_imm(*raw, base, imm12));
                self.core.text.emit_u32(enc::dup_16b_from_gp(dst_fp, *raw));
                Ok(())
            }
            OpcodeFD::V128_LOAD16_SPLAT => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 2) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load16_splat requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let raw = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldrh_imm(*raw, base, imm12));
                self.core.text.emit_u32(enc::dup_8h_from_gp(dst_fp, *raw));
                Ok(())
            }
            OpcodeFD::V128_LOAD32_SPLAT => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 4) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load32_splat requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let raw = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_32(*raw, base, imm12));
                self.core.text.emit_u32(enc::dup_4s_from_gp(dst_fp, *raw));
                Ok(())
            }
            OpcodeFD::V128_LOAD64_SPLAT => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load64_splat requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let raw = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_64(*raw, base, imm12));
                self.core.text.emit_u32(enc::dup_2d_from_gp(dst_fp, *raw));
                Ok(())
            }
            OpcodeFD::V128_LOAD32_ZERO => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 4) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load32_zero requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let raw = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_32(*raw, base, imm12));
                self.core
                    .text
                    .emit_u32(enc::eor_16b(dst_fp, dst_fp, dst_fp));
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, 0, *raw, 4));
                Ok(())
            }
            OpcodeFD::V128_LOAD64_ZERO => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load64_zero requires an immediate-offset address",
                    ));
                };
                let base = self.map_gp_reg(addr.base)?;
                let raw = self.gp_scratch.scoped_alloc().detach();
                self.core.text.emit_u32(enc::ldr_64(*raw, base, imm12));
                self.core
                    .text
                    .emit_u32(enc::eor_16b(dst_fp, dst_fp, dst_fp));
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, 0, *raw, 8));
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("load", opcode)),
        }
    }

    fn lower_simd_store_native(
        &mut self,
        opcode: u32,
        addr: MachineAddr,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = expect_v128_reg(self, src, "arm64 SIMD store needs a vector source")?;
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_STORE => {
                if let Some(imm12) = encode_q_imm(addr) {
                    self.core
                        .text
                        .emit_u32(enc::str_q(src_fp, self.map_gp_reg(addr.base)?, imm12));
                    return Ok(());
                }
                Err(WasmError::internal(
                    "arm64 v128.store currently requires an immediate-offset address",
                ))
            }
            _ => Err(unsupported_simd_opcode("store", opcode)),
        }
    }

    fn lower_simd_load_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        dst: MachineReg,
        addr: MachineAddr,
        vector: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)?;
        let vector_fp = expect_v128_reg(self, vector, "arm64 SIMD load-lane needs a vector input")?;
        if dst_fp != vector_fp {
            self.core
                .text
                .emit_u32(enc::orr_16b(dst_fp, vector_fp, vector_fp));
        }
        let base = self.map_gp_reg(addr.base)?;
        let raw = self.gp_scratch.scoped_alloc().detach();
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_LOAD8_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 1) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load8_lane requires an immediate-offset address",
                    ));
                };
                self.core.text.emit_u32(enc::ldrb_imm(*raw, base, imm12));
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, *raw, 1));
                Ok(())
            }
            OpcodeFD::V128_LOAD16_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 2) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load16_lane requires an immediate-offset address",
                    ));
                };
                self.core.text.emit_u32(enc::ldrh_imm(*raw, base, imm12));
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, *raw, 2));
                Ok(())
            }
            OpcodeFD::V128_LOAD32_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 4) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load32_lane requires an immediate-offset address",
                    ));
                };
                self.core.text.emit_u32(enc::ldr_32(*raw, base, imm12));
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, *raw, 4));
                Ok(())
            }
            OpcodeFD::V128_LOAD64_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.load64_lane requires an immediate-offset address",
                    ));
                };
                self.core.text.emit_u32(enc::ldr_64(*raw, base, imm12));
                self.core
                    .text
                    .emit_u32(enc::ins_lane_from_gp(dst_fp, lane, *raw, 8));
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("load_lane", opcode)),
        }
    }

    fn lower_simd_store_lane_native(
        &mut self,
        opcode: u32,
        lane: u8,
        addr: MachineAddr,
        vector: MachineValue,
    ) -> Result<(), WasmError> {
        let vector_fp =
            expect_v128_reg(self, vector, "arm64 SIMD store-lane needs a vector input")?;
        let base = self.map_gp_reg(addr.base)?;
        let raw = self.gp_scratch.scoped_alloc().detach();
        match OpcodeFD::try_from(opcode)? {
            OpcodeFD::V128_STORE8_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 1) else {
                    return Err(WasmError::internal(
                        "arm64 v128.store8_lane requires an immediate-offset address",
                    ));
                };
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*raw, vector_fp, lane, 1));
                self.core.text.emit_u32(enc::strb_imm(*raw, base, imm12));
                Ok(())
            }
            OpcodeFD::V128_STORE16_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 2) else {
                    return Err(WasmError::internal(
                        "arm64 v128.store16_lane requires an immediate-offset address",
                    ));
                };
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*raw, vector_fp, lane, 2));
                self.core.text.emit_u32(enc::strh_imm(*raw, base, imm12));
                Ok(())
            }
            OpcodeFD::V128_STORE32_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 4) else {
                    return Err(WasmError::internal(
                        "arm64 v128.store32_lane requires an immediate-offset address",
                    ));
                };
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*raw, vector_fp, lane, 4));
                self.core.text.emit_u32(enc::str_32(*raw, base, imm12));
                Ok(())
            }
            OpcodeFD::V128_STORE64_LANE => {
                let Some(imm12) = encode_scaled_imm(addr.offset, 8) else {
                    return Err(WasmError::internal(
                        "arm64 v128.store64_lane requires an immediate-offset address",
                    ));
                };
                self.core
                    .text
                    .emit_u32(enc::umov_lane_to_gp(*raw, vector_fp, lane, 8));
                self.core.text.emit_u32(enc::str_64(*raw, base, imm12));
                Ok(())
            }
            _ => Err(unsupported_simd_opcode("store_lane", opcode)),
        }
    }

    fn lower_i64x2_mul_native(
        &mut self,
        dst_fp: Arm64FpReg,
        lhs_fp: Arm64FpReg,
        rhs_fp: Arm64FpReg,
    ) -> Result<(), WasmError> {
        let lhs = self.gp_scratch.scoped_alloc().detach();
        let rhs = self.gp_scratch.scoped_alloc().detach();
        self.emit_adjust_stack_down(16);
        self.core
            .text
            .emit_u32(enc::umov_lane_to_gp(*lhs, lhs_fp, 0, 8));
        self.core
            .text
            .emit_u32(enc::umov_lane_to_gp(*rhs, rhs_fp, 0, 8));
        self.core.text.emit_u32(enc::mul_64(*lhs, *lhs, *rhs));
        self.core
            .text
            .emit_u32(enc::str_64(*lhs, abi::stack_reg(), 0));
        self.core
            .text
            .emit_u32(enc::umov_lane_to_gp(*lhs, lhs_fp, 1, 8));
        self.core
            .text
            .emit_u32(enc::umov_lane_to_gp(*rhs, rhs_fp, 1, 8));
        self.core.text.emit_u32(enc::mul_64(*lhs, *lhs, *rhs));
        self.core
            .text
            .emit_u32(enc::str_64(*lhs, abi::stack_reg(), 1));
        self.core
            .text
            .emit_u32(enc::ldr_q(dst_fp, abi::stack_reg(), 0));
        self.emit_adjust_stack_up(16);
        Ok(())
    }

    fn lower_bitmask_from_v128(
        &mut self,
        src_fp: Arm64FpReg,
        dst_gp: Arm64Reg,
        lane_bits: u32,
    ) -> Result<(), WasmError> {
        let lane = *self.gp_scratch.scoped_alloc();
        self.materialize_u64(dst_gp, 0);
        self.emit_adjust_stack_down(16);
        self.core
            .text
            .emit_u32(enc::str_q(src_fp, abi::stack_reg(), 0));

        let lane_count = 128 / lane_bits;
        let lane_bytes = lane_bits / 8;
        for lane_idx in 0..lane_count {
            let byte_off = lane_idx * lane_bytes;
            match lane_bits {
                8 => self
                    .core
                    .text
                    .emit_u32(enc::ldrb_imm(lane, abi::stack_reg(), byte_off)),
                16 => self
                    .core
                    .text
                    .emit_u32(enc::ldrh_imm(lane, abi::stack_reg(), byte_off / 2)),
                32 => self
                    .core
                    .text
                    .emit_u32(enc::ldr_32(lane, abi::stack_reg(), byte_off / 4)),
                64 => self
                    .core
                    .text
                    .emit_u32(enc::ldr_64(lane, abi::stack_reg(), byte_off / 8)),
                _ => {
                    return Err(WasmError::internal(
                        "arm64 SIMD bitmask lane width is invalid",
                    ))
                }
            };
            match lane_bits {
                8 => self.core.text.emit_u32(enc::lsr_imm_32(lane, lane, 7)),
                16 => self.core.text.emit_u32(enc::lsr_imm_32(lane, lane, 15)),
                32 => self.core.text.emit_u32(enc::lsr_imm_32(lane, lane, 31)),
                64 => self.core.text.emit_u32(enc::lsr_imm_64(lane, lane, 63)),
                _ => unreachable!(),
            };
            if lane_idx != 0 {
                self.core
                    .text
                    .emit_u32(enc::lsl_imm_64(lane, lane, lane_idx));
            }
            self.core
                .text
                .emit_u32(enc::orr_reg_64(dst_gp, dst_gp, lane));
        }
        self.emit_adjust_stack_up(16);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ShiftDup {
    B8,
    H16,
    S32,
    D64,
}

#[derive(Clone, Copy)]
enum ShiftOp {
    Signed,
    Unsigned,
}
