//! RV64 backend implementation.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        jit::arch::{
            common::{
                backend::ArchBackend,
                core::CompilerCore,
                pipeline::emit_call_arg_lanes,
                scratch_pool::ScratchPool,
                template::{
                    decode_template_chain_next, encode_template_chain_next, template_i32_delta,
                    TemplateBranchSense,
                },
                types::ParallelSource,
            },
            shared_64::{EmittedFunction64, ModuleLinkBackend64},
        },
        jit::machine::machine_ir::{
            MachineAddr, MachineBlockId, MachineBlockParam, MachineBranchCond, MachineCallArgs,
            MachineCallResults, MachineCallTarget, MachineCompareKind, MachineConstId,
            MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp, MachineFloatWidth,
            MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
            MachineLoadExtension, MachineMemWidth, MachineReg, MachineShiftOp, MachineSign,
            MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue, MACHINE_CTX_REG,
            MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        jit::runtime::{
            code::NativeRootEntry, code_buf::CodeBuffer, context::ctx_offset,
            runtime_call::call_runtime_entry_ptr, trap::raise_trap,
        },
    },
};

use super::abi;
use crate::vm::jit::arch::riscv::{
    enc,
    reg::{RiscvFpReg, RiscvReg},
};

use crate::vm::jit::arch::common::{
    helpers::{convert_result_float_width, is_fallthrough_edge, trap_code},
    types::DirectCallPatch,
};
use crate::vm::jit::runtime::preserved::{io as preserved_io, op as preserved_op};

const STACK_SLOT_BYTES: i32 = 8;
const CALLEE_SAVED_GP_FRAME_SIZE: i32 = abi::callee_saved_gp_count() as i32 * STACK_SLOT_BYTES;
const CALLEE_SAVED_FP_FRAME_OFFSET: i32 = CALLEE_SAVED_GP_FRAME_SIZE;
const CALLEE_SAVED_FP_FRAME_SIZE: i32 = abi::callee_saved_fp_count() as i32 * STACK_SLOT_BYTES;
const CALLEE_SAVED_FRAME_SIZE: i32 =
    ((CALLEE_SAVED_FP_FRAME_OFFSET + CALLEE_SAVED_FP_FRAME_SIZE + 15) / 16) * 16;
const BODY_LINK_RA_OFFSET: i32 = 0;
const BODY_LINK_FRAME_SIZE: i32 = 16;
const CALL_RECORD_SIZE: i32 = 16;

fn caller_results_base_delta(results: &MachineCallResults) -> u32 {
    match results {
        MachineCallResults::FrameFallback { caller_results, .. } => {
            u32::from(caller_results.base_slot) * 8
        }
        MachineCallResults::None
        | MachineCallResults::ScalarGp { .. }
        | MachineCallResults::ScalarGpPair { .. }
        | MachineCallResults::ScalarFp { .. } => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    Jal { rd: RiscvReg },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    pub inst_offset: usize,
    pub label: usize,
    pub kind: BranchFixupKind,
}

#[derive(Clone, Debug)]
pub(crate) struct CompiledRiscv64Entry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    #[cfg(sf_ir_dump)]
    pub debug_regions: collections::Vec<crate::vm::jit::arch::common::types::DebugRegion>,
}

struct RiscvTruncSpec {
    width: MachineFloatWidth,
    upper_bits: u64,
    lower_bits: u64,
    lower_traps_on_le: bool,
}

impl RiscvTruncSpec {
    fn new(op: MachineConvertOp) -> Self {
        match op {
            MachineConvertOp::I32TruncF32S => Self {
                width: MachineFloatWidth::F32,
                upper_bits: 2147483648.0_f32.to_bits() as u64,
                lower_bits: (-2147483648.0_f32).to_bits() as u64,
                lower_traps_on_le: false,
            },
            MachineConvertOp::I32TruncF32U => Self {
                width: MachineFloatWidth::F32,
                upper_bits: 4294967296.0_f32.to_bits() as u64,
                lower_bits: (-1.0_f32).to_bits() as u64,
                lower_traps_on_le: true,
            },
            MachineConvertOp::I32TruncF64S => Self {
                width: MachineFloatWidth::F64,
                upper_bits: 2147483648.0_f64.to_bits(),
                lower_bits: (-2147483649.0_f64).to_bits(),
                lower_traps_on_le: true,
            },
            MachineConvertOp::I32TruncF64U => Self {
                width: MachineFloatWidth::F64,
                upper_bits: 4294967296.0_f64.to_bits(),
                lower_bits: (-1.0_f64).to_bits(),
                lower_traps_on_le: true,
            },
            MachineConvertOp::I64TruncF32S => Self {
                width: MachineFloatWidth::F32,
                upper_bits: 9223372036854775808.0_f32.to_bits() as u64,
                lower_bits: (-9223372036854775808.0_f32).to_bits() as u64,
                lower_traps_on_le: false,
            },
            MachineConvertOp::I64TruncF32U => Self {
                width: MachineFloatWidth::F32,
                upper_bits: 18446744073709551616.0_f32.to_bits() as u64,
                lower_bits: (-1.0_f32).to_bits() as u64,
                lower_traps_on_le: true,
            },
            MachineConvertOp::I64TruncF64S => Self {
                width: MachineFloatWidth::F64,
                upper_bits: 9223372036854775808.0_f64.to_bits(),
                lower_bits: (-9223372036854775808.0_f64).to_bits(),
                lower_traps_on_le: false,
            },
            MachineConvertOp::I64TruncF64U => Self {
                width: MachineFloatWidth::F64,
                upper_bits: 18446744073709551616.0_f64.to_bits(),
                lower_bits: (-1.0_f64).to_bits(),
                lower_traps_on_le: true,
            },
            _ => unreachable!("not a trapping trunc op"),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Riscv64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: collections::Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<RiscvReg, 3>,
    pub(super) fp_scratch: ScratchPool<RiscvFpReg, 2>,
}

impl<'a> ArchBackend<'a> for Riscv64Backend<'a> {
    const NAME: &'static str = "riscv64";

    fn max_total_regs() -> usize {
        abi::max_total_machine_regs()
    }

    fn max_fp_regs() -> usize {
        abi::max_fp_machine_regs()
    }

    fn new(core: CompilerCore<'a>) -> Self {
        Self {
            core,
            fixups: collections::Vec::new(),
            gp_scratch: abi::new_gp_scratch_pool(),
            fp_scratch: abi::new_fp_scratch_pool(),
        }
    }

    fn core(&self) -> &CompilerCore<'a> {
        &self.core
    }

    fn core_mut(&mut self) -> &mut CompilerCore<'a> {
        &mut self.core
    }

    fn into_core(self) -> CompilerCore<'a> {
        self.core
    }

    fn lower_prologue(&mut self) {
        self.emit_addi(abi::stack_reg(), abi::stack_reg(), -CALLEE_SAVED_FRAME_SIZE);
        for (index, &reg) in abi::callee_saved_regs().iter().enumerate() {
            self.emit_sd(reg, abi::stack_reg(), (index as i32) * STACK_SLOT_BYTES);
        }
        for (index, &reg) in abi::callee_saved_fp_regs().iter().enumerate() {
            self.emit_fsd(
                reg,
                abi::stack_reg(),
                CALLEE_SAVED_FP_FRAME_OFFSET + (index as i32) * STACK_SLOT_BYTES,
            );
        }

        self.emit_mv(abi::map_fixed_reg(MACHINE_CTX_REG), abi::C_ARG0);
        self.emit_mv(abi::map_fixed_reg(MACHINE_FP_REG), abi::C_ARG1);
        self.emit_ld(
            abi::map_fixed_reg(MACHINE_MEM0_BASE_REG),
            abi::map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_BASE as i32,
        );
        self.emit_ld(
            abi::map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            abi::map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_SIZE as i32,
        );
    }

    fn lower_root_caller_stub(&mut self) {
        let fp = abi::map_fixed_reg(MACHINE_FP_REG);
        self.emit_addi(abi::stack_reg(), abi::stack_reg(), -CALL_RECORD_SIZE);
        self.emit_sd(fp, abi::stack_reg(), 0);
        self.emit_sd(fp, abi::stack_reg(), 8);
        self.lower_root_param_lanes_from_frame();
        let internal_entry_label = self.core.internal_entry_label;
        self.emit_jal(abi::link_reg(), internal_entry_label);
    }

    fn lower_epilogue(&mut self) {
        for (index, &reg) in abi::callee_saved_fp_regs().iter().enumerate() {
            self.emit_fld(
                reg,
                abi::stack_reg(),
                CALLEE_SAVED_FP_FRAME_OFFSET + (index as i32) * STACK_SLOT_BYTES,
            );
        }
        for (index, &reg) in abi::callee_saved_regs().iter().enumerate() {
            self.emit_ld(reg, abi::stack_reg(), (index as i32) * STACK_SLOT_BYTES);
        }
        self.emit_addi(abi::stack_reg(), abi::stack_reg(), CALLEE_SAVED_FRAME_SIZE);
        self.core.text.emit_u32(enc::ret());
    }

    fn lower_body_prelude(&mut self) {
        self.emit_addi(abi::stack_reg(), abi::stack_reg(), -BODY_LINK_FRAME_SIZE);
        self.emit_sd(abi::link_reg(), abi::stack_reg(), BODY_LINK_RA_OFFSET);
    }

    fn lower_body_local_error_tail(&mut self) {
        self.emit_ld(abi::link_reg(), abi::stack_reg(), BODY_LINK_RA_OFFSET);
        self.emit_restore_host_platform_regs(0);
        self.emit_ld(
            abi::map_fixed_reg(MACHINE_FP_REG),
            abi::stack_reg(),
            BODY_LINK_FRAME_SIZE + 8,
        );
        self.emit_addi(
            abi::stack_reg(),
            abi::stack_reg(),
            BODY_LINK_FRAME_SIZE + CALL_RECORD_SIZE,
        );
        self.core.text.emit_u32(enc::ret());
    }

    fn emit_inst_at(&mut self, inst: &'a MachineInst, index: usize) -> Result<(), WasmError> {
        self.core.current_op_index = Some(index);
        self.lower_inst(inst)?;
        self.gp_scratch.assert_all_free();
        self.fp_scratch.assert_all_free();
        Ok(())
    }

    fn end_block(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core.current_op_index = None;
        let result = self.lower_terminator(term, fallthrough);
        self.gp_scratch.assert_all_free();
        self.fp_scratch.assert_all_free();
        self.core.current_block = None;
        result
    }

    fn lower_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        self.lower_inst_dispatch(inst)
    }

    fn lower_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.lower_terminator_dispatch(term, fallthrough)
    }

    fn lower_trap(&mut self, kind: MachineTrapKind) {
        self.lower_trap_dispatch(kind);
    }

    fn lower_unconditional_branch(&mut self, label: usize) {
        self.emit_jal(abi::zero_reg(), label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        const JAL_MIN: isize = -(1 << 20);
        const JAL_MAX: isize = (1 << 20) - 2;
        for fixup in &self.fixups {
            let target = self
                .core
                .labels
                .get(fixup.label)
                .and_then(|offset| *offset)
                .ok_or_else(|| WasmError::internal("riscv64 branch target label is unresolved"))?;
            let delta = target as isize - fixup.inst_offset as isize;
            if delta & 1 != 0 {
                return Err(WasmError::internal(
                    "riscv64 branch fixup target is not halfword aligned",
                ));
            }
            if !(JAL_MIN..=JAL_MAX).contains(&delta) {
                return Err(WasmError::internal(
                    "riscv64 jal branch fixup is out of pc-relative range",
                ));
            }
            let patched = match fixup.kind {
                BranchFixupKind::Jal { rd } => enc::jal(rd, delta as i32),
            };
            self.core.text.patch_u32(fixup.inst_offset, patched);
        }
        Ok(())
    }

    fn alloc_gp_scratch(&mut self) -> u8 {
        self.gp_scratch.alloc()
    }

    fn free_gp_scratch(&mut self, id: u8) {
        self.gp_scratch.free_index(id)
    }

    fn alloc_fp_scratch(&mut self) -> u8 {
        self.fp_scratch.alloc()
    }

    fn free_fp_scratch(&mut self, id: u8) {
        self.fp_scratch.free_index(id)
    }

    fn lower_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        self.lower_source_move_dispatch(dst, src)
    }

    fn lower_gp_cycle_break(
        &mut self,
        dst: MachineReg,
        src: MachineReg,
        scratch_id: u8,
    ) -> Result<(), WasmError> {
        let temp = self.gp_scratch.reg(scratch_id);
        let dst = self.map_gp_reg(dst)?;
        let src = self.map_gp_reg(src)?;
        self.emit_mv(temp, dst);
        self.emit_mv(dst, src);
        Ok(())
    }

    fn lower_fp_cycle_break(
        &mut self,
        _dst: MachineBlockParam,
        _src: MachineReg,
        _float_width: Option<MachineFloatWidth>,
        _scratch_id: u8,
    ) -> Result<(), WasmError> {
        let temp = self.fp_scratch.reg(_scratch_id);
        let dst_fp = self.map_fp_reg(_dst.reg)?;
        let width = _dst.ty.float_width().expect("FP param width");
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmv_s(temp, dst_fp),
            MachineFloatWidth::F64 => enc::fmv_d(temp, dst_fp),
        });
        let src_fp = self.map_fp_reg(_src)?;
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmv_s(dst_fp, src_fp),
            MachineFloatWidth::F64 => enc::fmv_d(dst_fp, src_fp),
        });
        self.core.set_fp_reg_width(_dst.reg, width)?;
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        debug_assert!(
            bytes % 4 == 0,
            "RV64 NOP padding should stay instruction-aligned"
        );
        let nop = enc::nop().to_le_bytes();
        let mut remaining = bytes;
        while remaining >= 4 {
            buf.emit_bytes(&nop);
            remaining -= 4;
        }
        for _ in 0..remaining {
            buf.emit_u8(0);
        }
    }
}

impl<'a> ModuleLinkBackend64<'a> for Riscv64Backend<'a> {
    type CompiledEntry = CompiledRiscv64Entry;

    fn make_entry(buf: &CodeBuffer, emitted: &EmittedFunction64) -> Self::CompiledEntry {
        let entry = unsafe { buf.fn_ptr::<NativeRootEntry>(emitted.text_offset) };
        CompiledRiscv64Entry {
            entry,
            text_len: emitted.text_len,
            #[cfg(sf_ir_dump)]
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}

impl<'a> Riscv64Backend<'a> {
    #[inline]
    fn unsupported_error() -> WasmError {
        WasmError::invalid("riscv64 backend does not support this MachineIR instruction yet")
    }

    #[inline]
    pub(super) fn fits_i12(value: i32) -> bool {
        (-2048..=2047).contains(&value)
    }

    #[inline]
    pub(super) fn emit_addi(&mut self, dst: RiscvReg, src: RiscvReg, imm: i32) {
        debug_assert!(Self::fits_i12(imm));
        self.core.text.emit_u32(enc::addi(dst, src, imm));
    }

    #[inline]
    pub(super) fn emit_mv(&mut self, dst: RiscvReg, src: RiscvReg) {
        if dst != src {
            self.emit_addi(dst, src, 0);
        }
    }

    #[inline]
    fn emit_ld(&mut self, dst: RiscvReg, base: RiscvReg, offset: i32) {
        debug_assert!(Self::fits_i12(offset));
        self.core.text.emit_u32(enc::load(0b011, dst, base, offset));
    }

    #[inline]
    fn emit_sd(&mut self, src: RiscvReg, base: RiscvReg, offset: i32) {
        debug_assert!(Self::fits_i12(offset));
        self.core
            .text
            .emit_u32(enc::store(0b011, src, base, offset));
    }

    pub(super) fn emit_restore_host_platform_regs(&mut self, _body_link_base_offset: i32) {
        // RV64 keeps psABI gp/tp reserved, so generated code never has to
        // repair them at host boundaries.
    }

    #[inline]
    fn emit_fld(&mut self, dst: RiscvFpReg, base: RiscvReg, offset: i32) {
        debug_assert!(Self::fits_i12(offset));
        self.core
            .text
            .emit_u32(enc::fp_load(0b011, dst, base, offset));
    }

    #[inline]
    fn emit_fsd(&mut self, src: RiscvFpReg, base: RiscvReg, offset: i32) {
        debug_assert!(Self::fits_i12(offset));
        self.core
            .text
            .emit_u32(enc::fp_store(0b011, src, base, offset));
    }

    #[inline]
    pub(super) fn emit_jal(&mut self, rd: RiscvReg, label: usize) {
        let inst_offset = self.core.text.emit_u32(enc::jal(rd, 0));
        self.fixups.push(BranchFixup {
            inst_offset,
            label,
            kind: BranchFixupKind::Jal { rd },
        });
    }

    fn map_gp_reg(&self, reg: MachineReg) -> Result<RiscvReg, WasmError> {
        crate::vm::jit::arch::common::helpers::validate_gp_reg(self, reg)?;
        abi::map_reg(reg)
    }

    fn map_fp_reg(&self, reg: MachineReg) -> Result<RiscvFpReg, WasmError> {
        abi::map_fp_reg(reg)
    }

    pub(super) fn materialize_u64(&mut self, dst: RiscvReg, value: u64) {
        let signed = value as i64;
        if (-2048..=2047).contains(&signed) {
            self.emit_addi(dst, abi::zero_reg(), signed as i32);
            return;
        }

        let mut started = false;
        for byte_index in (0..8).rev() {
            let byte = ((value >> (byte_index * 8)) & 0xff) as i32;
            if !started {
                if byte == 0 {
                    continue;
                }
                self.emit_addi(dst, abi::zero_reg(), byte);
                started = true;
            } else {
                self.core.text.emit_u32(enc::slli(dst, dst, 8));
                if byte != 0 {
                    self.core.text.emit_u32(enc::ori(dst, dst, byte));
                }
            }
        }
        if !started {
            self.emit_addi(dst, abi::zero_reg(), 0);
        }
    }

    pub(super) fn load_value_into(
        &mut self,
        dst: RiscvReg,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(reg) => {
                if self.core.is_fp_reg(reg) {
                    let src = self.map_fp_reg(reg)?;
                    match self.core.fp_reg_width(reg)? {
                        MachineFloatWidth::F32 => {
                            self.core.text.emit_u32(enc::fmv_x_w(dst, src));
                            self.zext_w(dst, dst);
                        }
                        MachineFloatWidth::F64 => {
                            self.core.text.emit_u32(enc::fmv_x_d(dst, src));
                        }
                    }
                } else {
                    let src = self.map_gp_reg(reg)?;
                    self.emit_mv(dst, src);
                }
            }
            MachineValue::Imm64(value) => self.materialize_u64(dst, value),
            MachineValue::ReservedReg(_) => {
                return Err(WasmError::internal(
                    "riscv64 cannot consume reserved cache register as a real value",
                ))
            }
        }
        Ok(())
    }

    fn load_fp_value_into(
        &mut self,
        dst: RiscvFpReg,
        width: MachineFloatWidth,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(reg) if self.core.is_fp_reg(reg) => {
                let src = self.map_fp_reg(reg)?;
                let src_width = self.core.fp_reg_width(reg)?;
                if src_width != width {
                    return Err(WasmError::invalid("riscv64 float move width mismatch"));
                }
                if dst != src {
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmv_s(dst, src),
                        MachineFloatWidth::F64 => enc::fmv_d(dst, src),
                    });
                }
            }
            MachineValue::Reg(reg) => {
                let src = self.map_gp_reg(reg)?;
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmv_w_x(dst, src),
                    MachineFloatWidth::F64 => enc::fmv_d_x(dst, src),
                });
            }
            MachineValue::Imm64(value) => {
                let scratch = self.gp_scratch.scoped_alloc().detach();
                self.materialize_u64(*scratch, value);
                self.core.text.emit_u32(match width {
                    MachineFloatWidth::F32 => enc::fmv_w_x(dst, *scratch),
                    MachineFloatWidth::F64 => enc::fmv_d_x(dst, *scratch),
                });
            }
            MachineValue::ReservedReg(_) => {
                return Err(WasmError::internal(
                    "riscv64 cannot consume reserved cache register as an FP value",
                ))
            }
        }
        Ok(())
    }

    fn move_fp_to_gp(&mut self, dst: RiscvReg, src: RiscvFpReg, width: MachineFloatWidth) {
        match width {
            MachineFloatWidth::F32 => {
                self.core.text.emit_u32(enc::fmv_x_w(dst, src));
                self.zext_w(dst, dst);
            }
            MachineFloatWidth::F64 => {
                self.core.text.emit_u32(enc::fmv_x_d(dst, src));
            }
        }
    }

    #[inline]
    fn zext_w(&mut self, dst: RiscvReg, src: RiscvReg) {
        self.core.text.emit_u32(enc::slli(dst, src, 32));
        self.core.text.emit_u32(enc::srli(dst, dst, 32));
    }

    #[inline]
    fn sext_w(&mut self, dst: RiscvReg, src: RiscvReg) {
        self.core.text.emit_u32(enc::addiw(dst, src, 0));
    }

    fn canonicalize_compare_operand(
        &mut self,
        dst: RiscvReg,
        width: MachineIntWidth,
        sign: MachineSign,
    ) {
        if width == MachineIntWidth::I32 {
            match sign {
                MachineSign::Signed => self.sext_w(dst, dst),
                MachineSign::Unsigned => self.zext_w(dst, dst),
            }
        }
    }

    pub(super) fn emit_load_raw(
        &mut self,
        funct3: u32,
        dst: RiscvReg,
        base: RiscvReg,
        offset: i32,
    ) {
        if Self::fits_i12(offset) {
            self.core
                .text
                .emit_u32(enc::load(funct3, dst, base, offset));
            return;
        }
        let addr = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*addr, offset as i64 as u64);
        self.core.text.emit_u32(enc::add(*addr, base, *addr));
        self.core.text.emit_u32(enc::load(funct3, dst, *addr, 0));
    }

    pub(super) fn emit_store_raw(
        &mut self,
        funct3: u32,
        src: RiscvReg,
        base: RiscvReg,
        offset: i32,
    ) {
        if Self::fits_i12(offset) {
            self.core
                .text
                .emit_u32(enc::store(funct3, src, base, offset));
            return;
        }
        let addr = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*addr, offset as i64 as u64);
        self.core.text.emit_u32(enc::add(*addr, base, *addr));
        self.core.text.emit_u32(enc::store(funct3, src, *addr, 0));
    }

    pub(super) fn emit_fp_load_raw(
        &mut self,
        funct3: u32,
        dst: RiscvFpReg,
        base: RiscvReg,
        offset: i32,
    ) {
        if Self::fits_i12(offset) {
            self.core
                .text
                .emit_u32(enc::fp_load(funct3, dst, base, offset));
            return;
        }
        let addr = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*addr, offset as i64 as u64);
        self.core.text.emit_u32(enc::add(*addr, base, *addr));
        self.core.text.emit_u32(enc::fp_load(funct3, dst, *addr, 0));
    }

    pub(super) fn emit_fp_store_raw(
        &mut self,
        funct3: u32,
        src: RiscvFpReg,
        base: RiscvReg,
        offset: i32,
    ) {
        if Self::fits_i12(offset) {
            self.core
                .text
                .emit_u32(enc::fp_store(funct3, src, base, offset));
            return;
        }
        let addr = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*addr, offset as i64 as u64);
        self.core.text.emit_u32(enc::add(*addr, base, *addr));
        self.core
            .text
            .emit_u32(enc::fp_store(funct3, src, *addr, 0));
    }

    fn load_funct3(
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<u32, WasmError> {
        Ok(match (width, extension) {
            (MachineMemWidth::U8, MachineLoadExtension::SignExtend) => 0b000,
            (
                MachineMemWidth::U8,
                MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ) => 0b100,
            (MachineMemWidth::U16, MachineLoadExtension::SignExtend) => 0b001,
            (
                MachineMemWidth::U16,
                MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ) => 0b101,
            (MachineMemWidth::U32, MachineLoadExtension::SignExtend) => 0b010,
            (
                MachineMemWidth::U32,
                MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ) => 0b110,
            (
                MachineMemWidth::U64,
                MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ) => 0b011,
            (MachineMemWidth::U64, MachineLoadExtension::SignExtend) => {
                return Err(WasmError::invalid(
                    "riscv64 backend does not support sign-extending U64 loads",
                ))
            }
        })
    }

    #[inline]
    fn store_funct3(width: MachineMemWidth) -> u32 {
        match width {
            MachineMemWidth::U8 => 0b000,
            MachineMemWidth::U16 => 0b001,
            MachineMemWidth::U32 => 0b010,
            MachineMemWidth::U64 => 0b011,
        }
    }

    fn patch_pcrel_literal_load(
        &mut self,
        auipc_offset: usize,
        load_offset: usize,
        dst: RiscvReg,
        literal_offset: usize,
    ) -> Result<(), WasmError> {
        let delta = literal_offset as isize - auipc_offset as isize;
        let hi = (delta + 0x800) >> 12;
        let lo = delta - (hi << 12);
        if !(-524_288..=524_287).contains(&hi) || !(-2048..=2047).contains(&lo) {
            return Err(WasmError::internal(
                "riscv64 inline literal is out of pc-relative load range",
            ));
        }
        self.core
            .text
            .patch_u32(auipc_offset, enc::auipc(dst, (hi as u32) & 0x000f_ffff));
        self.core
            .text
            .patch_u32(load_offset, enc::load(0b011, dst, dst, lo as i32));
        Ok(())
    }

    fn align_inline_literal(&mut self) {
        if self.core.text.len() & 7 != 0 {
            self.core.text.emit_u32(enc::nop());
        }
    }

    fn emit_direct_call_target_literal(
        &mut self,
        dst: RiscvReg,
        callee: crate::vm::jit::machine::machine_ir::MachineFuncId,
        skip_after_return: bool,
    ) -> Result<(), WasmError> {
        let auipc_offset = self.core.text.emit_u32(enc::auipc(dst, 0));
        let load_offset = self.core.text.emit_u32(enc::load(0b011, dst, dst, 0));
        if skip_after_return {
            self.core.text.emit_u32(enc::jalr(abi::link_reg(), dst, 0));
            let after_literal = self.core.new_label();
            self.emit_jal(abi::zero_reg(), after_literal);
            self.align_inline_literal();
            let literal_offset = self.core.text.emit_u64(0);
            self.core.bind_label(after_literal);
            self.patch_pcrel_literal_load(auipc_offset, load_offset, dst, literal_offset)?;
            self.core
                .direct_call_patches
                .push(DirectCallPatch::address_literal(literal_offset, callee));
        } else {
            self.core.text.emit_u32(enc::jalr(abi::zero_reg(), dst, 0));
            self.align_inline_literal();
            let literal_offset = self.core.text.emit_u64(0);
            self.patch_pcrel_literal_load(auipc_offset, load_offset, dst, literal_offset)?;
            self.core
                .direct_call_patches
                .push(DirectCallPatch::address_literal(literal_offset, callee));
        }
        Ok(())
    }

    fn branch_cond_for_compare(kind: MachineCompareKind, sign: MachineSign) -> (enc::Cond, bool) {
        match (kind, sign) {
            (MachineCompareKind::Eq, _) => (enc::Cond::Eq, false),
            (MachineCompareKind::Ne, _) => (enc::Cond::Ne, false),
            (MachineCompareKind::Lt, MachineSign::Signed) => (enc::Cond::Lt, false),
            (MachineCompareKind::Ge, MachineSign::Signed) => (enc::Cond::Ge, false),
            (MachineCompareKind::Gt, MachineSign::Signed) => (enc::Cond::Lt, true),
            (MachineCompareKind::Le, MachineSign::Signed) => (enc::Cond::Ge, true),
            (MachineCompareKind::Lt, MachineSign::Unsigned) => (enc::Cond::Ltu, false),
            (MachineCompareKind::Ge, MachineSign::Unsigned) => (enc::Cond::Geu, false),
            (MachineCompareKind::Gt, MachineSign::Unsigned) => (enc::Cond::Ltu, true),
            (MachineCompareKind::Le, MachineSign::Unsigned) => (enc::Cond::Geu, true),
        }
    }

    pub(super) fn emit_branch_to(
        &mut self,
        cond: enc::Cond,
        lhs: RiscvReg,
        rhs: RiscvReg,
        label: usize,
    ) {
        self.core
            .text
            .emit_u32(enc::b_type(cond.invert(), lhs, rhs, 8));
        self.emit_jal(abi::zero_reg(), label);
    }

    fn lower_branch_if_cond(
        &mut self,
        cond: &MachineBranchCond,
        label: usize,
        take_true: bool,
    ) -> Result<(), WasmError> {
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(value) => {
                    if (value != 0) == take_true {
                        self.emit_jal(abi::zero_reg(), label);
                    }
                }
                MachineValue::Reg(reg) => {
                    let cond_s = self.gp_scratch.scoped_alloc().detach();
                    self.load_value_into(*cond_s, MachineValue::Reg(reg))?;
                    self.zext_w(*cond_s, *cond_s);
                    let branch = if take_true {
                        enc::Cond::Ne
                    } else {
                        enc::Cond::Eq
                    };
                    self.emit_branch_to(branch, *cond_s, abi::zero_reg(), label);
                }
                MachineValue::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "riscv64 branch condition cannot read reserved cache register",
                    ))
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                let lhs_s = self.gp_scratch.scoped_alloc().detach();
                let rhs_s = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*lhs_s, lhs)?;
                self.load_value_into(*rhs_s, rhs)?;
                self.canonicalize_compare_operand(*lhs_s, width, sign);
                self.canonicalize_compare_operand(*rhs_s, width, sign);
                let (mut branch, swap) = Self::branch_cond_for_compare(kind, sign);
                if !take_true {
                    branch = branch.invert();
                }
                let (lhs, rhs) = if swap {
                    (*rhs_s, *lhs_s)
                } else {
                    (*lhs_s, *rhs_s)
                };
                self.emit_branch_to(branch, lhs, rhs, label);
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                let src_s = self.gp_scratch.scoped_alloc().detach();
                let mask_s = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*src_s, src)?;
                self.load_value_into(*mask_s, mask)?;
                if width == MachineIntWidth::I32 {
                    self.zext_w(*src_s, *src_s);
                    self.zext_w(*mask_s, *mask_s);
                }
                self.core.text.emit_u32(enc::and(*src_s, *src_s, *mask_s));
                let mut branch = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "riscv64 TestBits branch uses unsupported compare kind",
                        ))
                    }
                };
                if !take_true {
                    branch = branch.invert();
                }
                self.emit_branch_to(branch, *src_s, abi::zero_reg(), label);
            }
        }
        Ok(())
    }

    pub(crate) fn emit_template_skip_unless(
        &mut self,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        skip_bytes: usize,
    ) -> Result<(), WasmError> {
        let skip_when_true = jump_when == TemplateBranchSense::IfFalse;
        match *cond {
            MachineBranchCond::Value(value) => match value {
                MachineValue::Imm64(value) => {
                    let cond_true = value != 0;
                    if cond_true == skip_when_true {
                        let skip_target = self.template_skip_target(skip_bytes)?;
                        let delta = template_i32_delta(self.core.text.len(), skip_target)?;
                        self.core.text.emit_u32(enc::jal(abi::zero_reg(), delta));
                    }
                }
                MachineValue::Reg(reg) => {
                    let cond_s = self.gp_scratch.scoped_alloc().detach();
                    self.load_value_into(*cond_s, MachineValue::Reg(reg))?;
                    self.zext_w(*cond_s, *cond_s);
                    let branch = if skip_when_true {
                        enc::Cond::Ne
                    } else {
                        enc::Cond::Eq
                    };
                    let skip_target = self.template_skip_target(skip_bytes)?;
                    self.emit_template_cond_branch_to_offset(
                        branch,
                        *cond_s,
                        abi::zero_reg(),
                        skip_target,
                    )?;
                }
                MachineValue::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "riscv64 template branch cannot read reserved cache register",
                    ))
                }
            },
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                let lhs_s = self.gp_scratch.scoped_alloc().detach();
                let rhs_s = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*lhs_s, lhs)?;
                self.load_value_into(*rhs_s, rhs)?;
                self.canonicalize_compare_operand(*lhs_s, width, sign);
                self.canonicalize_compare_operand(*rhs_s, width, sign);
                let (mut branch, swap) = Self::branch_cond_for_compare(kind, sign);
                if !skip_when_true {
                    branch = branch.invert();
                }
                let (lhs, rhs) = if swap {
                    (*rhs_s, *lhs_s)
                } else {
                    (*lhs_s, *rhs_s)
                };
                let skip_target = self.template_skip_target(skip_bytes)?;
                self.emit_template_cond_branch_to_offset(branch, lhs, rhs, skip_target)?;
            }
            MachineBranchCond::TestBits {
                width,
                kind,
                src,
                mask,
            } => {
                let src_s = self.gp_scratch.scoped_alloc().detach();
                let mask_s = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*src_s, src)?;
                self.load_value_into(*mask_s, mask)?;
                if width == MachineIntWidth::I32 {
                    self.zext_w(*src_s, *src_s);
                    self.zext_w(*mask_s, *mask_s);
                }
                self.core.text.emit_u32(enc::and(*src_s, *src_s, *mask_s));
                let mut branch = match kind {
                    MachineCompareKind::Eq => enc::Cond::Eq,
                    MachineCompareKind::Ne => enc::Cond::Ne,
                    _ => {
                        return Err(WasmError::internal(
                            "riscv64 template TestBits branch uses unsupported compare kind",
                        ))
                    }
                };
                if !skip_when_true {
                    branch = branch.invert();
                }
                let skip_target = self.template_skip_target(skip_bytes)?;
                self.emit_template_cond_branch_to_offset(
                    branch,
                    *src_s,
                    abi::zero_reg(),
                    skip_target,
                )?;
            }
        }
        Ok(())
    }

    fn template_skip_target(&self, skip_bytes: usize) -> Result<usize, WasmError> {
        self.core
            .text
            .len()
            .checked_add(4)
            .and_then(|offset| offset.checked_add(skip_bytes))
            .ok_or_else(|| WasmError::internal("riscv64 template skip offset overflow"))
    }

    fn emit_template_cond_branch_to_offset(
        &mut self,
        cond: enc::Cond,
        lhs: RiscvReg,
        rhs: RiscvReg,
        target: usize,
    ) -> Result<(), WasmError> {
        let site = self.core.text.len();
        let delta = template_i32_delta(site, target)?;
        if !(-4096..=4094).contains(&delta) || delta & 1 != 0 {
            return Err(WasmError::internal(
                "riscv64 template conditional branch out of range",
            ));
        }
        self.core.text.emit_u32(enc::b_type(cond, lhs, rhs, delta));
        Ok(())
    }

    pub(crate) fn emit_template_jump_placeholder(
        &mut self,
        next: usize,
    ) -> Result<usize, WasmError> {
        let site = self.core.text.emit_u32(encode_template_chain_next(next)?);
        self.core.text.emit_u32(0);
        Ok(site)
    }

    pub(crate) fn read_template_jump_next(&self, site: usize) -> Result<usize, WasmError> {
        Ok(decode_template_chain_next(self.core.text.read_u32(site)))
    }

    pub(crate) fn patch_template_jump(
        &mut self,
        site: usize,
        target: usize,
    ) -> Result<(), WasmError> {
        self.patch_template_long_jump(site, target)
    }

    pub(crate) fn emit_template_jump_to_offset(&mut self, target: usize) -> Result<(), WasmError> {
        let site = self.core.text.emit_u32(0);
        self.core.text.emit_u32(0);
        self.patch_template_long_jump(site, target)
    }

    fn patch_template_long_jump(&mut self, site: usize, target: usize) -> Result<(), WasmError> {
        let delta = target as isize - site as isize;
        let hi = (delta + 0x800) >> 12;
        let lo = delta - (hi << 12);
        if !(-524_288..=524_287).contains(&hi) || !(-2048..=2047).contains(&lo) {
            return Err(WasmError::internal(
                "riscv64 template jump out of pc-relative range",
            ));
        }
        let scratch = RiscvReg::X5;
        self.core
            .text
            .patch_u32(site, enc::auipc(scratch, (hi as u32) & 0x000f_ffff));
        self.core
            .text
            .patch_u32(site + 4, enc::jalr(abi::zero_reg(), scratch, lo as i32));
        Ok(())
    }

    fn lower_inst_dispatch(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        match &inst.kind {
            MachineInstKind::Move { ty, dst, src, .. } => self.lower_move(*ty, *dst, *src),
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
                addr, width, src, ..
            } => self.lower_store(*addr, *width, *src),
            MachineInstKind::IndexedLoad {
                dst,
                base,
                index,
                index_extend,
                offset,
                width,
                extension,
            } => self.lower_indexed_load(
                *dst,
                *base,
                *index,
                *index_extend,
                *offset,
                *width,
                *extension,
            ),
            MachineInstKind::IndexedStore {
                base,
                index,
                index_extend,
                offset,
                width,
                src,
            } => self.lower_indexed_store(*base, *index, *index_extend, *offset, *width, *src),
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
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => self.lower_int_compare(*width, *kind, *sign, *dst, *lhs, *rhs),
            MachineInstKind::BitfieldExtractU {
                width,
                dst,
                src,
                lsb,
                bits,
            } => self.lower_bitfield_extract_u(*width, *dst, *src, *lsb, *bits),
            MachineInstKind::IntBinaryShifted {
                width,
                op,
                dst,
                lhs,
                rhs,
                shift,
                amount,
            } => self.lower_int_binary_shifted(*width, *op, *dst, *lhs, *rhs, *shift, *amount),
            MachineInstKind::TestBits {
                width,
                kind,
                dst,
                src,
                mask,
            } => self.lower_test_bits(*width, *kind, *dst, *src, *mask),
            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
            } => self.lower_select(*ty, *dst, *on_true, *on_false, *cond),
            MachineInstKind::TrapIf { kind, cond } => self.lower_trap_if(*kind, cond),
            MachineInstKind::CallRuntime(call) => self.lower_call_runtime(call.metadata.0 as usize),
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
            MachineInstKind::EhThrow { tag_idx, args } => self.lower_preserved_no_result(
                preserved_op::EH_THROW,
                *tag_idx,
                0,
                MachineValue::Reg(MACHINE_FP_REG),
                MachineValue::Imm64(args.start.0 as u64),
                MachineValue::Imm64(args.count as u64),
            ),
            MachineInstKind::EhThrowRef { exnref_slot } => self.lower_preserved_no_result(
                preserved_op::EH_THROW_REF,
                0,
                0,
                MachineValue::Reg(MACHINE_FP_REG),
                MachineValue::Imm64(exnref_slot.0 as u64),
                MachineValue::Imm64(0),
            ),
            MachineInstKind::EhAllocExnRef { tag_idx, dst } => self.lower_preserved_result(
                preserved_op::EH_ALLOC_EXN_REF,
                *tag_idx,
                0,
                MachineValue::Reg(MACHINE_FP_REG),
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::MemoryGrow {
                mem_idx,
                dst,
                delta,
            } => self.lower_memory_grow(*mem_idx, *dst, *delta),
            MachineInstKind::MemoryFill {
                mem_idx,
                dest,
                val,
                len,
            } => self.lower_memory_fill(*mem_idx, *dest, *val, *len),
            MachineInstKind::MemoryCopy {
                dst_mem,
                src_mem,
                dest,
                src,
                len,
            } => self.lower_memory_copy(*dst_mem, *src_mem, *dest, *src, *len),
            MachineInstKind::MemoryInit {
                mem_idx,
                data_idx,
                dest,
                src,
                len,
            } => self.lower_memory_init(*mem_idx, *data_idx, *dest, *src, *len),
            MachineInstKind::DataDrop { data_idx } => self.lower_data_drop(*data_idx),
            MachineInstKind::TableGrow {
                table_idx,
                dst,
                init_val,
                delta,
            } => self.lower_table_grow(*table_idx, *dst, *init_val, *delta),
            MachineInstKind::TableFill {
                table_idx,
                start,
                val,
                len,
            } => self.lower_table_fill(*table_idx, *start, *val, *len),
            MachineInstKind::TableCopy {
                dst_tbl,
                src_tbl,
                dest,
                src,
                len,
            } => self.lower_table_copy(*dst_tbl, *src_tbl, *dest, *src, *len),
            MachineInstKind::TableInit {
                table_idx,
                elem_idx,
                dest,
                src,
                len,
            } => self.lower_table_init(*table_idx, *elem_idx, *dest, *src, *len),
            MachineInstKind::ElemDrop { elem_idx } => self.lower_elem_drop(*elem_idx),
            MachineInstKind::RefFunc { func_idx, dst } => self.lower_preserved_result(
                preserved_op::REF_FUNC,
                *func_idx,
                0,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefAsNonNull { src, dst } => self.lower_preserved_result(
                preserved_op::REF_AS_NON_NULL,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefEq { lhs, rhs, dst } => self.lower_preserved_result(
                preserved_op::REF_EQ,
                0,
                0,
                *lhs,
                *rhs,
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::RefI31 { src, dst } => self.lower_preserved_result(
                preserved_op::REF_I31,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::I31GetS { src, dst } => self.lower_preserved_result(
                preserved_op::I31_GET_S,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::I31GetU { src, dst } => self.lower_preserved_result(
                preserved_op::I31_GET_U,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::AnyConvertExtern { src, dst } => self.lower_preserved_result(
                preserved_op::ANY_CONVERT_EXTERN,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::ExternConvertAny { src, dst } => self.lower_preserved_result(
                preserved_op::EXTERN_CONVERT_ANY,
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
                    preserved_op::REF_TEST,
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
                    preserved_op::REF_CAST,
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
                preserved_op::STRUCT_NEW_DEFAULT,
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
                        "riscv64 backend received pair-valued struct.get",
                    ));
                }
                let op_code = match signed {
                    None => preserved_op::STRUCT_GET,
                    Some(true) => preserved_op::STRUCT_GET_S,
                    Some(false) => preserved_op::STRUCT_GET_U,
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
                        "riscv64 backend received pair-valued struct.set",
                    ));
                }
                self.lower_preserved_no_result(
                    preserved_op::STRUCT_SET,
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
                        "riscv64 backend received pair-valued array.new",
                    ));
                }
                self.lower_preserved_result(
                    preserved_op::ARRAY_NEW,
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
                preserved_op::ARRAY_NEW_DEFAULT,
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
                preserved_op::ARRAY_NEW_DATA,
                &[
                    (preserved_io::IMM0, *type_idx),
                    (preserved_io::IMM1, *data_idx),
                ],
                &[(preserved_io::ARG0, *src), (preserved_io::ARG1, *len)],
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
                preserved_op::ARRAY_NEW_ELEM,
                &[
                    (preserved_io::IMM0, *type_idx),
                    (preserved_io::IMM1, *elem_idx),
                ],
                &[(preserved_io::ARG0, *src), (preserved_io::ARG1, *len)],
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
                        "riscv64 backend received pair-valued array.get",
                    ));
                }
                let op_code = match signed {
                    None => preserved_op::ARRAY_GET,
                    Some(true) => preserved_op::ARRAY_GET_S,
                    Some(false) => preserved_op::ARRAY_GET_U,
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
                        "riscv64 backend received pair-valued array.set",
                    ));
                }
                self.lower_preserved_no_result(
                    preserved_op::ARRAY_SET,
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
                        "riscv64 backend received pair-valued array.fill",
                    ));
                }
                self.lower_preserved_no_result_extended(
                    preserved_op::ARRAY_FILL,
                    &[(preserved_io::IMM0, *type_idx)],
                    &[
                        (preserved_io::ARG0, *ref_src),
                        (preserved_io::ARG1, *index),
                        (preserved_io::ARG2, *value_lo),
                        (preserved_io::ARG3, *len),
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
                preserved_op::ARRAY_COPY,
                &[
                    (preserved_io::IMM0, *dst_type_idx),
                    (preserved_io::IMM1, *src_type_idx),
                ],
                &[
                    (preserved_io::ARG0, *dst_ref),
                    (preserved_io::ARG1, *dst_index),
                    (preserved_io::ARG2, *src_ref),
                    (preserved_io::ARG3, *src_index),
                    (preserved_io::ARG4, *len),
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
                preserved_op::ARRAY_INIT_DATA,
                &[
                    (preserved_io::IMM0, *type_idx),
                    (preserved_io::IMM1, *data_idx),
                ],
                &[
                    (preserved_io::ARG0, *ref_src),
                    (preserved_io::ARG1, *dst_index),
                    (preserved_io::ARG2, *src_index),
                    (preserved_io::ARG3, *len),
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
                preserved_op::ARRAY_INIT_ELEM,
                &[
                    (preserved_io::IMM0, *type_idx),
                    (preserved_io::IMM1, *elem_idx),
                ],
                &[
                    (preserved_io::ARG0, *ref_src),
                    (preserved_io::ARG1, *dst_index),
                    (preserved_io::ARG2, *src_index),
                    (preserved_io::ARG3, *len),
                ],
            ),
            MachineInstKind::ArrayLen { src, dst } => self.lower_preserved_result(
                preserved_op::ARRAY_LEN,
                0,
                0,
                *src,
                MachineValue::Imm64(0),
                MachineValue::Imm64(0),
                MachineStorageType::GpWord,
                *dst,
            ),
            MachineInstKind::Convert { op, dst, src } => self.lower_convert(*op, *dst, *src),
            _ => Err(Self::unsupported_error()),
        }
    }

    fn lower_source_move_dispatch(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        if let Some(width) = dst.ty.float_width() {
            let dst_fp = self.map_fp_reg(dst.reg)?;
            match src {
                ParallelSource::Reg { reg, .. } if self.core.is_fp_reg(reg) => {
                    let src_fp = self.map_fp_reg(reg)?;
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmv_s(dst_fp, src_fp),
                        MachineFloatWidth::F64 => enc::fmv_d(dst_fp, src_fp),
                    });
                }
                ParallelSource::Reg { reg, .. } => {
                    let src_gp = self.map_gp_reg(reg)?;
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmv_w_x(dst_fp, src_gp),
                        MachineFloatWidth::F64 => enc::fmv_d_x(dst_fp, src_gp),
                    });
                }
                ParallelSource::Imm(value) => {
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    self.materialize_u64(*scratch, value);
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmv_w_x(dst_fp, *scratch),
                        MachineFloatWidth::F64 => enc::fmv_d_x(dst_fp, *scratch),
                    });
                }
                ParallelSource::FpTemp(id, temp_width) => {
                    let temp = self.fp_scratch.reg(id);
                    if temp_width != width {
                        return Err(WasmError::invalid("riscv64 FP temp width mismatch"));
                    }
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmv_s(dst_fp, temp),
                        MachineFloatWidth::F64 => enc::fmv_d(dst_fp, temp),
                    });
                }
                ParallelSource::GpTemp(id) => {
                    let temp = self.gp_scratch.reg(id);
                    self.core.text.emit_u32(match width {
                        MachineFloatWidth::F32 => enc::fmv_w_x(dst_fp, temp),
                        MachineFloatWidth::F64 => enc::fmv_d_x(dst_fp, temp),
                    });
                }
                ParallelSource::ReservedReg(_) => {
                    return Err(WasmError::internal(
                        "riscv64 received non-identity reserved cache edge move",
                    ))
                }
            }
            self.core.set_fp_reg_width(dst.reg, width)?;
            return Ok(());
        }
        let dst_gp = self.map_gp_reg(dst.reg)?;
        match src {
            ParallelSource::Reg { reg, .. } => {
                if self.core.is_fp_reg(reg) {
                    let src_fp = self.map_fp_reg(reg)?;
                    self.move_fp_to_gp(dst_gp, src_fp, self.core.fp_reg_width(reg)?);
                } else {
                    let src = self.map_gp_reg(reg)?;
                    self.emit_mv(dst_gp, src);
                }
            }
            ParallelSource::ReservedReg(_) => {
                return Err(WasmError::internal(
                    "riscv64 received non-identity reserved cache edge move",
                ))
            }
            ParallelSource::Imm(value) => self.materialize_u64(dst_gp, value),
            ParallelSource::GpTemp(id) => self.emit_mv(dst_gp, self.gp_scratch.reg(id)),
            ParallelSource::FpTemp(id, width) => {
                self.move_fp_to_gp(dst_gp, self.fp_scratch.reg(id), width);
            }
        }
        Ok(())
    }

    fn lower_move(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        if let Some(width) = ty.float_width() {
            if self.core.is_fp_reg(dst) {
                let dst_fp = self.map_fp_reg(dst)?;
                self.load_fp_value_into(dst_fp, width, src)?;
                self.core.set_fp_reg_width(dst, width)?;
            } else {
                let tmp = self.fp_scratch.scoped_alloc().detach();
                self.load_fp_value_into(*tmp, width, src)?;
                let dst = self.map_gp_reg(dst)?;
                self.move_fp_to_gp(dst, *tmp, width);
            }
            return Ok(());
        }
        let dst = self.map_gp_reg(dst)?;
        self.load_value_into(dst, src)
    }

    fn lower_load(
        &mut self,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        if self.core.is_fp_reg(dst) {
            if !matches!(
                (width, extension),
                (
                    MachineMemWidth::U32,
                    MachineLoadExtension::None | MachineLoadExtension::ZeroExtend
                ) | (
                    MachineMemWidth::U64,
                    MachineLoadExtension::None | MachineLoadExtension::ZeroExtend
                )
            ) {
                return Err(Self::unsupported_error());
            }
            let dst_fp = self.map_fp_reg(dst)?;
            let base = self.map_gp_reg(addr.base)?;
            match width {
                MachineMemWidth::U32 => {
                    self.emit_fp_load_raw(0b010, dst_fp, base, addr.offset);
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                }
                MachineMemWidth::U64 => {
                    self.emit_fp_load_raw(0b011, dst_fp, base, addr.offset);
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                }
                _ => unreachable!(),
            }
            return Ok(());
        }
        let dst = self.map_gp_reg(dst)?;
        let base = self.map_gp_reg(addr.base)?;
        let funct3 = Self::load_funct3(width, extension)?;
        self.emit_load_raw(funct3, dst, base, addr.offset);
        Ok(())
    }

    fn lower_store(
        &mut self,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(addr.base)?;
        if let MachineValue::Reg(reg) = src {
            if self.core.is_fp_reg(reg) {
                let src_fp = self.map_fp_reg(reg)?;
                match width {
                    MachineMemWidth::U32 => {
                        self.emit_fp_store_raw(0b010, src_fp, base, addr.offset)
                    }
                    MachineMemWidth::U64 => {
                        self.emit_fp_store_raw(0b011, src_fp, base, addr.offset)
                    }
                    _ => return Err(Self::unsupported_error()),
                }
                return Ok(());
            }
        }
        let src_s = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*src_s, src)?;
        self.emit_store_raw(Self::store_funct3(width), *src_s, base, addr.offset);
        Ok(())
    }

    fn lower_indexed_load(
        &mut self,
        dst: MachineReg,
        base: MachineReg,
        index: MachineReg,
        index_extend: crate::vm::jit::machine::machine_ir::MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(base)?;
        let index = self.map_gp_reg(index)?;
        let addr = self.gp_scratch.scoped_alloc().detach();
        match index_extend {
            crate::vm::jit::machine::machine_ir::MachineIndexExtend::None => {
                self.emit_mv(*addr, index)
            }
            crate::vm::jit::machine::machine_ir::MachineIndexExtend::ZeroExtend32 => {
                self.zext_w(*addr, index)
            }
        }
        self.core.text.emit_u32(enc::add(*addr, base, *addr));
        if self.core.is_fp_reg(dst) {
            if !matches!(
                (width, extension),
                (
                    MachineMemWidth::U32,
                    MachineLoadExtension::None | MachineLoadExtension::ZeroExtend
                ) | (
                    MachineMemWidth::U64,
                    MachineLoadExtension::None | MachineLoadExtension::ZeroExtend
                )
            ) {
                return Err(Self::unsupported_error());
            }
            let dst_fp = self.map_fp_reg(dst)?;
            match width {
                MachineMemWidth::U32 => {
                    self.emit_fp_load_raw(0b010, dst_fp, *addr, offset);
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F32)?;
                }
                MachineMemWidth::U64 => {
                    self.emit_fp_load_raw(0b011, dst_fp, *addr, offset);
                    self.core.set_fp_reg_width(dst, MachineFloatWidth::F64)?;
                }
                _ => unreachable!(),
            }
        } else {
            let dst = self.map_gp_reg(dst)?;
            self.emit_load_raw(Self::load_funct3(width, extension)?, dst, *addr, offset);
        }
        Ok(())
    }

    fn lower_indexed_store(
        &mut self,
        base: MachineReg,
        index: MachineReg,
        index_extend: crate::vm::jit::machine::machine_ir::MachineIndexExtend,
        offset: i32,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let base = self.map_gp_reg(base)?;
        let index = self.map_gp_reg(index)?;
        let addr = self.gp_scratch.scoped_alloc().detach();
        match index_extend {
            crate::vm::jit::machine::machine_ir::MachineIndexExtend::None => {
                self.emit_mv(*addr, index)
            }
            crate::vm::jit::machine::machine_ir::MachineIndexExtend::ZeroExtend32 => {
                self.zext_w(*addr, index)
            }
        }
        self.core.text.emit_u32(enc::add(*addr, base, *addr));
        let store_offset = if Self::fits_i12(offset) {
            offset
        } else {
            {
                let offset_s = self.gp_scratch.scoped_alloc().detach();
                self.materialize_u64(*offset_s, offset as i64 as u64);
                self.core.text.emit_u32(enc::add(*addr, *addr, *offset_s));
            }
            0
        };
        if let MachineValue::Reg(reg) = src {
            if self.core.is_fp_reg(reg) {
                let src_fp = self.map_fp_reg(reg)?;
                match width {
                    MachineMemWidth::U32 => {
                        self.emit_fp_store_raw(0b010, src_fp, *addr, store_offset)
                    }
                    MachineMemWidth::U64 => {
                        self.emit_fp_store_raw(0b011, src_fp, *addr, store_offset)
                    }
                    _ => return Err(Self::unsupported_error()),
                }
                return Ok(());
            }
        }
        let src_s = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*src_s, src)?;
        self.emit_store_raw(Self::store_funct3(width), *src_s, *addr, store_offset);
        Ok(())
    }

    fn lower_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        self.load_value_into(dst, src)?;
        match (width, op) {
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend8S) => {
                self.core.text.emit_u32(enc::slli(dst, dst, 56));
                self.core.text.emit_u32(enc::srai(dst, dst, 56));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend16S) => {
                self.core.text.emit_u32(enc::slli(dst, dst, 48));
                self.core.text.emit_u32(enc::srai(dst, dst, 48));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntUnaryOp::Extend32S) => self.zext_w(dst, dst),
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend8S) => {
                self.core.text.emit_u32(enc::slli(dst, dst, 56));
                self.core.text.emit_u32(enc::srai(dst, dst, 56));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend16S) => {
                self.core.text.emit_u32(enc::slli(dst, dst, 48));
                self.core.text.emit_u32(enc::srai(dst, dst, 48));
            }
            (MachineIntWidth::I64, MachineIntUnaryOp::Extend32S) => self.sext_w(dst, dst),
            (_, MachineIntUnaryOp::Clz) => self.emit_clz(width, dst),
            (_, MachineIntUnaryOp::Ctz) => self.emit_ctz(width, dst),
            (_, MachineIntUnaryOp::Popcnt) => self.emit_popcnt(width, dst),
        }
        Ok(())
    }

    fn emit_clz(&mut self, width: MachineIntWidth, dst: RiscvReg) {
        let value = self.gp_scratch.scoped_alloc().detach();
        let probe = self.gp_scratch.scoped_alloc().detach();
        self.emit_mv(*value, dst);
        if width == MachineIntWidth::I32 {
            self.zext_w(*value, *value);
        }
        let zero = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch_to(enc::Cond::Eq, *value, abi::zero_reg(), zero);
        self.emit_addi(dst, abi::zero_reg(), 0);
        let total_bits = match width {
            MachineIntWidth::I32 => 32,
            MachineIntWidth::I64 => 64,
        };
        let steps: &[u32] = match width {
            MachineIntWidth::I32 => &[16, 8, 4, 2, 1],
            MachineIntWidth::I64 => &[32, 16, 8, 4, 2, 1],
        };
        for &step in steps {
            let nonzero = self.core.new_label();
            self.core
                .text
                .emit_u32(enc::srli(*probe, *value, total_bits - step));
            self.emit_branch_to(enc::Cond::Ne, *probe, abi::zero_reg(), nonzero);
            self.emit_addi(dst, dst, step as i32);
            self.core.text.emit_u32(enc::slli(*value, *value, step));
            self.core.bind_label(nonzero);
        }
        self.emit_jal(abi::zero_reg(), done);
        self.core.bind_label(zero);
        self.emit_addi(dst, abi::zero_reg(), total_bits as i32);
        self.core.bind_label(done);
    }

    fn emit_ctz(&mut self, width: MachineIntWidth, dst: RiscvReg) {
        let value = self.gp_scratch.scoped_alloc().detach();
        let probe = self.gp_scratch.scoped_alloc().detach();
        self.emit_mv(*value, dst);
        if width == MachineIntWidth::I32 {
            self.zext_w(*value, *value);
        }
        let zero = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch_to(enc::Cond::Eq, *value, abi::zero_reg(), zero);
        self.emit_addi(dst, abi::zero_reg(), 0);
        let total_bits = match width {
            MachineIntWidth::I32 => 32,
            MachineIntWidth::I64 => 64,
        };
        let steps: &[u32] = match width {
            MachineIntWidth::I32 => &[16, 8, 4, 2, 1],
            MachineIntWidth::I64 => &[32, 16, 8, 4, 2, 1],
        };
        for &step in steps {
            let has_low_bits = self.core.new_label();
            self.materialize_u64(*probe, (1u64 << step) - 1);
            self.core.text.emit_u32(enc::and(*probe, *value, *probe));
            self.emit_branch_to(enc::Cond::Ne, *probe, abi::zero_reg(), has_low_bits);
            self.emit_addi(dst, dst, step as i32);
            self.core.text.emit_u32(enc::srli(*value, *value, step));
            self.core.bind_label(has_low_bits);
        }
        self.emit_jal(abi::zero_reg(), done);
        self.core.bind_label(zero);
        self.emit_addi(dst, abi::zero_reg(), total_bits as i32);
        self.core.bind_label(done);
    }

    fn emit_popcnt(&mut self, width: MachineIntWidth, dst: RiscvReg) {
        let tmp = self.gp_scratch.scoped_alloc().detach();
        let mask = self.gp_scratch.scoped_alloc().detach();
        if width == MachineIntWidth::I32 {
            self.zext_w(dst, dst);
        }
        self.core.text.emit_u32(enc::srli(*tmp, dst, 1));
        self.materialize_u64(*mask, 0x5555_5555_5555_5555);
        self.core.text.emit_u32(enc::and(*tmp, *tmp, *mask));
        self.core.text.emit_u32(enc::sub(dst, dst, *tmp));

        self.materialize_u64(*mask, 0x3333_3333_3333_3333);
        self.core.text.emit_u32(enc::and(*tmp, dst, *mask));
        self.core.text.emit_u32(enc::srli(dst, dst, 2));
        self.core.text.emit_u32(enc::and(dst, dst, *mask));
        self.core.text.emit_u32(enc::add(dst, dst, *tmp));

        self.core.text.emit_u32(enc::srli(*tmp, dst, 4));
        self.core.text.emit_u32(enc::add(dst, dst, *tmp));
        self.materialize_u64(*mask, 0x0f0f_0f0f_0f0f_0f0f);
        self.core.text.emit_u32(enc::and(dst, dst, *mask));

        self.core.text.emit_u32(enc::srli(*tmp, dst, 8));
        self.core.text.emit_u32(enc::add(dst, dst, *tmp));
        self.core.text.emit_u32(enc::srli(*tmp, dst, 16));
        self.core.text.emit_u32(enc::add(dst, dst, *tmp));
        if width == MachineIntWidth::I64 {
            self.core.text.emit_u32(enc::srli(*tmp, dst, 32));
            self.core.text.emit_u32(enc::add(dst, dst, *tmp));
            self.materialize_u64(*mask, 0x7f);
        } else {
            self.materialize_u64(*mask, 0x3f);
        }
        self.core.text.emit_u32(enc::and(dst, dst, *mask));
    }

    fn emit_int_binary_regs(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: RiscvReg,
        lhs: RiscvReg,
        rhs: RiscvReg,
    ) -> Result<(), WasmError> {
        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::Add) => {
                self.core.text.emit_u32(enc::add(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Sub) => {
                self.core.text.emit_u32(enc::sub(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Mul) => {
                self.core.text.emit_u32(enc::mul(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::And) => {
                self.core.text.emit_u32(enc::and(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Or) => {
                self.core.text.emit_u32(enc::or(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Xor) => {
                self.core.text.emit_u32(enc::xor(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::Shl) => {
                self.core.text.emit_u32(enc::sll(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrU) => {
                self.core.text.emit_u32(enc::srl(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::ShrS) => {
                self.core.text.emit_u32(enc::sra(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Add) => {
                self.core.text.emit_u32(enc::addw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Sub) => {
                self.core.text.emit_u32(enc::subw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Mul) => {
                self.core.text.emit_u32(enc::mulw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::And) => {
                self.core.text.emit_u32(enc::and(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Or) => {
                self.core.text.emit_u32(enc::or(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Xor) => {
                self.core.text.emit_u32(enc::xor(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::Shl) => {
                self.core.text.emit_u32(enc::sllw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrU) => {
                self.core.text.emit_u32(enc::srlw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::ShrS) => {
                self.core.text.emit_u32(enc::sraw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            _ => return Err(Self::unsupported_error()),
        }
        Ok(())
    }

    fn lower_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let lhs_s = self.gp_scratch.scoped_alloc().detach();
        let rhs_s = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*lhs_s, lhs)?;
        self.load_value_into(*rhs_s, rhs)?;
        match op {
            MachineIntBinaryOp::DivS
            | MachineIntBinaryOp::DivU
            | MachineIntBinaryOp::RemS
            | MachineIntBinaryOp::RemU => self.lower_div_rem_regs(width, op, dst, *lhs_s, *rhs_s),
            MachineIntBinaryOp::Rotl | MachineIntBinaryOp::Rotr => {
                match (width, op) {
                    (MachineIntWidth::I64, MachineIntBinaryOp::Rotl) => {
                        self.core.text.emit_u32(enc::sll(dst, *lhs_s, *rhs_s));
                        self.core
                            .text
                            .emit_u32(enc::sub(*rhs_s, abi::zero_reg(), *rhs_s));
                        self.core.text.emit_u32(enc::srl(*rhs_s, *lhs_s, *rhs_s));
                    }
                    (MachineIntWidth::I64, MachineIntBinaryOp::Rotr) => {
                        self.core.text.emit_u32(enc::srl(dst, *lhs_s, *rhs_s));
                        self.core
                            .text
                            .emit_u32(enc::sub(*rhs_s, abi::zero_reg(), *rhs_s));
                        self.core.text.emit_u32(enc::sll(*rhs_s, *lhs_s, *rhs_s));
                    }
                    (MachineIntWidth::I32, MachineIntBinaryOp::Rotl) => {
                        self.core.text.emit_u32(enc::sllw(dst, *lhs_s, *rhs_s));
                        self.core
                            .text
                            .emit_u32(enc::sub(*rhs_s, abi::zero_reg(), *rhs_s));
                        self.core.text.emit_u32(enc::srlw(*rhs_s, *lhs_s, *rhs_s));
                    }
                    (MachineIntWidth::I32, MachineIntBinaryOp::Rotr) => {
                        self.core.text.emit_u32(enc::srlw(dst, *lhs_s, *rhs_s));
                        self.core
                            .text
                            .emit_u32(enc::sub(*rhs_s, abi::zero_reg(), *rhs_s));
                        self.core.text.emit_u32(enc::sllw(*rhs_s, *lhs_s, *rhs_s));
                    }
                    _ => unreachable!(),
                }
                self.core.text.emit_u32(enc::or(dst, dst, *rhs_s));
                if width == MachineIntWidth::I32 {
                    self.zext_w(dst, dst);
                }
                Ok(())
            }
            _ => self.emit_int_binary_regs(width, op, dst, *lhs_s, *rhs_s),
        }
    }

    fn lower_div_rem_regs(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: RiscvReg,
        lhs: RiscvReg,
        rhs: RiscvReg,
    ) -> Result<(), WasmError> {
        let sign = match op {
            MachineIntBinaryOp::DivS | MachineIntBinaryOp::RemS => MachineSign::Signed,
            MachineIntBinaryOp::DivU | MachineIntBinaryOp::RemU => MachineSign::Unsigned,
            _ => unreachable!(),
        };
        self.canonicalize_compare_operand(lhs, width, sign);
        self.canonicalize_compare_operand(rhs, width, sign);

        let div_zero_label = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerDivideByZero);
        self.emit_branch_to(enc::Cond::Eq, rhs, abi::zero_reg(), div_zero_label);

        if matches!(op, MachineIntBinaryOp::DivS | MachineIntBinaryOp::RemS) {
            let normal = self.core.new_label();
            let done = self.core.new_label();
            let min = match width {
                MachineIntWidth::I32 => i32::MIN as i64 as u64,
                MachineIntWidth::I64 => i64::MIN as u64,
            };
            self.materialize_u64(dst, min);
            self.emit_branch_to(enc::Cond::Ne, lhs, dst, normal);
            self.materialize_u64(dst, (-1i64) as u64);
            self.emit_branch_to(enc::Cond::Ne, rhs, dst, normal);
            if op == MachineIntBinaryOp::DivS {
                let overflow_label = self
                    .core
                    .ensure_trap_label(MachineTrapKind::IntegerOverflow);
                self.emit_jal(abi::zero_reg(), overflow_label);
            } else {
                self.emit_addi(dst, abi::zero_reg(), 0);
                self.emit_jal(abi::zero_reg(), done);
            }
            self.core.bind_label(normal);
            match (width, op) {
                (MachineIntWidth::I64, MachineIntBinaryOp::DivS) => {
                    self.core.text.emit_u32(enc::div(dst, lhs, rhs));
                }
                (MachineIntWidth::I64, MachineIntBinaryOp::RemS) => {
                    self.core.text.emit_u32(enc::rem(dst, lhs, rhs));
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::DivS) => {
                    self.core.text.emit_u32(enc::divw(dst, lhs, rhs));
                    self.zext_w(dst, dst);
                }
                (MachineIntWidth::I32, MachineIntBinaryOp::RemS) => {
                    self.core.text.emit_u32(enc::remw(dst, lhs, rhs));
                    self.zext_w(dst, dst);
                }
                _ => unreachable!(),
            };
            self.core.bind_label(done);
            return Ok(());
        }

        match (width, op) {
            (MachineIntWidth::I64, MachineIntBinaryOp::DivU) => {
                self.core.text.emit_u32(enc::divu(dst, lhs, rhs));
            }
            (MachineIntWidth::I64, MachineIntBinaryOp::RemU) => {
                self.core.text.emit_u32(enc::remu(dst, lhs, rhs));
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::DivU) => {
                self.core.text.emit_u32(enc::divuw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            (MachineIntWidth::I32, MachineIntBinaryOp::RemU) => {
                self.core.text.emit_u32(enc::remuw(dst, lhs, rhs));
                self.zext_w(dst, dst);
            }
            _ => unreachable!(),
        };
        Ok(())
    }

    fn lower_int_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let lhs_s = self.gp_scratch.scoped_alloc().detach();
        let rhs_s = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*lhs_s, lhs)?;
        self.load_value_into(*rhs_s, rhs)?;
        self.canonicalize_compare_operand(*lhs_s, width, sign);
        self.canonicalize_compare_operand(*rhs_s, width, sign);
        let (cond, swap) = Self::branch_cond_for_compare(kind, sign);
        let (lhs, rhs) = if swap {
            (*rhs_s, *lhs_s)
        } else {
            (*lhs_s, *rhs_s)
        };
        match cond {
            enc::Cond::Eq => {
                self.core.text.emit_u32(enc::xor(dst, lhs, rhs));
                self.core.text.emit_u32(enc::sltiu(dst, dst, 1));
            }
            enc::Cond::Ne => {
                self.core.text.emit_u32(enc::xor(dst, lhs, rhs));
                self.core
                    .text
                    .emit_u32(enc::sltu(dst, abi::zero_reg(), dst));
            }
            enc::Cond::Lt => {
                self.core.text.emit_u32(enc::slt(dst, lhs, rhs));
            }
            enc::Cond::Ltu => {
                self.core.text.emit_u32(enc::sltu(dst, lhs, rhs));
            }
            enc::Cond::Ge => {
                self.core.text.emit_u32(enc::slt(dst, lhs, rhs));
                self.core.text.emit_u32(enc::xori(dst, dst, 1));
            }
            enc::Cond::Geu => {
                self.core.text.emit_u32(enc::sltu(dst, lhs, rhs));
                self.core.text.emit_u32(enc::xori(dst, dst, 1));
            }
        }
        Ok(())
    }

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
        self.emit_mv(dst, src);
        if width == MachineIntWidth::I32 {
            self.zext_w(dst, dst);
        }
        if lsb > 0 {
            self.core.text.emit_u32(enc::srli(dst, dst, u32::from(lsb)));
        }
        let mask = if bits as u32 >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let scratch = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*scratch, mask);
        self.core.text.emit_u32(enc::and(dst, dst, *scratch));
        if width == MachineIntWidth::I32 {
            self.zext_w(dst, dst);
        }
        Ok(())
    }

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
        let dst = self.map_gp_reg(dst)?;
        let lhs = self.map_gp_reg(lhs)?;
        let rhs = self.map_gp_reg(rhs)?;
        let shifted = self.gp_scratch.scoped_alloc().detach();
        self.emit_mv(*shifted, rhs);
        match (width, shift) {
            (MachineIntWidth::I64, MachineShiftOp::Lsl) => {
                self.core
                    .text
                    .emit_u32(enc::slli(*shifted, *shifted, u32::from(amount)));
            }
            (MachineIntWidth::I64, MachineShiftOp::Lsr) => {
                self.core
                    .text
                    .emit_u32(enc::srli(*shifted, *shifted, u32::from(amount)));
            }
            (MachineIntWidth::I64, MachineShiftOp::Asr) => {
                self.core
                    .text
                    .emit_u32(enc::srai(*shifted, *shifted, u32::from(amount)));
            }
            (MachineIntWidth::I32, MachineShiftOp::Lsl) => {
                self.materialize_u64(*shifted, u64::from(amount));
                self.core.text.emit_u32(enc::sllw(*shifted, rhs, *shifted));
            }
            (MachineIntWidth::I32, MachineShiftOp::Lsr) => {
                self.materialize_u64(*shifted, u64::from(amount));
                self.core.text.emit_u32(enc::srlw(*shifted, rhs, *shifted));
            }
            (MachineIntWidth::I32, MachineShiftOp::Asr) => {
                self.materialize_u64(*shifted, u64::from(amount));
                self.core.text.emit_u32(enc::sraw(*shifted, rhs, *shifted));
            }
            (MachineIntWidth::I64, MachineShiftOp::Ror) => {
                if amount != 0 {
                    let other = self.gp_scratch.scoped_alloc().detach();
                    self.core
                        .text
                        .emit_u32(enc::srli(*shifted, rhs, u32::from(amount)));
                    self.core
                        .text
                        .emit_u32(enc::slli(*other, rhs, 64_u32 - u32::from(amount)));
                    self.core.text.emit_u32(enc::or(*shifted, *shifted, *other));
                }
            }
            (MachineIntWidth::I32, MachineShiftOp::Ror) => {
                if amount != 0 {
                    let other = self.gp_scratch.scoped_alloc().detach();
                    self.materialize_u64(*shifted, u64::from(amount));
                    self.core.text.emit_u32(enc::srlw(*shifted, rhs, *shifted));
                    self.materialize_u64(*other, u64::from(32 - amount));
                    self.core.text.emit_u32(enc::sllw(*other, rhs, *other));
                    self.core.text.emit_u32(enc::or(*shifted, *shifted, *other));
                    self.zext_w(*shifted, *shifted);
                }
            }
        };
        self.emit_int_binary_regs(width, op, dst, lhs, *shifted)
    }

    fn lower_test_bits(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        src: MachineReg,
        mask: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let src_s = self.gp_scratch.scoped_alloc().detach();
        let mask_s = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*src_s, MachineValue::Reg(src))?;
        self.load_value_into(*mask_s, mask)?;
        if width == MachineIntWidth::I32 {
            self.zext_w(*src_s, *src_s);
            self.zext_w(*mask_s, *mask_s);
        }
        self.core.text.emit_u32(enc::and(dst, *src_s, *mask_s));
        match kind {
            MachineCompareKind::Eq => {
                self.core.text.emit_u32(enc::sltiu(dst, dst, 1));
            }
            MachineCompareKind::Ne => {
                self.core
                    .text
                    .emit_u32(enc::sltu(dst, abi::zero_reg(), dst));
            }
            _ => {
                return Err(WasmError::internal(
                    "riscv64 TestBits uses unsupported compare kind",
                ))
            }
        }
        Ok(())
    }

    fn lower_select(
        &mut self,
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
        cond: MachineValue,
    ) -> Result<(), WasmError> {
        if let MachineValue::Imm64(value) = cond {
            return self.lower_move(ty, dst, if value != 0 { on_true } else { on_false });
        }
        if ty.float_width().is_some() {
            let cond_s = self.gp_scratch.scoped_alloc().detach();
            self.load_value_into(*cond_s, cond)?;
            self.zext_w(*cond_s, *cond_s);
            let false_path = self.core.new_label();
            let done = self.core.new_label();
            self.emit_branch_to(enc::Cond::Eq, *cond_s, abi::zero_reg(), false_path);
            drop(cond_s);
            self.lower_move(ty, dst, on_true)?;
            self.emit_jal(abi::zero_reg(), done);
            self.core.bind_label(false_path);
            self.lower_move(ty, dst, on_false)?;
            self.core.bind_label(done);
            return Ok(());
        }
        let cond_s = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*cond_s, cond)?;
        self.zext_w(*cond_s, *cond_s);
        let false_path = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch_to(enc::Cond::Eq, *cond_s, abi::zero_reg(), false_path);
        drop(cond_s);
        self.lower_move(ty, dst, on_true)?;
        self.emit_jal(abi::zero_reg(), done);
        self.core.bind_label(false_path);
        self.lower_move(ty, dst, on_false)?;
        self.core.bind_label(done);
        Ok(())
    }

    fn lower_float_const(
        &mut self,
        width: MachineFloatWidth,
        dst: MachineReg,
        bits: u64,
    ) -> Result<(), WasmError> {
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.load_fp_value_into(dst_fp, width, MachineValue::Imm64(bits))?;
            self.core.set_fp_reg_width(dst, width)?;
        } else {
            let dst_gp = self.map_gp_reg(dst)?;
            self.materialize_u64(dst_gp, bits);
        }
        Ok(())
    }

    fn lower_float_unary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let src_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*src_fp, width, src)?;
        let needs_distinct_result = matches!(
            op,
            MachineFloatUnaryOp::Ceil
                | MachineFloatUnaryOp::Floor
                | MachineFloatUnaryOp::Trunc
                | MachineFloatUnaryOp::Nearest
        );
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else if needs_distinct_result {
            *self.fp_scratch.scoped_alloc()
        } else {
            *src_fp
        };
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
                self.core.text.emit_u32(enc::fabs_s(result_fp, *src_fp));
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
                self.core.text.emit_u32(enc::fabs_d(result_fp, *src_fp));
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
                self.core.text.emit_u32(enc::fneg_s(result_fp, *src_fp));
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
                self.core.text.emit_u32(enc::fneg_d(result_fp, *src_fp));
            }
            (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
                self.core.text.emit_u32(enc::fsqrt_s(result_fp, *src_fp));
            }
            (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
                self.core.text.emit_u32(enc::fsqrt_d(result_fp, *src_fp));
            }
            (_, MachineFloatUnaryOp::Ceil)
            | (_, MachineFloatUnaryOp::Floor)
            | (_, MachineFloatUnaryOp::Trunc)
            | (_, MachineFloatUnaryOp::Nearest) => {
                self.emit_float_round(width, op, result_fp, *src_fp)?;
            }
        }
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            self.move_fp_to_gp(dst_gp, result_fp, width);
        }
        Ok(())
    }

    fn lower_float_binary(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let lhs_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*lhs_fp, width, lhs)?;
        let rhs_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*rhs_fp, width, rhs)?;
        let result_fp = if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.set_fp_reg_width(dst, width)?;
            dst_fp
        } else {
            *lhs_fp
        };
        match (width, op) {
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
                self.core
                    .text
                    .emit_u32(enc::fadd_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
                self.core
                    .text
                    .emit_u32(enc::fadd_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
                self.core
                    .text
                    .emit_u32(enc::fsub_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
                self.core
                    .text
                    .emit_u32(enc::fsub_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
                self.core
                    .text
                    .emit_u32(enc::fmul_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
                self.core
                    .text
                    .emit_u32(enc::fmul_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
                self.core
                    .text
                    .emit_u32(enc::fdiv_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
                self.core
                    .text
                    .emit_u32(enc::fdiv_d(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
                self.emit_float_min_max_patch(width, true, result_fp, *lhs_fp, *rhs_fp)
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
                self.emit_float_min_max_patch(width, true, result_fp, *lhs_fp, *rhs_fp)
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
                self.emit_float_min_max_patch(width, false, result_fp, *lhs_fp, *rhs_fp)
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
                self.emit_float_min_max_patch(width, false, result_fp, *lhs_fp, *rhs_fp)
            }
            (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
                self.core
                    .text
                    .emit_u32(enc::fsgnj_s(result_fp, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
                self.core
                    .text
                    .emit_u32(enc::fsgnj_d(result_fp, *lhs_fp, *rhs_fp));
            }
        }
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            self.move_fp_to_gp(dst_gp, result_fp, width);
        }
        Ok(())
    }

    fn emit_float_min_max_patch(
        &mut self,
        width: MachineFloatWidth,
        is_min: bool,
        result_fp: RiscvFpReg,
        lhs_fp: RiscvFpReg,
        rhs_fp: RiscvFpReg,
    ) {
        let ordered = self.gp_scratch.scoped_alloc().detach();
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::feq_s(*ordered, lhs_fp, lhs_fp),
            MachineFloatWidth::F64 => enc::feq_d(*ordered, lhs_fp, lhs_fp),
        });
        let nan = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch_to(enc::Cond::Eq, *ordered, abi::zero_reg(), nan);
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::feq_s(*ordered, rhs_fp, rhs_fp),
            MachineFloatWidth::F64 => enc::feq_d(*ordered, rhs_fp, rhs_fp),
        });
        self.emit_branch_to(enc::Cond::Eq, *ordered, abi::zero_reg(), nan);
        self.core.text.emit_u32(match (width, is_min) {
            (MachineFloatWidth::F32, true) => enc::fmin_s(result_fp, lhs_fp, rhs_fp),
            (MachineFloatWidth::F64, true) => enc::fmin_d(result_fp, lhs_fp, rhs_fp),
            (MachineFloatWidth::F32, false) => enc::fmax_s(result_fp, lhs_fp, rhs_fp),
            (MachineFloatWidth::F64, false) => enc::fmax_d(result_fp, lhs_fp, rhs_fp),
        });
        self.emit_jal(abi::zero_reg(), done);
        self.core.bind_label(nan);
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fadd_s(result_fp, lhs_fp, rhs_fp),
            MachineFloatWidth::F64 => enc::fadd_d(result_fp, lhs_fp, rhs_fp),
        });
        self.core.bind_label(done);
    }

    fn lower_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let lhs_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*lhs_fp, width, lhs)?;
        let rhs_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*rhs_fp, width, rhs)?;
        match (width, kind) {
            (MachineFloatWidth::F32, MachineCompareKind::Eq) => {
                self.core.text.emit_u32(enc::feq_s(dst, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineCompareKind::Eq) => {
                self.core.text.emit_u32(enc::feq_d(dst, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineCompareKind::Ne) => {
                self.core.text.emit_u32(enc::feq_s(dst, *lhs_fp, *rhs_fp));
                self.core.text.emit_u32(enc::xori(dst, dst, 1));
            }
            (MachineFloatWidth::F64, MachineCompareKind::Ne) => {
                self.core.text.emit_u32(enc::feq_d(dst, *lhs_fp, *rhs_fp));
                self.core.text.emit_u32(enc::xori(dst, dst, 1));
            }
            (MachineFloatWidth::F32, MachineCompareKind::Lt) => {
                self.core.text.emit_u32(enc::flt_s(dst, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineCompareKind::Lt) => {
                self.core.text.emit_u32(enc::flt_d(dst, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineCompareKind::Gt) => {
                self.core.text.emit_u32(enc::flt_s(dst, *rhs_fp, *lhs_fp));
            }
            (MachineFloatWidth::F64, MachineCompareKind::Gt) => {
                self.core.text.emit_u32(enc::flt_d(dst, *rhs_fp, *lhs_fp));
            }
            (MachineFloatWidth::F32, MachineCompareKind::Le) => {
                self.core.text.emit_u32(enc::fle_s(dst, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F64, MachineCompareKind::Le) => {
                self.core.text.emit_u32(enc::fle_d(dst, *lhs_fp, *rhs_fp));
            }
            (MachineFloatWidth::F32, MachineCompareKind::Ge) => {
                self.core.text.emit_u32(enc::fle_s(dst, *rhs_fp, *lhs_fp));
            }
            (MachineFloatWidth::F64, MachineCompareKind::Ge) => {
                self.core.text.emit_u32(enc::fle_d(dst, *rhs_fp, *lhs_fp));
            }
        }
        Ok(())
    }

    fn lower_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_float_width = convert_result_float_width(op);
        match op {
            MachineConvertOp::I32WrapI64 | MachineConvertOp::I64ExtendI32U => {
                let dst = self.map_gp_reg(dst)?;
                let src_s = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*src_s, src)?;
                self.zext_w(dst, *src_s);
            }
            MachineConvertOp::I64ExtendI32S => {
                let dst = self.map_gp_reg(dst)?;
                let src_s = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*src_s, src)?;
                self.sext_w(dst, *src_s);
            }
            MachineConvertOp::I32ReinterpretF32 | MachineConvertOp::I64ReinterpretF64 => {
                let dst = self.map_gp_reg(dst)?;
                self.load_value_into(dst, src)?;
                if matches!(op, MachineConvertOp::I32ReinterpretF32) {
                    self.zext_w(dst, dst);
                }
            }
            MachineConvertOp::F32ReinterpretI32 | MachineConvertOp::F64ReinterpretI64 => {
                let width = dst_float_width.expect("float reinterpret width");
                self.lower_move(
                    match width {
                        MachineFloatWidth::F32 => MachineStorageType::Fp32,
                        MachineFloatWidth::F64 => MachineStorageType::Fp64,
                    },
                    dst,
                    src,
                )?;
            }
            MachineConvertOp::F64PromoteF32 => {
                let src_fp = self.fp_scratch.scoped_alloc().detach();
                self.load_fp_value_into(*src_fp, MachineFloatWidth::F32, src)?;
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F64)?;
                self.core.text.emit_u32(enc::fcvt_d_s(dst_fp, *src_fp));
                self.finish_convert_fp_dst(dst, dst_fp, MachineFloatWidth::F64)?;
            }
            MachineConvertOp::F32DemoteF64 => {
                let src_fp = self.fp_scratch.scoped_alloc().detach();
                self.load_fp_value_into(*src_fp, MachineFloatWidth::F64, src)?;
                let dst_fp = self.resolve_convert_fp_dst(dst, MachineFloatWidth::F32)?;
                self.core.text.emit_u32(enc::fcvt_s_d(dst_fp, *src_fp));
                self.finish_convert_fp_dst(dst, dst_fp, MachineFloatWidth::F32)?;
            }
            MachineConvertOp::F32ConvertI32S
            | MachineConvertOp::F32ConvertI32U
            | MachineConvertOp::F32ConvertI64S
            | MachineConvertOp::F32ConvertI64U
            | MachineConvertOp::F64ConvertI32S
            | MachineConvertOp::F64ConvertI32U
            | MachineConvertOp::F64ConvertI64S
            | MachineConvertOp::F64ConvertI64U => {
                let src_gp = self.gp_scratch.scoped_alloc().detach();
                self.load_value_into(*src_gp, src)?;
                let width = dst_float_width.expect("int-to-float result width");
                let dst_fp = self.resolve_convert_fp_dst(dst, width)?;
                self.core.text.emit_u32(match op {
                    MachineConvertOp::F32ConvertI32S => enc::fcvt_s_w(dst_fp, *src_gp),
                    MachineConvertOp::F32ConvertI32U => enc::fcvt_s_wu(dst_fp, *src_gp),
                    MachineConvertOp::F32ConvertI64S => enc::fcvt_s_l(dst_fp, *src_gp),
                    MachineConvertOp::F32ConvertI64U => enc::fcvt_s_lu(dst_fp, *src_gp),
                    MachineConvertOp::F64ConvertI32S => enc::fcvt_d_w(dst_fp, *src_gp),
                    MachineConvertOp::F64ConvertI32U => enc::fcvt_d_wu(dst_fp, *src_gp),
                    MachineConvertOp::F64ConvertI64S => enc::fcvt_d_l(dst_fp, *src_gp),
                    MachineConvertOp::F64ConvertI64U => enc::fcvt_d_lu(dst_fp, *src_gp),
                    _ => unreachable!(),
                });
                self.finish_convert_fp_dst(dst, dst_fp, width)?;
            }
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => {
                self.lower_saturating_trunc(op, dst, src)?;
            }
            MachineConvertOp::I32TruncF32S
            | MachineConvertOp::I32TruncF32U
            | MachineConvertOp::I32TruncF64S
            | MachineConvertOp::I32TruncF64U
            | MachineConvertOp::I64TruncF32S
            | MachineConvertOp::I64TruncF32U
            | MachineConvertOp::I64TruncF64S
            | MachineConvertOp::I64TruncF64U => {
                self.lower_trapping_trunc(op, dst, src)?;
            }
        }
        Ok(())
    }

    fn resolve_convert_fp_dst(
        &mut self,
        dst: MachineReg,
        width: MachineFloatWidth,
    ) -> Result<RiscvFpReg, WasmError> {
        if self.core.is_fp_reg(dst) {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.set_fp_reg_width(dst, width)?;
            Ok(dst_fp)
        } else {
            Ok(*self.fp_scratch.scoped_alloc())
        }
    }

    fn finish_convert_fp_dst(
        &mut self,
        dst: MachineReg,
        dst_fp: RiscvFpReg,
        width: MachineFloatWidth,
    ) -> Result<(), WasmError> {
        if !self.core.is_fp_reg(dst) {
            let dst_gp = self.map_gp_reg(dst)?;
            self.move_fp_to_gp(dst_gp, dst_fp, width);
        }
        Ok(())
    }

    fn lower_saturating_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let width = match op {
            MachineConvertOp::I32TruncSatF32S
            | MachineConvertOp::I32TruncSatF32U
            | MachineConvertOp::I64TruncSatF32S
            | MachineConvertOp::I64TruncSatF32U => MachineFloatWidth::F32,
            MachineConvertOp::I32TruncSatF64S
            | MachineConvertOp::I32TruncSatF64U
            | MachineConvertOp::I64TruncSatF64S
            | MachineConvertOp::I64TruncSatF64U => MachineFloatWidth::F64,
            _ => unreachable!(),
        };
        let src_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*src_fp, width, src)?;
        let ordered = self.gp_scratch.scoped_alloc().detach();
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::feq_s(*ordered, *src_fp, *src_fp),
            MachineFloatWidth::F64 => enc::feq_d(*ordered, *src_fp, *src_fp),
        });
        let convert = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch_to(enc::Cond::Ne, *ordered, abi::zero_reg(), convert);
        self.emit_addi(dst, abi::zero_reg(), 0);
        self.emit_jal(abi::zero_reg(), done);
        self.core.bind_label(convert);
        self.core.text.emit_u32(match op {
            MachineConvertOp::I32TruncSatF32S => enc::fcvt_w_s_rtz(dst, *src_fp),
            MachineConvertOp::I32TruncSatF32U => enc::fcvt_wu_s_rtz(dst, *src_fp),
            MachineConvertOp::I32TruncSatF64S => enc::fcvt_w_d_rtz(dst, *src_fp),
            MachineConvertOp::I32TruncSatF64U => enc::fcvt_wu_d_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncSatF32S => enc::fcvt_l_s_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncSatF32U => enc::fcvt_lu_s_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncSatF64S => enc::fcvt_l_d_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncSatF64U => enc::fcvt_lu_d_rtz(dst, *src_fp),
            _ => unreachable!(),
        });
        if matches!(
            op,
            MachineConvertOp::I32TruncSatF32S
                | MachineConvertOp::I32TruncSatF32U
                | MachineConvertOp::I32TruncSatF64S
                | MachineConvertOp::I32TruncSatF64U
        ) {
            self.zext_w(dst, dst);
        }
        self.core.bind_label(done);
        Ok(())
    }

    fn emit_float_round(
        &mut self,
        width: MachineFloatWidth,
        op: MachineFloatUnaryOp,
        result_fp: RiscvFpReg,
        src_fp: RiscvFpReg,
    ) -> Result<(), WasmError> {
        let bits = self.gp_scratch.scoped_alloc().detach();
        self.move_fp_to_gp(*bits, src_fp, width);
        let tmp = self.gp_scratch.scoped_alloc().detach();
        let threshold = match width {
            MachineFloatWidth::F32 => 0x4b00_0000_u64, // 2^23
            MachineFloatWidth::F64 => 0x4330_0000_0000_0000_u64, // 2^52
        };
        let abs_mask = match width {
            MachineFloatWidth::F32 => 0x7fff_ffff_u64,
            MachineFloatWidth::F64 => 0x7fff_ffff_ffff_ffff_u64,
        };
        self.materialize_u64(*tmp, abs_mask);
        self.core.text.emit_u32(enc::and(*bits, *bits, *tmp));

        let return_src = self.core.new_label();
        let done = self.core.new_label();
        self.emit_branch_to(enc::Cond::Eq, *bits, abi::zero_reg(), return_src);
        self.materialize_u64(*tmp, threshold);
        self.emit_branch_to(enc::Cond::Geu, *bits, *tmp, return_src);

        let rm = match op {
            MachineFloatUnaryOp::Ceil => enc::ROUND_RUP,
            MachineFloatUnaryOp::Floor => enc::ROUND_RDN,
            MachineFloatUnaryOp::Trunc => enc::ROUND_RTZ,
            MachineFloatUnaryOp::Nearest => enc::ROUND_RNE,
            _ => unreachable!(),
        };
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fcvt_l_s_round(*tmp, src_fp, rm),
            MachineFloatWidth::F64 => enc::fcvt_l_d_round(*tmp, src_fp, rm),
        });
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fcvt_s_l(result_fp, *tmp),
            MachineFloatWidth::F64 => enc::fcvt_d_l(result_fp, *tmp),
        });
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fsgnj_s(result_fp, result_fp, src_fp),
            MachineFloatWidth::F64 => enc::fsgnj_d(result_fp, result_fp, src_fp),
        });
        self.emit_jal(abi::zero_reg(), done);

        self.core.bind_label(return_src);
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmv_s(result_fp, src_fp),
            MachineFloatWidth::F64 => enc::fmv_d(result_fp, src_fp),
        });
        self.core.bind_label(done);
        Ok(())
    }

    fn lower_trapping_trunc(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        let spec = RiscvTruncSpec::new(op);
        let src_fp = self.fp_scratch.scoped_alloc().detach();
        self.load_fp_value_into(*src_fp, spec.width, src)?;
        let bound_fp = self.fp_scratch.scoped_alloc().detach();
        let pred = self.gp_scratch.scoped_alloc().detach();

        self.core.text.emit_u32(match spec.width {
            MachineFloatWidth::F32 => enc::feq_s(*pred, *src_fp, *src_fp),
            MachineFloatWidth::F64 => enc::feq_d(*pred, *src_fp, *src_fp),
        });
        let invalid = self
            .core
            .ensure_trap_label(MachineTrapKind::InvalidConversion);
        self.emit_branch_to(enc::Cond::Eq, *pred, abi::zero_reg(), invalid);

        self.load_fp_value_into(*bound_fp, spec.width, MachineValue::Imm64(spec.lower_bits))?;
        self.core
            .text
            .emit_u32(match (spec.width, spec.lower_traps_on_le) {
                (MachineFloatWidth::F32, true) => enc::fle_s(*pred, *src_fp, *bound_fp),
                (MachineFloatWidth::F64, true) => enc::fle_d(*pred, *src_fp, *bound_fp),
                (MachineFloatWidth::F32, false) => enc::flt_s(*pred, *src_fp, *bound_fp),
                (MachineFloatWidth::F64, false) => enc::flt_d(*pred, *src_fp, *bound_fp),
            });
        let overflow = self
            .core
            .ensure_trap_label(MachineTrapKind::IntegerOverflow);
        self.emit_branch_to(enc::Cond::Ne, *pred, abi::zero_reg(), overflow);

        self.load_fp_value_into(*bound_fp, spec.width, MachineValue::Imm64(spec.upper_bits))?;
        self.core.text.emit_u32(match spec.width {
            MachineFloatWidth::F32 => enc::flt_s(*pred, *src_fp, *bound_fp),
            MachineFloatWidth::F64 => enc::flt_d(*pred, *src_fp, *bound_fp),
        });
        self.emit_branch_to(enc::Cond::Eq, *pred, abi::zero_reg(), overflow);

        self.core.text.emit_u32(match op {
            MachineConvertOp::I32TruncF32S => enc::fcvt_w_s_rtz(dst, *src_fp),
            MachineConvertOp::I32TruncF32U => enc::fcvt_wu_s_rtz(dst, *src_fp),
            MachineConvertOp::I32TruncF64S => enc::fcvt_w_d_rtz(dst, *src_fp),
            MachineConvertOp::I32TruncF64U => enc::fcvt_wu_d_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncF32S => enc::fcvt_l_s_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncF32U => enc::fcvt_lu_s_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncF64S => enc::fcvt_l_d_rtz(dst, *src_fp),
            MachineConvertOp::I64TruncF64U => enc::fcvt_lu_d_rtz(dst, *src_fp),
            _ => unreachable!(),
        });
        if matches!(
            op,
            MachineConvertOp::I32TruncF32S
                | MachineConvertOp::I32TruncF32U
                | MachineConvertOp::I32TruncF64S
                | MachineConvertOp::I32TruncF64U
        ) {
            self.zext_w(dst, dst);
        }
        Ok(())
    }

    fn lower_trap_if(
        &mut self,
        kind: MachineTrapKind,
        cond: &MachineBranchCond,
    ) -> Result<(), WasmError> {
        let trap_label = self.core.ensure_trap_label(kind);
        self.lower_branch_if_cond(cond, trap_label, true)
    }

    fn lower_memory_grow(
        &mut self,
        mem_idx: u32,
        dst: MachineReg,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_value(preserved_io::ARG0, delta)?;
        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close(preserved_op::MEMORY_GROW, Some(result_scratch_idx));
        self.emit_mv(dst, result_scratch);
        self.gp_scratch.free_index(result_scratch_idx);
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
        self.emit_preserved_frame_open();
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
        self.emit_preserved_frame_open();
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, imm0);
        self.emit_io_store_imm(preserved_io::IMM1, imm1);
        self.emit_io_store_value(preserved_io::ARG0, arg0)?;
        self.emit_io_store_value(preserved_io::ARG1, arg1)?;
        self.emit_io_store_value(preserved_io::ARG2, arg2)?;

        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close(op_code, Some(result_scratch_idx));
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.text.emit_u32(match width {
                MachineFloatWidth::F32 => enc::fmv_w_x(dst_fp, result_scratch),
                MachineFloatWidth::F64 => enc::fmv_d_x(dst_fp, result_scratch),
            });
            self.core.set_fp_reg_width(dst, width)?;
        } else {
            let dst = self.map_gp_reg(dst)?;
            self.emit_mv(dst, result_scratch);
        }
        self.gp_scratch.free_index(result_scratch_idx);
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
        self.emit_preserved_frame_open();
        for &(slot, imm) in imms {
            self.emit_io_store_imm(slot, imm);
        }
        for &(slot, value) in args {
            self.emit_io_store_value(slot, value)?;
        }

        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close(op_code, Some(result_scratch_idx));
        if let Some(width) = ty.float_width() {
            let dst_fp = self.map_fp_reg(dst)?;
            self.core.text.emit_u32(match width {
                MachineFloatWidth::F32 => enc::fmv_w_x(dst_fp, result_scratch),
                MachineFloatWidth::F64 => enc::fmv_d_x(dst_fp, result_scratch),
            });
            self.core.set_fp_reg_width(dst, width)?;
        } else {
            let dst = self.map_gp_reg(dst)?;
            self.emit_mv(dst, result_scratch);
        }
        self.gp_scratch.free_index(result_scratch_idx);
        Ok(())
    }

    fn lower_struct_new(
        &mut self,
        type_idx: u32,
        fields: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.lower_payload_preserved_op(
            preserved_op::STRUCT_NEW,
            type_idx,
            fields,
            dst,
            "riscv64 backend received pair-valued struct.new",
        )
    }

    fn lower_array_new_fixed(
        &mut self,
        type_idx: u32,
        elements: &[(MachineValue, Option<MachineValue>)],
        dst: MachineReg,
    ) -> Result<(), WasmError> {
        self.lower_payload_preserved_op(
            preserved_op::ARRAY_NEW_FIXED,
            type_idx,
            elements,
            dst,
            "riscv64 backend received pair-valued array.new_fixed",
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
        let payload_bytes = ((items.len() as u32 * 8) + 15) & !15;
        let payload_slots = (payload_bytes / 8) as usize;
        self.emit_preserved_frame_open_with_prefix(payload_bytes);
        for (index, (value_lo, value_hi)) in items.iter().enumerate() {
            if value_hi.is_some() {
                return Err(WasmError::internal(pair_error));
            }
            self.emit_io_store_value_at(0, index, *value_lo)?;
        }
        self.emit_io_store_imm_at(payload_slots, preserved_io::IMM0, type_idx);
        self.emit_io_store_imm_at(payload_slots, preserved_io::IMM1, items.len() as u32);
        if items.is_empty() {
            self.emit_io_store_imm_at(payload_slots, preserved_io::ARG0, 0);
        } else {
            let scratch = self.gp_scratch.scoped_alloc().detach();
            self.emit_mv(*scratch, abi::stack_reg());
            self.emit_store_raw(
                0b011,
                *scratch,
                abi::stack_reg(),
                ((payload_slots + preserved_io::ARG0) * 8) as i32,
            );
        }
        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close_with_prefix(
            op_code,
            Some(result_scratch_idx),
            payload_bytes,
        );
        let dst = self.map_gp_reg(dst)?;
        self.emit_mv(dst, result_scratch);
        self.gp_scratch.free_index(result_scratch_idx);
        Ok(())
    }

    fn lower_memory_fill(
        &mut self,
        mem_idx: u32,
        dest: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, val)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::MEMORY_FILL, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_mem);
        self.emit_io_store_imm(preserved_io::IMM1, src_mem);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::MEMORY_COPY, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, mem_idx);
        self.emit_io_store_imm(preserved_io::IMM1, data_idx);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::MEMORY_INIT, None);
        Ok(())
    }

    fn lower_data_drop(&mut self, data_idx: u32) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, data_idx);
        self.emit_preserved_call_and_close(preserved_op::DATA_DROP, None);
        Ok(())
    }

    fn lower_table_grow(
        &mut self,
        table_idx: u32,
        dst: MachineReg,
        init_val: MachineValue,
        delta: MachineValue,
    ) -> Result<(), WasmError> {
        let dst = self.map_gp_reg(dst)?;
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_value(preserved_io::ARG0, init_val)?;
        self.emit_io_store_value(preserved_io::ARG1, delta)?;
        let result_scratch_idx = self.gp_scratch.alloc();
        let result_scratch = self.gp_scratch.reg(result_scratch_idx);
        self.emit_preserved_call_and_close(preserved_op::TABLE_GROW, Some(result_scratch_idx));
        self.emit_mv(dst, result_scratch);
        self.gp_scratch.free_index(result_scratch_idx);
        Ok(())
    }

    fn lower_table_fill(
        &mut self,
        table_idx: u32,
        start: MachineValue,
        val: MachineValue,
        len: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_value(preserved_io::ARG0, start)?;
        self.emit_io_store_value(preserved_io::ARG1, val)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::TABLE_FILL, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, dst_tbl);
        self.emit_io_store_imm(preserved_io::IMM1, src_tbl);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::TABLE_COPY, None);
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
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, table_idx);
        self.emit_io_store_imm(preserved_io::IMM1, elem_idx);
        self.emit_io_store_value(preserved_io::ARG0, dest)?;
        self.emit_io_store_value(preserved_io::ARG1, src)?;
        self.emit_io_store_value(preserved_io::ARG2, len)?;
        self.emit_preserved_call_and_close(preserved_op::TABLE_INIT, None);
        Ok(())
    }

    fn lower_elem_drop(&mut self, elem_idx: u32) -> Result<(), WasmError> {
        self.emit_preserved_frame_open();
        self.emit_io_store_imm(preserved_io::IMM0, elem_idx);
        self.emit_preserved_call_and_close(preserved_op::ELEM_DROP, None);
        Ok(())
    }

    fn lower_terminator_dispatch(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        match term {
            MachineTerminator::Jump(edge) => {
                if is_fallthrough_edge(
                    edge.target,
                    &edge.args,
                    fallthrough,
                    self.core.mir_blocks()?,
                ) {
                    return Ok(());
                }
                let label = self.core.emit_edge(edge.target, &edge.args)?;
                self.emit_jal(abi::zero_reg(), label);
                Ok(())
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.lower_branch(cond, then_edge, else_edge, fallthrough),
            MachineTerminator::Return | MachineTerminator::ReturnScalar { .. } => {
                self.lower_return_sequence()
            }
            MachineTerminator::Trap { kind } => {
                let trap_label = self.core.ensure_trap_label(*kind);
                self.emit_jal(abi::zero_reg(), trap_label);
                Ok(())
            }
            MachineTerminator::JumpTable { index, entries } => {
                self.lower_jump_table(*index, entries)
            }
            MachineTerminator::Call {
                target,
                frame_delta,
                args,
                results,
                success,
            } => self.lower_call(target, *frame_delta, args, results, success, fallthrough),
            MachineTerminator::TailCall { target, args } => self.lower_tail_call(target, args),
        }
    }

    fn lower_branch(
        &mut self,
        cond: &MachineBranchCond,
        then_edge: &crate::vm::jit::machine::machine_ir::MachineEdge,
        else_edge: &crate::vm::jit::machine::machine_ir::MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let blocks = self.core.mir_blocks()?;
        let then_fallthrough =
            is_fallthrough_edge(then_edge.target, &then_edge.args, fallthrough, blocks);
        let else_fallthrough =
            is_fallthrough_edge(else_edge.target, &else_edge.args, fallthrough, blocks);
        let then_label = (!then_fallthrough)
            .then(|| self.core.emit_edge(then_edge.target, &then_edge.args))
            .transpose()?;
        let else_label = (!else_fallthrough)
            .then(|| self.core.emit_edge(else_edge.target, &else_edge.args))
            .transpose()?;

        if else_fallthrough {
            if let Some(label) = then_label {
                self.lower_branch_if_cond(cond, label, true)?;
            }
        } else if then_fallthrough {
            if let Some(label) = else_label {
                self.lower_branch_if_cond(cond, label, false)?;
            }
        } else if let (Some(then_label), Some(else_label)) = (then_label, else_label) {
            self.lower_branch_if_cond(cond, then_label, true)?;
            self.emit_jal(abi::zero_reg(), else_label);
        }
        Ok(())
    }

    fn lower_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = self.core.runtime_for(self.core.func_id)?.clone();
        let fp = abi::map_fixed_reg(MACHINE_FP_REG);
        let result_base = self.gp_scratch.scoped_alloc().detach();
        let temp = self.gp_scratch.scoped_alloc().detach();
        self.emit_ld(*result_base, abi::stack_reg(), BODY_LINK_FRAME_SIZE);

        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                let frame_offset = (results.base_slot as i32 + index) * STACK_SLOT_BYTES;
                self.emit_load_raw(0b011, *temp, fp, frame_offset);
                self.emit_store_raw(0b011, *temp, *result_base, index * STACK_SLOT_BYTES);
            }
        }

        self.emit_ld(abi::link_reg(), abi::stack_reg(), BODY_LINK_RA_OFFSET);
        self.emit_restore_host_platform_regs(0);
        self.emit_ld(
            fp,
            abi::stack_reg(),
            BODY_LINK_FRAME_SIZE + STACK_SLOT_BYTES,
        );
        self.emit_addi(
            abi::stack_reg(),
            abi::stack_reg(),
            BODY_LINK_FRAME_SIZE + CALL_RECORD_SIZE,
        );
        self.emit_addi(abi::C_RET0, abi::zero_reg(), 0);
        self.core.text.emit_u32(enc::ret());
        Ok(())
    }

    fn lower_call(
        &mut self,
        target: &MachineCallTarget,
        frame_delta: u32,
        args: &MachineCallArgs,
        results: &MachineCallResults,
        success: &crate::vm::jit::machine::machine_ir::MachineEdge,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        let fp = abi::map_fixed_reg(MACHINE_FP_REG);
        let body_local_error_label = self.core.body_local_error_label;
        let continuation_label = self.core.block_label(success.target)?;
        let continuation_is_fallthrough = fallthrough == Some(success.target);

        emit_call_arg_lanes::<Self>(self, args)?;

        let callee_fp_idx = self.gp_scratch.alloc();
        let callee_fp = self.gp_scratch.reg(callee_fp_idx);
        self.materialize_frame_addr(callee_fp, fp, frame_delta);
        let caller_result_base_idx = self.gp_scratch.alloc();
        let caller_result_base = self.gp_scratch.reg(caller_result_base_idx);
        self.materialize_frame_addr(caller_result_base, fp, caller_results_base_delta(results));

        self.emit_addi(abi::stack_reg(), abi::stack_reg(), -CALL_RECORD_SIZE);
        self.emit_sd(caller_result_base, abi::stack_reg(), 0);
        self.emit_sd(fp, abi::stack_reg(), 8);
        self.gp_scratch.free_index(caller_result_base_idx);
        self.emit_mv(fp, callee_fp);
        self.gp_scratch.free_index(callee_fp_idx);

        match target {
            MachineCallTarget::Direct(callee) => {
                let scratch = self.gp_scratch.scoped_alloc().detach();
                self.emit_restore_host_platform_regs(CALL_RECORD_SIZE);
                self.emit_direct_call_target_literal(*scratch, *callee, true)?;
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                let target = self.gp_scratch.scoped_alloc().detach();
                self.emit_mv(*target, callee_entry);
                self.emit_restore_host_platform_regs(CALL_RECORD_SIZE);
                self.core
                    .text
                    .emit_u32(enc::jalr(abi::link_reg(), *target, 0));
            }
        }

        self.emit_branch_to(
            enc::Cond::Ne,
            abi::C_RET0,
            abi::zero_reg(),
            body_local_error_label,
        );
        if !continuation_is_fallthrough {
            self.emit_jal(abi::zero_reg(), continuation_label);
        }
        Ok(())
    }

    fn lower_tail_call(
        &mut self,
        target: &MachineCallTarget,
        args: &MachineCallArgs,
    ) -> Result<(), WasmError> {
        let fp = abi::map_fixed_reg(MACHINE_FP_REG);

        emit_call_arg_lanes::<Self>(self, args)?;

        match target {
            MachineCallTarget::Direct(callee) => {
                let scratch = self
                    .gp_scratch
                    .scoped_alloc_excluding(abi::link_reg())
                    .detach();
                let _ = fp;
                self.emit_ld(abi::link_reg(), abi::stack_reg(), BODY_LINK_RA_OFFSET);
                self.emit_restore_host_platform_regs(0);
                self.emit_addi(abi::stack_reg(), abi::stack_reg(), BODY_LINK_FRAME_SIZE);
                self.emit_direct_call_target_literal(*scratch, *callee, false)?;
            }
            MachineCallTarget::Indirect { callee_entry, .. } => {
                let callee_entry = self.map_gp_reg(*callee_entry)?;
                let target = self
                    .gp_scratch
                    .scoped_alloc_excluding(abi::link_reg())
                    .detach();
                self.emit_mv(*target, callee_entry);
                let _ = fp;
                self.emit_ld(abi::link_reg(), abi::stack_reg(), BODY_LINK_RA_OFFSET);
                self.emit_restore_host_platform_regs(0);
                self.emit_addi(abi::stack_reg(), abi::stack_reg(), BODY_LINK_FRAME_SIZE);
                self.core
                    .text
                    .emit_u32(enc::jalr(abi::zero_reg(), *target, 0));
            }
        }
        Ok(())
    }

    fn lower_call_runtime(&mut self, const_idx: usize) -> Result<(), WasmError> {
        let metadata = self
            .core
            .compiled
            .const_ptr(MachineConstId(const_idx as u32))
            .ok_or_else(|| WasmError::internal("riscv64 runtime-call metadata is out of range"))?;

        self.emit_preserved_frame_open();
        self.emit_mv(abi::C_ARG0, abi::map_fixed_reg(MACHINE_CTX_REG));
        self.emit_mv(abi::C_ARG1, abi::map_fixed_reg(MACHINE_FP_REG));
        self.materialize_u64(abi::C_ARG2, metadata as u64);
        {
            let scratch = self.gp_scratch.scoped_alloc().detach();
            self.materialize_u64(*scratch, call_runtime_entry_ptr() as usize as u64);
            self.emit_restore_host_platform_regs(abi::PRESERVED_HELPER_FRAME_SIZE as i32);
            self.core
                .text
                .emit_u32(enc::jalr(abi::link_reg(), *scratch, 0));
        }

        let status = self.gp_scratch.scoped_alloc().detach();
        self.emit_mv(*status, abi::C_RET0);
        self.emit_restore_preserved_gp(0);
        self.emit_restore_preserved_fp(0);
        self.emit_mv(abi::C_RET0, *status);
        self.emit_adjust_stack_up(abi::PRESERVED_HELPER_FRAME_SIZE);

        let body_local_error_label = self.core.body_local_error_label;
        self.emit_branch_to(
            enc::Cond::Ne,
            abi::C_RET0,
            abi::zero_reg(),
            body_local_error_label,
        );
        Ok(())
    }

    fn materialize_frame_addr(&mut self, dst: RiscvReg, base: RiscvReg, delta: u32) {
        if delta == 0 {
            self.emit_mv(dst, base);
        } else if delta <= i32::MAX as u32 && Self::fits_i12(delta as i32) {
            self.emit_addi(dst, base, delta as i32);
        } else {
            self.materialize_u64(dst, u64::from(delta));
            self.core.text.emit_u32(enc::add(dst, base, dst));
        }
    }

    fn lower_jump_table(
        &mut self,
        index: MachineValue,
        entries: &[crate::vm::jit::machine::machine_ir::MachineEdge],
    ) -> Result<(), WasmError> {
        if entries.is_empty() {
            return Err(WasmError::internal(
                "riscv64 MachineIR jump table requires at least one entry",
            ));
        }
        if entries.len() == 1 {
            let label = self.core.emit_edge(entries[0].target, &entries[0].args)?;
            self.emit_jal(abi::zero_reg(), label);
            return Ok(());
        }
        let index_s = self.gp_scratch.scoped_alloc().detach();
        let probe = self.gp_scratch.scoped_alloc().detach();
        self.load_value_into(*index_s, index)?;
        self.zext_w(*index_s, *index_s);
        for (entry_index, entry) in entries.iter().enumerate().take(entries.len() - 1) {
            self.materialize_u64(*probe, entry_index as u64);
            let label = self.core.emit_edge(entry.target, &entry.args)?;
            self.emit_branch_to(enc::Cond::Eq, *index_s, *probe, label);
        }
        let default = entries.last().expect("nonempty jump table");
        let label = self.core.emit_edge(default.target, &default.args)?;
        self.emit_jal(abi::zero_reg(), label);
        Ok(())
    }

    fn lower_trap_dispatch(&mut self, kind: MachineTrapKind) {
        self.emit_mv(abi::C_ARG0, abi::map_fixed_reg(MACHINE_CTX_REG));
        self.materialize_u64(abi::C_ARG1, trap_code(kind));
        let scratch = self.gp_scratch.scoped_alloc().detach();
        self.materialize_u64(*scratch, raise_trap as *const () as usize as u64);
        self.emit_restore_host_platform_regs(0);
        self.core
            .text
            .emit_u32(enc::jalr(abi::link_reg(), *scratch, 0));
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_jal(abi::zero_reg(), body_local_error_label);
    }
}
