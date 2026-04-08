//! x86_64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the x86_64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFunction,
            MachineInst, MachineReg, MachineTerminator, MachineTrapKind, MachineValue,
            MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{CompiledNativeModule, NativeCodePtr, NativeRootEntry},
            code_buf::CodeBuffer,
            context::ctx_offset,
        },
    },
};

use super::{
    abi::{
        self, fp_machine_reg, map_fixed_reg, map_reg, max_fp_machine_regs, max_total_machine_regs,
        C_ARG0, C_ARG1, C_RET0, FP_MACHINE_REG_COUNT,
    },
    callconv,
    enc::{self, Cc},
    helpers::x86_64_raise_trap,
    reg::X86Reg,
};
#[cfg(sf_has_debug_regions)]
use crate::vm::arch::common::types::DebugRegion;
use crate::vm::arch::common::{
    backend::ArchBackend,
    core::CompilerCore,
    helpers::trap_code,
    scratch_pool::ScratchPool,
    types::ParallelSource,
};

// ── Frame layout ────────────────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;

/// Extra bytes subtracted from RSP after the GP saves. Owned by the active
/// calling convention because the exact shape depends on whether Win64 XMM
/// spill and shadow space are required. See `callconv::sysv::STACK_PADDING`
/// / `callconv::win64::STACK_PADDING`.
const STACK_PADDING: u32 = callconv::STACK_PADDING;

// ── Branch fixup types ───────────────────────────────────────────────────────

/// x86_64 branch fixup: we emit a JMP/Jcc with a placeholder rel32, then
/// patch it once the target label is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    /// Byte offset of the rel32 field in the text.
    pub rel32_offset: usize,
    /// Label index to resolve.
    pub label: usize,
}

// ── Compiled entry ───────────────────────────────────────────────────────────

/// Result of compiling one function to x86_64 machine code.
#[derive(Clone, Debug)]
pub(crate) struct CompiledX86_64Entry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    pub root_return: NativeCodePtr,
    #[cfg(sf_has_guard_pages)]
    pub return_error: NativeCodePtr,
    #[cfg(sf_ir_dump)]
    pub debug_regions: Vec<DebugRegion>,
}

// ── X86_64Backend ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct X86_64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<X86Reg, 2>,
    pub(super) fp_scratch: ScratchPool<u32, 3>,
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for X86_64Backend<'a> {
    const NAME: &'static str = "x86_64";

    fn max_total_regs() -> usize {
        max_total_machine_regs()
    }
    fn max_fp_regs() -> usize {
        max_fp_machine_regs()
    }

    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        Self {
            core: CompilerCore::new(compiled, function, FP_MACHINE_REG_COUNT),
            fixups: Vec::new(),
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
        for &reg in abi::callee_saved_gp_regs() {
            enc::push(&mut self.core.text, reg);
        }
        if STACK_PADDING > 0 {
            if STACK_PADDING <= 127 {
                enc::sub_rsp_imm8(&mut self.core.text, STACK_PADDING as u8);
            } else {
                enc::sub_rsp_imm32(&mut self.core.text, STACK_PADDING);
            }
        }
        // ABI-specific spills (e.g. XMM6..XMM15 on Win64).
        callconv::emit_prologue_extra(self);
        enc::mov_rr_64(&mut self.core.text, map_fixed_reg(MACHINE_CTX_REG), C_ARG0);
        enc::mov_rr_64(&mut self.core.text, map_fixed_reg(MACHINE_FP_REG), C_ARG1);
        // Load mem0 base/size from ctx
        enc::load_64(
            &mut self.core.text,
            map_fixed_reg(MACHINE_MEM0_BASE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_BASE as i32,
        );
        enc::load_64(
            &mut self.core.text,
            map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_SIZE as i32,
        );
    }

    fn lower_epilogue(&mut self) {
        // ABI-specific restores (mirror of `callconv::emit_prologue_extra`).
        callconv::emit_epilogue_extra(self);
        if STACK_PADDING > 0 {
            if STACK_PADDING <= 127 {
                enc::add_rsp_imm8(&mut self.core.text, STACK_PADDING as u8);
            } else {
                enc::add_rsp_imm32(&mut self.core.text, STACK_PADDING);
            }
        }
        for &reg in abi::callee_saved_gp_regs().iter().rev() {
            enc::pop(&mut self.core.text, reg);
        }
        enc::ret(&mut self.core.text);
    }

    fn lower_return_ok_status(&mut self) {
        enc::mov_ri_32(&mut self.core.text, C_RET0, 0);
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
        let scratch = self.gp_scratch.reg(1); // R11
                                              // MOV RDI, ctx (arg0)
        enc::mov_rr_64(&mut self.core.text, C_ARG0, map_fixed_reg(MACHINE_CTX_REG));
        // MOV RSI, trap_code (arg1)
        self.materialize_u64(C_ARG1, trap_code(kind));
        // MOV R11, x86_64_raise_trap
        self.materialize_u64(scratch, x86_64_raise_trap as usize as u64);
        // CALL R11
        enc::call_reg(&mut self.core.text, scratch);
        // JMP return_error_label
        self.emit_jmp(self.core.return_error_label);
    }

    fn lower_unconditional_branch(&mut self, label: usize) {
        self.emit_jmp(label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self
                .core
                .labels
                .get(fixup.label)
                .and_then(|value| *value)
                .ok_or_else(|| {
                    WasmError::internal("x86_64 branch target label is unresolved".into())
                })?;
            enc::patch_rel32(&mut self.core.text, fixup.rel32_offset, target);
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
        let dst_gp = self.map_gp_reg(dst)?;
        let src_gp = self.map_gp_reg(src)?;
        enc::mov_rr_64(&mut self.core.text, temp, dst_gp);
        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
        Ok(())
    }

    fn lower_fp_cycle_break(
        &mut self,
        dst: MachineBlockParam,
        src: MachineReg,
        _float_width: Option<MachineFloatWidth>,
        scratch_id: u8,
    ) -> Result<(), WasmError> {
        let temp = self.fp_scratch.reg(scratch_id);
        let dst_fp = self.map_fp_reg(dst.reg)? as u8;
        let width = dst.ty.float_width().expect("FP param width");
        match width {
            MachineFloatWidth::F32 => enc::movss_rr(&mut self.core.text, temp as u8, dst_fp),
            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.core.text, temp as u8, dst_fp),
        };
        let src_fp = self.map_fp_reg(src)? as u8;
        match width {
            MachineFloatWidth::F32 => enc::movss_rr(&mut self.core.text, dst_fp, src_fp),
            MachineFloatWidth::F64 => enc::movsd_rr(&mut self.core.text, dst_fp, src_fp),
        };
        self.core.set_fp_reg_width(dst.reg, width)?;
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        const INT3: u8 = 0xCC;
        for _ in 0..bytes {
            buf.emit_bytes(&[INT3]);
        }
    }

    type CompiledEntry = CompiledX86_64Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::arch::common::pipeline::EmittedFunction,
    ) -> Self::CompiledEntry {
        let entry = unsafe { buf.fn_ptr::<NativeRootEntry>(emitted.text_offset) };
        let root_return = unsafe { buf.ptr(emitted.text_offset + emitted.root_return_offset) };
        #[cfg(sf_has_guard_pages)]
        let return_error = unsafe { buf.ptr(emitted.text_offset + emitted.return_error_offset) };
        CompiledX86_64Entry {
            entry,
            root_return,
            #[cfg(sf_has_guard_pages)]
            return_error,
            text_len: emitted.text_len,
            #[cfg(sf_ir_dump)]
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}

// ── Inherent helper methods ──────────────────────────────────────────────────

impl<'a> X86_64Backend<'a> {
    // ── Branch fixup helpers ─────────────────────────────────────────────

    /// Emit JMP rel32 with a fixup to be patched later.
    pub(super) fn emit_jmp(&mut self, label: usize) {
        let rel32_offset = enc::jmp_rel32(&mut self.core.text);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    /// Emit Jcc rel32 with a fixup to be patched later.
    pub(super) fn emit_jcc(&mut self, cc: Cc, label: usize) {
        let rel32_offset = enc::jcc_rel32(&mut self.core.text, cc);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    // ── Register mapping ─────────────────────────────────────────────────

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<X86Reg, WasmError> {
        self.core.validate_gp_reg(reg)?;
        map_reg(reg)
    }

    pub(super) fn map_fp_reg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let index = self.core.fp_reg_index(reg)?;
        fp_machine_reg(index).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "x86_64 backend has no physical FP mapping for machine reg {}",
                reg.0
            ))
        })
    }

    // ── Value materialization ────────────────────────────────────────────

    pub(super) fn materialize_u64(&mut self, dst: X86Reg, value: u64) {
        if value == 0 {
            enc::xor_rr_32(&mut self.core.text, dst, dst);
        } else if value <= u32::MAX as u64 {
            enc::mov_ri_32(&mut self.core.text, dst, value as u32);
        } else {
            enc::mov_ri_64(&mut self.core.text, dst, value);
        }
    }

    pub(super) fn materialize_value(
        &mut self,
        scratch: X86Reg,
        value: MachineValue,
    ) -> Result<X86Reg, WasmError> {
        match value {
            MachineValue::Reg(reg) if self.core.is_fp_reg(reg) => {
                let src_fp = self.map_fp_reg(reg)?;
                match self.core.fp_reg_width(reg)? {
                    MachineFloatWidth::F32 => {
                        enc::movd_r32_xmm(&mut self.core.text, scratch, src_fp as u8);
                    }
                    MachineFloatWidth::F64 => {
                        enc::movq_r64_xmm(&mut self.core.text, scratch, src_fp as u8);
                    }
                };
                Ok(scratch)
            }
            MachineValue::Reg(reg) => self.map_gp_reg(reg),
            MachineValue::Imm64(value) => {
                self.materialize_u64(scratch, value);
                Ok(scratch)
            }
        }
    }

    pub(super) fn prepare_float_operand(
        &mut self,
        width: MachineFloatWidth,
        value: MachineValue,
        gp_scratch: X86Reg,
        fp_scratch: u32,
    ) -> Result<u32, WasmError> {
        if let MachineValue::Reg(reg) = value {
            if self.core.is_fp_reg(reg) {
                return Ok(self.map_fp_reg(reg)?);
            }
        }
        let gp = self.materialize_value(gp_scratch, value)?;
        match width {
            MachineFloatWidth::F32 => enc::movd_xmm_r32(&mut self.core.text, fp_scratch as u8, gp),
            MachineFloatWidth::F64 => enc::movq_xmm_r64(&mut self.core.text, fp_scratch as u8, gp),
        };
        Ok(fp_scratch)
    }

    // ── Source move dispatch ─────────────────────────────────────────────

    fn lower_source_move_dispatch(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        match src {
            ParallelSource::Reg {
                reg: src_reg,
                float_width: src_float_width,
            } => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    if self.core.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movss_rr(&mut self.core.text, dst_fp, src_fp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movsd_rr(&mut self.core.text, dst_fp, src_fp)
                            }
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        match width {
                            MachineFloatWidth::F32 => {
                                enc::movd_xmm_r32(&mut self.core.text, dst_fp, src_gp)
                            }
                            MachineFloatWidth::F64 => {
                                enc::movq_xmm_r64(&mut self.core.text, dst_fp, src_gp)
                            }
                        };
                    }
                    self.core.set_fp_reg_width(dst.reg, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    if self.core.is_fp_reg(src_reg) {
                        let src_fp = self.map_fp_reg(src_reg)? as u8;
                        match src_float_width.ok_or_else(|| {
                            WasmError::invalid(alloc::format!(
                                "x86_64 edge move is missing float-width metadata for machine reg {}",
                                src_reg.0
                            ))
                        })? {
                            MachineFloatWidth::F32 => enc::movd_r32_xmm(&mut self.core.text, dst_gp, src_fp),
                            MachineFloatWidth::F64 => enc::movq_r64_xmm(&mut self.core.text, dst_gp, src_fp),
                        };
                    } else {
                        let src_gp = self.map_gp_reg(src_reg)?;
                        enc::mov_rr_64(&mut self.core.text, dst_gp, src_gp);
                    }
                }
            }
            ParallelSource::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "x86_64 received non-identity reserved cache edge move into {} from {}",
                    dst.reg.0,
                    reg.0
                )));
            }
            ParallelSource::Imm(value) => {
                if let Some(width) = dst.ty.float_width() {
                    let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                    let scratch = self.gp_scratch.reg(0);
                    self.materialize_u64(scratch, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, scratch)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, scratch)
                        }
                    };
                    self.core.set_fp_reg_width(dst.reg, width)?;
                } else {
                    self.materialize_u64(self.map_gp_reg(dst.reg)?, value);
                }
            }
            ParallelSource::GpTemp(id) => {
                let temp = self.gp_scratch.reg(id);
                let dst_gp = self.map_gp_reg(dst.reg)?;
                enc::mov_rr_64(&mut self.core.text, dst_gp, temp);
            }
            ParallelSource::FpTemp(id, width) => {
                let temp = self.fp_scratch.reg(id);
                let dst_fp = self.map_fp_reg(dst.reg)? as u8;
                match width {
                    MachineFloatWidth::F32 => {
                        enc::movss_rr(&mut self.core.text, dst_fp, temp as u8)
                    }
                    MachineFloatWidth::F64 => {
                        enc::movsd_rr(&mut self.core.text, dst_fp, temp as u8)
                    }
                };
                self.core.set_fp_reg_width(dst.reg, width)?;
            }
        }
        Ok(())
    }
}
