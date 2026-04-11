//! x86_64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the x86_64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFunction,
            MachineInst, MachineIntWidth, MachineReg, MachineStorageType, MachineTerminator,
            MachineTrapKind, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{CompiledNativeModule, NativeRootEntry},
            code_buf::CodeBuffer,
            context::ctx_offset,
        },
    },
};

use super::{
    abi::{
        self, fp_machine_reg, map_fixed_reg, map_reg, max_fp_machine_regs, max_total_machine_regs,
        C_ARG0, C_ARG1, FP_MACHINE_REG_COUNT,
    },
    callconv,
    enc::{self, Cc},
    gp_scratch::GpScratchPool,
    reg::X86Reg,
};
#[cfg(sf_has_debug_regions)]
use crate::vm::arch::common::types::DebugRegion;
use crate::vm::arch::common::{
    backend::ArchBackend, core::CompilerCore, scratch_pool::ScratchPool, types::ParallelSource,
};

// ── Frame layout ────────────────────────────────────────────────────────────

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
    #[cfg(sf_ir_dump)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

// ── X86_64Backend ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct X86_64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: collections::Vec<BranchFixup>,
    pub(super) gp_scratch: GpScratchPool,
    pub(super) fp_scratch: ScratchPool<u32, 2>,
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
            fixups: collections::Vec::new(),
            gp_scratch: GpScratchPool::new(abi::gp_backend_owned_regs()),
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

    /// Public-entry caller stub. Pushes a root call record onto the host
    /// stack — two 8-byte slots, low = caller_result_base, high = caller_fp,
    /// both = `fp_reg` (= MACHINE_FP_REG) for the root call — then `call`s
    /// the internal entry. After the body's unified Return rets, control
    /// falls through to `lower_epilogue`.
    fn lower_root_caller_stub(&mut self) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // push caller_fp (→ higher slot after the next push)
        enc::push(&mut self.core.text, fp_reg);
        // push caller_result_base (→ lower slot)
        enc::push(&mut self.core.text, fp_reg);
        // call internal_entry_label (patched by patch_fixups)
        let internal_entry_label = self.core.internal_entry_label;
        self.emit_call_rel32(internal_entry_label);
    }

    /// Body entry prelude. On x86_64, `call` already pushed the return
    /// address onto the host stack when the caller stub entered, leaving SP
    /// misaligned by 8 (relative to the 16-byte requirement). The body
    /// prelude subtracts 8 more to restore 16-byte alignment so the body
    /// can make nested C helper / preserved-helper calls without tripping
    /// the ABI's stack alignment check.
    ///
    /// The body's unified Return and `body_local_error_label` both `add
    /// rsp, 8` before popping the call record, mirroring this shim.
    fn lower_body_prelude(&mut self) {
        enc::sub_rsp_imm8(&mut self.core.text, 8);
    }

    /// Body-local error tail (`body_local_error_label`). Reached from trap
    /// stubs and post-call status checks. Undoes the body prelude, loads
    /// the caller's `fp_reg` back from the call record, and `ret 16`s —
    /// releasing the 16-byte call record sitting above the return
    /// address. `C_RET0` is untouched, preserving the error code set by
    /// the trap stub or inherited from a trapped descendant's call.
    ///
    /// Stack layout at entry:
    ///
    /// ```text
    ///   [rsp +  0] = alignment shim (body prelude)
    ///   [rsp +  8] = return address
    ///   [rsp + 16] = caller_result_base (unused in the error path)
    ///   [rsp + 24] = caller_fp
    /// ```
    fn lower_body_local_error_tail(&mut self) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // Undo body prelude alignment shim.
        enc::add_rsp_imm8(&mut self.core.text, 8);
        // Load caller_fp from [rsp+16] into fp_reg.
        enc::load_64(&mut self.core.text, fp_reg, X86Reg::RSP, 16);
        // ret 16 — pops return address, then releases the 16-byte call
        // record. C_RET0 untouched — preserves the error code.
        enc::ret_imm16(&mut self.core.text, 16);
    }

    fn lower_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        self.lower_inst_dispatch(inst)
    }

    fn lower_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core.current_block = Some(block.id);
        self.core.current_edge_target = None;
        self.core.reset_block_fp_state(block)?;
        for (index, inst) in block.ops.iter().enumerate() {
            self.core.current_op_index = Some(index);
            self.lower_inst(inst)?;
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
        }
        self.core.current_op_index = None;

        let result = self.lower_terminator(&block.terminator, fallthrough);
        self.gp_scratch.assert_all_free();
        self.fp_scratch.assert_all_free();
        self.core.current_block = None;
        result
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
}

impl<'a> crate::vm::arch::shared_64::ModuleLinkBackend64<'a> for X86_64Backend<'a> {
    type CompiledEntry = CompiledX86_64Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::arch::shared_64::EmittedFunction64,
    ) -> Self::CompiledEntry {
        let entry =
            unsafe { buf.fn_ptr::<crate::vm::runtime::code::NativeRootEntry>(emitted.text_offset) };
        CompiledX86_64Entry {
            entry,
            text_len: emitted.text_len,
            #[cfg(sf_ir_dump)]
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}

// ── Inherent helper methods ──────────────────────────────────────────────────

impl<'a> X86_64Backend<'a> {
    #[inline]
    pub(super) fn emit_gp_move_width(&mut self, width: MachineIntWidth, dst: X86Reg, src: X86Reg) {
        match width {
            MachineIntWidth::I32 => enc::mov_rr_32(&mut self.core.text, dst, src),
            MachineIntWidth::I64 => enc::mov_rr_64(&mut self.core.text, dst, src),
        }
    }

    #[inline]
    pub(super) fn emit_gp_move_ty(
        &mut self,
        ty: MachineStorageType,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        match ty {
            // GpWord carries both i32 values and references. Preserve the full
            // 64-bit carrier here so ref null sentinels survive plain moves.
            MachineStorageType::GpWord => enc::mov_rr_64(&mut self.core.text, dst, src),
            MachineStorageType::GpI64 => enc::mov_rr_64(&mut self.core.text, dst, src),
            MachineStorageType::Fp32 | MachineStorageType::Fp64 => {
                return Err(WasmError::internal(
                    "x86_64 GP move requested for FP storage type".into(),
                ))
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn emit_gp_cmov_ty(
        &mut self,
        ty: MachineStorageType,
        cc: Cc,
        dst: X86Reg,
        src: X86Reg,
    ) -> Result<(), WasmError> {
        match ty {
            // GpWord covers references as well as i32 carriers; keep the full
            // 64-bit payload when selecting between them.
            MachineStorageType::GpWord => enc::cmovcc_rr_64(&mut self.core.text, cc, dst, src),
            MachineStorageType::GpI64 => enc::cmovcc_rr_64(&mut self.core.text, cc, dst, src),
            MachineStorageType::Fp32 | MachineStorageType::Fp64 => {
                return Err(WasmError::internal(
                    "x86_64 GP cmov requested for FP storage type".into(),
                ))
            }
        }
        Ok(())
    }

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

    /// Emit CALL rel32 with a fixup to be patched later.
    pub(super) fn emit_call_rel32(&mut self, label: usize) {
        let rel32_offset = enc::call_rel32(&mut self.core.text);
        self.fixups.push(BranchFixup {
            rel32_offset,
            label,
        });
    }

    // ── Register mapping ─────────────────────────────────────────────────

    pub(super) fn map_gp_reg(&self, reg: MachineReg) -> Result<X86Reg, WasmError> {
        crate::vm::arch::shared_64::validate_gp_reg(self, reg)?;
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
            MachineValue::ReservedReg(reg) => Err(WasmError::internal(alloc::format!(
                "x86_64 cannot materialize reserved cache register {}",
                reg.0
            ))),
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
                    let scratch = self.gp_scratch.scoped_alloc().detach();
                    self.materialize_u64(*scratch, value);
                    match width {
                        MachineFloatWidth::F32 => {
                            enc::movd_xmm_r32(&mut self.core.text, dst_fp, *scratch)
                        }
                        MachineFloatWidth::F64 => {
                            enc::movq_xmm_r64(&mut self.core.text, dst_fp, *scratch)
                        }
                    };
                    self.core.set_fp_reg_width(dst.reg, width)?;
                } else {
                    let dst_gp = self.map_gp_reg(dst.reg)?;
                    self.materialize_u64(dst_gp, value);
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
