//! Backend state and `ArchBackend` trait glue for ARMv7-A.
//!
//! This is the bridge between the common pipeline and the ARM32-specific
//! instruction lowering. All lowering logic lives in `inst.rs` and `control.rs`.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        arch::common::{
            backend::ArchBackend,
            core::CompilerCore,
            scratch_pool::ScratchPool,
            types::{DebugRegion, ParallelSource},
        },
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFuncId,
            MachineFloatWidth, MachineFunction, MachineInst, MachineReg, MachineTerminator,
            MachineTrapKind, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG, MachineFunctionRuntime,
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
        self, emit_shared_epilogue, emit_shared_prologue, fp_machine_reg, map_fixed_reg,
        map_reg,
    },
    armv7a_raise_trap,
    enc::{self, Cond},
    inst::{emit_load_u32_into, emit_patchable_addr_into},
    reg::Arm32Reg,
    select,
};

// ── Branch fixup types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    B,
    BCond(Cond),
}

#[derive(Clone, Copy, Debug)]
struct BranchFixup {
    offset: usize,
    kind: BranchFixupKind,
    target: usize,
}

// ── Compiled entry ───────────────────────────────────────────────────────────

/// Result of compiling one function to ARM32 machine code.
#[derive(Clone, Debug)]
pub(crate) struct CompiledArm32Entry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: NativeCodePtr,
    #[cfg(has_guard_pages)]
    pub return_error: NativeCodePtr,
}

// ── Arm32Backend ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct Arm32Backend<'a> {
    pub core: CompilerCore<'a>,
    fixups: Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<Arm32Reg, 2>,
    pub(super) fp_scratch: ScratchPool<u32, 3>,
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for Arm32Backend<'a> {
    const NAME: &'static str = "armv7a";

    fn max_total_regs() -> usize { abi::max_total_machine_regs() }
    fn max_fp_regs() -> usize { abi::max_fp_machine_regs() }

    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        Self {
            core: CompilerCore::new(compiled, function, Self::max_fp_regs()),
            fixups: Vec::new(),
            gp_scratch: abi::new_gp_scratch_pool(),
            fp_scratch: abi::new_fp_scratch_pool(),
        }
    }

    fn core(&self) -> &CompilerCore<'a> { &self.core }
    fn core_mut(&mut self) -> &mut CompilerCore<'a> { &mut self.core }
    fn into_core(self) -> CompilerCore<'a> { self.core }

    fn lower_prologue(&mut self) {
        emit_shared_prologue(&mut self.core.text);
        // Move args: entry is `fn(ctx: *mut NativeContext, fp: *mut u64) -> u32`
        // R0 = ctx, R1 = fp → fixed CTX / FP machine regs
        self.core.text.emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_CTX_REG), Arm32Reg::R0));
        self.core.text.emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), Arm32Reg::R1));
        // Load fixed mem0 registers
        self.core.text.emit_u32(enc::ldr_imm(
            map_fixed_reg(MACHINE_MEM0_BASE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_BASE as i32,
        ));
        self.core.text.emit_u32(enc::ldr_imm(
            map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_SIZE as i32,
        ));
    }

    fn lower_epilogue(&mut self) {
        emit_shared_epilogue(&mut self.core.text);
    }

    fn lower_return_ok_status(&mut self) {
        self.emit_load_u32(Arm32Reg::R0, 0);
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
        }
        self.core.current_op_index = None;
        let result = self.lower_terminator(&block.terminator, fallthrough);
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
        // Shared trap handler: called once per trap kind, shared by all TrapIf sites.
        // R0 = ctx, R1 = trap code, R2 = trap site
        self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
        let trap_code = select::trap_kind_to_u32(kind);
        self.emit_load_u32(Arm32Reg::R1, trap_code);
        let site = select::encode_trap_site(self.core.function.id.0, None);
        self.emit_load_u32(Arm32Reg::R2, site);
        self.emit_host_call(armv7a_raise_trap as usize);
        // armv7a_raise_trap returns 1 in R0 → branch to return_error_label
        let return_error = self.core.return_error_label;
        self.emit_branch(BranchFixupKind::B, return_error);
    }

    fn lower_unconditional_branch(&mut self, label: usize) {
        self.emit_branch(BranchFixupKind::B, label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target_offset = self.core.labels
                .get(fixup.target)
                .and_then(|v| *v)
                .ok_or_else(|| WasmError::internal("armv7a branch label unresolved".into()))?;
            let delta = target_offset as i32 - fixup.offset as i32;
            let inst = match fixup.kind {
                BranchFixupKind::B => enc::b(delta),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta),
            };
            self.core.text.patch_u32(fixup.offset, inst);
        }
        Ok(())
    }

    // ── Scratch allocation for parallel-move protocol ────────────────────

    fn alloc_gp_scratch(&mut self) -> u8 { self.gp_scratch.alloc() }
    fn free_gp_scratch(&mut self, id: u8) { self.gp_scratch.free_index(id) }
    fn alloc_fp_scratch(&mut self) -> u8 { self.fp_scratch.alloc() }
    fn free_fp_scratch(&mut self, id: u8) { self.fp_scratch.free_index(id) }

    // ── Parallel move primitives ─────────────────────────────────────────

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
        let dst_gp = map_reg(dst)?;
        let src_gp = map_reg(src)?;
        let temp = self.gp_scratch.reg(scratch_id);
        self.core.text.emit_u32(enc::mov_reg(temp, dst_gp));
        self.core.text.emit_u32(enc::mov_reg(dst_gp, src_gp));
        Ok(())
    }

    fn lower_fp_cycle_break(
        &mut self,
        dst: MachineBlockParam,
        src: MachineReg,
        _float_width: Option<MachineFloatWidth>,
        scratch_id: u8,
    ) -> Result<(), WasmError> {
        let dst_d = self.map_fp_dreg(dst.reg)?;
        let src_d = self.map_fp_dreg(src)?;
        let temp = self.fp_scratch.reg(scratch_id);
        self.core.text.emit_u32(enc::vmov_d(temp, dst_d));
        self.core.text.emit_u32(enc::vmov_d(dst_d, src_d));
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        debug_assert!(bytes % 4 == 0, "ARM32 NOP padding must be 4-byte aligned");
        const ARM32_NOP: [u8; 4] = 0xe1a00000_u32.to_le_bytes();
        for _ in 0..bytes / 4 {
            buf.emit_bytes(&ARM32_NOP);
        }
    }

    type CompiledEntry = CompiledArm32Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::arch::common::pipeline::EmittedFunction,
    ) -> Self::CompiledEntry {
        let entry = unsafe { buf.fn_ptr::<NativeRootEntry>(emitted.text_offset) };
        let root_return = unsafe { buf.ptr(emitted.text_offset + emitted.root_return_offset) };
        #[cfg(has_guard_pages)]
        let return_error = unsafe { buf.ptr(emitted.text_offset + emitted.return_error_offset) };
        CompiledArm32Entry {
            entry,
            text_len: emitted.text_len,
            debug_regions: emitted.debug_regions.clone(),
            root_return,
            #[cfg(has_guard_pages)]
            return_error,
        }
    }
}

// ── Helper methods ───────────────────────────────────────────────────────────

impl<'a> Arm32Backend<'a> {

    // ── FP register mapping ──────────────────────────────────────────────

    #[inline]
    pub(super) fn is_fp_machine_reg(&self, reg: MachineReg) -> bool {
        self.core.is_fp_reg(reg)
    }

    #[inline]
    pub(super) fn map_fp_dreg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let fp_idx = crate::vm::machine::machine_ir::fp_reg_index(reg, self.core.compiled.backend())
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "armv7a: expected FP register, got GP machine reg {}",
                    reg.0
                ))
            })?;
        fp_machine_reg(fp_idx).ok_or_else(|| {
            WasmError::invalid(alloc::format!(
                "armv7a: FP machine reg index {} out of range",
                fp_idx
            ))
        })
    }

    // ── Branch emission ──────────────────────────────────────────────────

    pub(super) fn emit_branch(&mut self, kind: BranchFixupKind, target: usize) {
        let offset = self.core.text.len();
        self.core.text.emit_u32(enc::nop());
        self.fixups.push(BranchFixup {
            offset,
            kind,
            target,
        });
    }

    // ── Immediate loading ────────────────────────────────────────────────

    /// Load a 32-bit immediate into a register using MOV imm / MOVW+MOVT.
    pub(super) fn emit_load_u32(&mut self, dst: Arm32Reg, value: u32) {
        if let Some((imm8, rot)) = enc::encode_arm_imm(value) {
            self.core.text.emit_u32(enc::mov_imm(dst, imm8, rot));
        } else {
            self.core.text.emit_u32(enc::movw(dst, value as u16));
            let hi = (value >> 16) as u16;
            if hi != 0 {
                self.core.text.emit_u32(enc::movt(dst, hi));
            }
        }
    }

    /// Load a pointer-sized absolute address into a register.
    pub(super) fn emit_load_addr(&mut self, dst: Arm32Reg, addr: usize) {
        self.emit_load_u32(dst, addr as u32);
    }

    // ── Host call ────────────────────────────────────────────────────────

    #[inline]
    pub(super) fn emit_host_call(&mut self, target: usize) {
        let s = self.gp_scratch.scoped_alloc();
        emit_load_u32_into(&mut self.core.text, *s, target as u32);
        self.core.text.emit_u32(enc::blx_reg(*s));
    }

    // ── GP value materialization ─────────────────────────────────────────

    pub(super) fn emit_move_gp_value(
        &mut self,
        dst: Arm32Reg,
        value: &MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                if dst != src {
                    self.core.text.emit_u32(enc::mov_reg(dst, src));
                }
            }
            MachineValue::Imm64(value) => self.emit_load_u32(dst, *value as u32),
        }
        Ok(())
    }

    pub(super) fn materialize_gp_into(
        &mut self,
        dst: Arm32Reg,
        value: &MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                if dst != src {
                    self.core.text.emit_u32(enc::mov_reg(dst, src));
                }
            }
            MachineValue::Imm64(v) => {
                self.emit_load_u32(dst, *v as u32);
            }
        }
        Ok(())
    }

    // ── Pair argument / result shuffling ─────────────────────────────────

    pub(super) fn emit_pair_args_to_r0_r1(
        &mut self,
        src_lo: &MachineValue,
        src_hi: &MachineValue,
    ) -> Result<(), WasmError> {
        let src_lo_reg = match src_lo {
            MachineValue::Reg(r) => Some(map_reg(*r)?),
            MachineValue::Imm64(_) => None,
        };
        let src_hi_reg = match src_hi {
            MachineValue::Reg(r) => Some(map_reg(*r)?),
            MachineValue::Imm64(_) => None,
        };
        if matches!(src_lo_reg, Some(Arm32Reg::R1)) && matches!(src_hi_reg, Some(Arm32Reg::R0)) {
            let s = self.gp_scratch.scoped_alloc();
            self.core.text.emit_u32(enc::mov_reg(*s, Arm32Reg::R0));
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R1));
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R1, *s));
            return Ok(());
        }
        let moved_hi = matches!(src_hi_reg, Some(Arm32Reg::R0)) && !matches!(src_lo_reg, Some(Arm32Reg::R0));
        if moved_hi {
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R1, Arm32Reg::R0));
        }
        if !matches!(src_lo_reg, Some(Arm32Reg::R0)) {
            self.emit_move_gp_value(Arm32Reg::R0, src_lo)?;
        }
        if !moved_hi && !matches!(src_hi_reg, Some(Arm32Reg::R1)) {
            self.emit_move_gp_value(Arm32Reg::R1, src_hi)?;
        }
        Ok(())
    }

    pub(super) fn emit_pair_results_from_r0_r1(
        &mut self,
        dst_lo: MachineReg,
        dst_hi: MachineReg,
    ) -> Result<(), WasmError> {
        let dst_lo_hw = map_reg(dst_lo)?;
        let dst_hi_hw = map_reg(dst_hi)?;
        if dst_lo_hw == Arm32Reg::R1 && dst_hi_hw == Arm32Reg::R0 {
            let s = self.gp_scratch.scoped_alloc();
            self.core.text.emit_u32(enc::mov_reg(*s, Arm32Reg::R0));
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R1));
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R1, *s));
            return Ok(());
        }
        let moved_lo = dst_hi_hw == Arm32Reg::R0 && dst_lo_hw != Arm32Reg::R0;
        if moved_lo {
            self.core.text.emit_u32(enc::mov_reg(dst_lo_hw, Arm32Reg::R0));
        }
        let moved_hi = dst_lo_hw == Arm32Reg::R1 && dst_hi_hw != Arm32Reg::R1;
        if moved_hi {
            self.core.text.emit_u32(enc::mov_reg(dst_hi_hw, Arm32Reg::R1));
        }
        if !moved_lo && dst_lo_hw != Arm32Reg::R0 {
            self.core.text.emit_u32(enc::mov_reg(dst_lo_hw, Arm32Reg::R0));
        }
        if !moved_hi && dst_hi_hw != Arm32Reg::R1 {
            self.core.text.emit_u32(enc::mov_reg(dst_hi_hw, Arm32Reg::R1));
        }
        Ok(())
    }

    // ── Caller-saved register spill / restore ────────────────────────────

    pub(super) fn spill_caller_saved_gp_regs(&mut self) {
        self.core.text.emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 16, 0));
        self.core.text.emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
        self.core.text.emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
        self.core.text.emit_u32(enc::str_imm(Arm32Reg::R2, Arm32Reg::SP, 8));
        self.core.text.emit_u32(enc::str_imm(Arm32Reg::R3, Arm32Reg::SP, 12));
    }

    pub(super) fn restore_caller_saved_gp_regs(&mut self, preserved: &[Arm32Reg]) {
        if !preserved.contains(&Arm32Reg::R0) {
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
        }
        if !preserved.contains(&Arm32Reg::R1) {
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
        }
        if !preserved.contains(&Arm32Reg::R2) {
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 8));
        }
        if !preserved.contains(&Arm32Reg::R3) {
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 12));
        }
        self.core.text.emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 16, 0));
    }

    // ── Stack-staged value moves ─────────────────────────────────────────

    pub(super) fn emit_values_to_regs_via_stack(
        &mut self,
        regs: &[Arm32Reg],
        values: &[&MachineValue],
    ) -> Result<(), WasmError> {
        if regs.len() != values.len() {
            return Err(WasmError::internal(
                "armv7a stack-staged value move requires matching regs and values".into(),
            ));
        }
        for value in values {
            let s = self.gp_scratch.scoped_alloc();
            let push_mask = 1 << (*s).idx();
            match *value {
                MachineValue::Reg(r) => {
                    let src = map_reg(*r)?;
                    if *s != src {
                        self.core.text.emit_u32(enc::mov_reg(*s, src));
                    }
                }
                MachineValue::Imm64(v) => emit_load_u32_into(&mut self.core.text, *s, *v as u32),
            }
            self.core.text.emit_u32(enc::push(push_mask));
        }
        for reg in regs.iter().rev() {
            self.core.text.emit_u32(enc::pop(1 << reg.idx()));
        }
        Ok(())
    }

    pub(super) fn emit_quad_args_to_r0_r3(
        &mut self,
        value0: &MachineValue,
        value1: &MachineValue,
        value2: &MachineValue,
        value3: &MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_values_to_regs_via_stack(
            &[Arm32Reg::R0, Arm32Reg::R1, Arm32Reg::R2, Arm32Reg::R3],
            &[value0, value1, value2, value3],
        )
    }

    // ── Compare / bool helpers ───────────────────────────────────────────

    pub(super) fn emit_set_bool_immediate(&mut self, dst: Arm32Reg, value: bool) {
        self.emit_load_u32(dst, u32::from(value));
    }

    // ── Stack temp alloc/free ────────────────────────────────────────────

    pub(super) fn emit_stack_temp_alloc(&mut self, bytes: u32) {
        self.core.text.emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, bytes, 0));
    }

    pub(super) fn emit_stack_temp_free(&mut self, bytes: u32) {
        self.core.text.emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, bytes, 0));
    }

    pub(super) fn emit_trunc_result_buffer_alloc(&mut self) {
        self.emit_stack_temp_alloc(16);
    }

    pub(super) fn emit_trunc_result_buffer_free(&mut self) {
        self.emit_stack_temp_free(16);
    }

    // ── Parallel move source dispatch ────────────────────────────────────

    fn lower_source_move_dispatch(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        if self.is_fp_machine_reg(dst.reg) {
            let dd = self.map_fp_dreg(dst.reg)?;
            match src {
                ParallelSource::Reg { reg, .. } if self.is_fp_machine_reg(reg) => {
                    let sd = self.map_fp_dreg(reg)?;
                    if dd != sd {
                        self.core.text.emit_u32(enc::vmov_d(dd, sd));
                    }
                }
                ParallelSource::Reg { reg, .. } => {
                    // GP → FP
                    let src_gp = map_reg(reg)?;
                    self.emit_load_u32(Arm32Reg::R1, 0);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, src_gp, Arm32Reg::R1));
                }
                ParallelSource::Imm(value) => {
                    let lo = value as u32;
                    let hi = (value >> 32) as u32;
                    self.emit_load_u32(Arm32Reg::R0, lo);
                    self.emit_load_u32(Arm32Reg::R1, hi);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                }
                ParallelSource::GpTemp(id) => {
                    let temp = self.gp_scratch.reg(id);
                    self.emit_load_u32(Arm32Reg::R1, 0);
                    self.core.text.emit_u32(enc::vmov_d_rr(dd, temp, Arm32Reg::R1));
                }
                ParallelSource::FpTemp(id, _) => {
                    let temp = self.fp_scratch.reg(id);
                    self.core.text.emit_u32(enc::vmov_d(dd, temp));
                }
            }
            if let Some(width) = dst.ty.float_width() {
                self.core.set_fp_reg_width(dst.reg, width)?;
            }
        } else {
            let dst_gp = map_reg(dst.reg)?;
            match src {
                ParallelSource::Reg { reg, .. } if self.is_fp_machine_reg(reg) => {
                    // FP → GP: extract low 32 bits
                    let sd = self.map_fp_dreg(reg)?;
                    self.core.text.emit_u32(enc::vmov_rr_d(dst_gp, Arm32Reg::R1, sd));
                }
                ParallelSource::Reg { reg, .. } => {
                    let src_gp = map_reg(reg)?;
                    if dst_gp != src_gp {
                        self.core.text.emit_u32(enc::mov_reg(dst_gp, src_gp));
                    }
                }
                ParallelSource::Imm(value) => {
                    self.emit_load_u32(dst_gp, value as u32);
                }
                ParallelSource::GpTemp(id) => {
                    let temp = self.gp_scratch.reg(id);
                    self.core.text.emit_u32(enc::mov_reg(dst_gp, temp));
                }
                ParallelSource::FpTemp(id, _) => {
                    let temp = self.fp_scratch.reg(id);
                    self.core.text.emit_u32(enc::vmov_rr_d(dst_gp, Arm32Reg::R1, temp));
                }
            }
        }
        Ok(())
    }

    // ── Call infrastructure ──────────────────────────────────────────────

    pub(super) fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionRuntime, WasmError> {
        self.core.runtime_for(func_id)
    }

    pub(super) fn emit_call_direct(
        &mut self,
        callee: MachineFuncId,
        callee_frame_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_runtime = self.runtime_for(callee)?;
        let call_scratch = callee_runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("armv7a direct call requires callee call scratch".into())
        })?;
        let call_link = self.core.compiled.runtime().call_link;
        let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;

        let callee_fp_orig = map_reg(callee_frame_base)?;
        // callee_fp must stay allocated for the entire sequence so that
        // inner scoped_alloc() calls never return the same register.
        let callee_fp_guard = self.gp_scratch.scoped_alloc();
        let callee_fp = *callee_fp_guard;
        self.core.text.emit_u32(enc::mov_reg(callee_fp, callee_fp_orig));

        // Store continuation address (patchable) into callee frame
        let cont_byte_offset = (continuation_slot as i32) * 8;
        let cont_patch = {
            let s = self.gp_scratch.scoped_alloc();
            let patch = emit_patchable_addr_into(&mut self.core.text, *s);
            self.core.text.emit_u32(enc::str_imm(*s, callee_fp, cont_byte_offset));
            patch
        };
        emit_load_u32_into(&mut self.core.text, Arm32Reg::R3, 0);
        self.core.text.emit_u32(enc::str_imm(Arm32Reg::R3, callee_fp, cont_byte_offset + 4));

        // Load callee entry (patchable) and jump
        let callee_patch = {
            let s = self.gp_scratch.scoped_alloc();
            let patch = emit_patchable_addr_into(&mut self.core.text, *s);
            self.core.text.emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), callee_fp));
            self.core.text.emit_u32(enc::bx(*s));
            patch
        };
        drop(callee_fp_guard);

        // Record patches
        let continuation_label = self.core.block_label(continuation)?;
        self.core.local_ptr_patches.push(
            crate::vm::arch::common::types::PendingLocalPtrPatch {
                literal_offset: cont_patch,
                target_label: continuation_label,
            },
        );
        self.core.direct_call_patches.push(
            crate::vm::arch::common::types::DirectCallPatch {
                literal_offset: callee_patch,
                callee,
            },
        );
        Ok(())
    }

    pub(super) fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.runtime_for(self.core.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("armv7a local return requires call scratch".into())
        })?;
        let call_link = self.core.compiled.runtime().call_link;
        let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;
        let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
        let caller_result_base_slot =
            call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;

        let fp_reg = map_fixed_reg(MACHINE_FP_REG);

        // Load continuation address into scratch (lives until bx)
        let cont_reg = self.gp_scratch.scoped_alloc().release();
        self.core.text.emit_u32(enc::ldr_imm(cont_reg, fp_reg, (continuation_slot as i32) * 8));
        // Load caller FP
        self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R3, fp_reg, (caller_frame_slot as i32) * 8));
        // Load caller result base
        self.core.text.emit_u32(enc::ldr_imm(
            Arm32Reg::R0, fp_reg, (caller_result_base_slot as i32) * 8,
        ));
        // Compute absolute result address: caller_fp + result_base
        self.core.text.emit_u32(enc::add_reg(Arm32Reg::R0, Arm32Reg::R3, Arm32Reg::R0));

        // Copy return results to caller frame
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                self.core.text.emit_u32(enc::ldr_imm(
                    Arm32Reg::R1, fp_reg, (results.base_slot as i32 + index) * 8,
                ));
                self.core.text.emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::R0, index * 8));
                self.core.text.emit_u32(enc::ldr_imm(
                    Arm32Reg::R1, fp_reg, (results.base_slot as i32 + index) * 8 + 4,
                ));
                self.core.text.emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::R0, index * 8 + 4));
            }
        }

        self.core.text.emit_u32(enc::mov_reg(fp_reg, Arm32Reg::R3));
        self.core.text.emit_u32(enc::bx(cont_reg));
        Ok(())
    }

    pub(super) fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        _arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_fp_orig = map_reg(callee_frame_base)?;
        // callee_fp must stay allocated so inner scoped_alloc() never
        // returns the same register.
        let callee_fp_guard = self.gp_scratch.scoped_alloc();
        let callee_fp = *callee_fp_guard;
        self.core.text.emit_u32(enc::mov_reg(callee_fp, callee_fp_orig));

        // Materialize callee ID into R3
        let callee_id_reg = match callee_target {
            MachineValue::Reg(r) => map_reg(r)?,
            MachineValue::Imm64(v) => {
                emit_load_u32_into(&mut self.core.text, Arm32Reg::R3, v as u32);
                Arm32Reg::R3
            }
        };
        if callee_id_reg != Arm32Reg::R3 {
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R3, callee_id_reg));
        }

        // Load function info table base, compute entry address, load fields,
        // and perform stack overflow check — all using one scratch that is
        // dropped before emit_branch (which needs &mut self).
        {
            let s = self.gp_scratch.scoped_alloc();
            let table_patch = emit_patchable_addr_into(&mut self.core.text, *s);
            self.core.function_table_patches.push(table_patch);

            // Each Arm32FunctionInfo is 16 bytes. Compute entry: table + callee_id * 16
            self.core.text.emit_u32(enc::lsl_imm(Arm32Reg::R3, Arm32Reg::R3, 4));
            self.core.text.emit_u32(enc::add_reg(*s, *s, Arm32Reg::R3));

            // Load function info fields
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R0, *s, 0)); // entry
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R1, *s, 4)); // total_frame_bytes
            self.core.text.emit_u32(enc::ldr_imm(Arm32Reg::R2, *s, 12)); // call_scratch_base_slot

            // Stack overflow check
            self.core.text.emit_u32(enc::add_reg(*s, callee_fp, Arm32Reg::R1));
            self.core.text.emit_u32(enc::ldr_imm(
                Arm32Reg::R3,
                map_fixed_reg(MACHINE_CTX_REG),
                ctx_offset::STACK_END as i32,
            ));
            self.core.text.emit_u32(enc::cmp_reg(*s, Arm32Reg::R3));
        }
        // Inline emit_branch: callee_fp_guard is alive, can't call &mut self.
        let stack_overflow = self.core.stack_overflow_label;
        {
            let offset = self.core.text.len();
            self.core.text.emit_u32(enc::nop());
            self.fixups.push(BranchFixup { offset, kind: BranchFixupKind::BCond(Cond::Hi), target: stack_overflow });
        }

        // Compute call_scratch absolute byte offset
        self.core.text.emit_u32(enc::lsl_imm(Arm32Reg::R2, Arm32Reg::R2, 3));
        self.core.text.emit_u32(enc::add_reg(Arm32Reg::R2, callee_fp, Arm32Reg::R2));

        // Store continuation address (patchable)
        let call_link = self.core.compiled.runtime().call_link;
        let cont_patch = {
            let s = self.gp_scratch.scoped_alloc();
            let patch = emit_patchable_addr_into(&mut self.core.text, *s);
            self.core.text.emit_u32(enc::str_imm(*s, Arm32Reg::R2, call_link.continuation_offset));
            patch
        };
        emit_load_u32_into(&mut self.core.text, Arm32Reg::R3, 0);
        self.core.text.emit_u32(enc::str_imm(Arm32Reg::R3, Arm32Reg::R2, call_link.continuation_offset + 4));

        // Store caller FP
        self.core.text.emit_u32(enc::str_imm(
            map_fixed_reg(MACHINE_FP_REG), Arm32Reg::R2, call_link.caller_frame_offset,
        ));
        self.core.text.emit_u32(enc::str_imm(
            Arm32Reg::R3, Arm32Reg::R2, call_link.caller_frame_offset + 4,
        ));

        // Store caller result base
        {
            let s = self.gp_scratch.scoped_alloc();
            emit_load_u32_into(&mut self.core.text, *s, u32::from(caller_result_base) * 8);
            self.core.text.emit_u32(enc::str_imm(
                *s, Arm32Reg::R2, call_link.caller_result_base_offset,
            ));
        }
        self.core.text.emit_u32(enc::str_imm(
            Arm32Reg::R3, Arm32Reg::R2, call_link.caller_result_base_offset + 4,
        ));

        // Set FP to callee and jump
        self.core.text.emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.core.text.emit_u32(enc::bx(Arm32Reg::R0));
        drop(callee_fp_guard);

        // Record continuation patch
        let continuation_label = self.core.block_label(continuation)?;
        self.core.local_ptr_patches.push(
            crate::vm::arch::common::types::PendingLocalPtrPatch {
                literal_offset: cont_patch,
                target_label: continuation_label,
            },
        );
        Ok(())
    }
}
