//! ARM64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the arm64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth,
            MachineInst, MachineFunction, MachineInstKind,
            MachineReg, MachineTerminator, MachineTrapKind,
            MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{NativeCodePtr, NativeRootEntry, CompiledNativeModule},
            code_buf::CodeBuffer,
            context::ctx_offset,
        },
    },
};

use super::{abi, enc, reg::{Arm64FpReg, Arm64Reg}};
use super::abi::{
    max_fp_machine_regs, max_total_machine_regs, FP_MACHINE_REG_COUNT,
};
use crate::vm::arch::common::{
    backend::ArchBackend,
    core::CompilerCore,
    scratch_pool::ScratchPool,
    types::{DebugRegion, ParallelSource},
};

// ── Frame layout constants ───────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;
const CALLEE_SAVED_GP_FRAME_SIZE: u32 = abi::REG_PLAN.callee_saved_gp_pairs.len() as u32 * (2 * STACK_SLOT_BYTES);
const CALLEE_SAVED_FP_FRAME_OFFSET: u32 = CALLEE_SAVED_GP_FRAME_SIZE;
const CALLEE_SAVED_FP_FRAME_SIZE: u32 = abi::REG_PLAN.callee_saved_fp.len() as u32 * STACK_SLOT_BYTES;
const CALLEE_SAVED_FRAME_SIZE: u32 = {
    let total = CALLEE_SAVED_FP_FRAME_OFFSET + CALLEE_SAVED_FP_FRAME_SIZE;
    total.div_ceil(abi::REG_PLAN.stack_alignment_bytes) * abi::REG_PLAN.stack_alignment_bytes
};

const fn stack_u64_slot(offset_bytes: u32) -> u32 {
    offset_bytes / STACK_SLOT_BYTES
}

const fn stack_pair_imm(offset_bytes: u32) -> i32 {
    (offset_bytes / STACK_SLOT_BYTES) as i32
}

// ── Branch fixup types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    B,
    BCond(enc::Cond),
    Cbz(Arm64Reg),
    Cbnz(Arm64Reg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    pub inst_offset: usize,
    pub label: usize,
    pub kind: BranchFixupKind,
}

// ── Compiled entry ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct CompiledArm64Entry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: NativeCodePtr,
}

// ── Arm64Backend ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct Arm64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<Arm64Reg, 2>,
    pub(super) fp_scratch: ScratchPool<Arm64FpReg, 3>,
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for Arm64Backend<'a> {
    const NAME: &'static str = "arm64";

    fn max_total_regs() -> usize { max_total_machine_regs() }
    fn max_fp_regs() -> usize { max_fp_machine_regs() }

    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        Self {
            core: CompilerCore::new(compiled, function, FP_MACHINE_REG_COUNT),
            fixups: Vec::new(),
            gp_scratch: abi::new_gp_scratch_pool(),
            fp_scratch: abi::new_fp_scratch_pool(),
        }
    }

    fn core(&self) -> &CompilerCore<'a> { &self.core }
    fn core_mut(&mut self) -> &mut CompilerCore<'a> { &mut self.core }
    fn into_core(self) -> CompilerCore<'a> { self.core }

    fn lower_prologue(&mut self) {
        // Allocate frame and save callee-saved registers.
        self.core.text.emit_u32(enc::sub_imm_64(
            Arm64Reg::SP, Arm64Reg::SP, CALLEE_SAVED_FRAME_SIZE,
        ));
        for (index, (lhs, rhs)) in abi::REG_PLAN.callee_saved_gp_pairs.iter().copied().enumerate() {
            self.core.text.emit_u32(enc::stp_64(
                lhs, rhs, Arm64Reg::SP,
                stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
            ));
        }
        for (index, reg) in abi::REG_PLAN.callee_saved_fp.iter().copied().enumerate() {
            self.core.text.emit_u32(enc::str_d(
                reg, Arm64Reg::SP,
                stack_u64_slot(CALLEE_SAVED_FP_FRAME_OFFSET + index as u32 * STACK_SLOT_BYTES),
            ));
        }

        // Move entry arguments into pinned roles.
        let ctx = abi::map_fixed_reg(MACHINE_CTX_REG);
        let frame = abi::map_fixed_reg(MACHINE_FP_REG);
        self.core.text.emit_u32(enc::mov_reg_64(ctx, abi::C_ARG0));
        self.core.text.emit_u32(enc::mov_reg_64(frame, abi::C_ARG1));
        self.core.text.emit_u32(enc::ldr_64(
            abi::map_fixed_reg(MACHINE_MEM0_BASE_REG),
            ctx,
            (ctx_offset::MEM0_BASE / 8) as u32,
        ));
        self.core.text.emit_u32(enc::ldr_64(
            abi::map_fixed_reg(MACHINE_MEM0_SIZE_REG),
            ctx,
            (ctx_offset::MEM0_SIZE / 8) as u32,
        ));
    }

    fn lower_epilogue(&mut self) {
        // Restore callee-saved FP registers.
        for (index, reg) in abi::REG_PLAN.callee_saved_fp.iter().copied().enumerate() {
            self.core.text.emit_u32(enc::ldr_d(
                reg, Arm64Reg::SP,
                stack_u64_slot(CALLEE_SAVED_FP_FRAME_OFFSET + index as u32 * STACK_SLOT_BYTES),
            ));
        }
        // Restore callee-saved GP registers and deallocate frame.
        for (index, (lhs, rhs)) in abi::REG_PLAN.callee_saved_gp_pairs.iter().copied().enumerate() {
            self.core.text.emit_u32(enc::ldp_64(
                lhs, rhs, Arm64Reg::SP,
                stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
            ));
        }
        self.core.text.emit_u32(enc::add_imm_64(
            Arm64Reg::SP, Arm64Reg::SP, CALLEE_SAVED_FRAME_SIZE,
        ));
        self.core.text.emit_u32(enc::ret());
    }

    fn lower_return_ok_status(&mut self) {
        self.materialize_u64(abi::C_RET0, 0);
    }

    fn lower_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core.current_block = Some(block.id);
        self.core.current_edge_target = None;
        self.core.reset_block_fp_state(block)?;

        let fused_fcmp_cond = super::fusion::float_compare_branch_fusion(
            block, &self.core.function.program.blocks,
        );

        let mut index = 0;
        while index < block.ops.len() {
            self.core.current_op_index = Some(index);
            if fused_fcmp_cond.is_some() && index == block.ops.len() - 1 {
                if let MachineInstKind::FloatCompare { width, lhs, rhs, .. }
                    = &block.ops[index].kind
                {
                    self.lower_fcmp_values(*width, *lhs, *rhs)?;
                    self.gp_scratch.assert_all_free();
                    self.fp_scratch.assert_all_free();
                    index += 1;
                    continue;
                }
            }
            if let Some((base, imm7)) = super::fusion::zero_store_pair_fusion(block, index) {
                let base_reg = self.map_gp_reg(base)?;
                self.core.text.emit_u32(
                    enc::stp_64(Arm64Reg::Xzr, Arm64Reg::Xzr, base_reg, imm7),
                );
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                index += 2;
                continue;
            }
            self.lower_inst(&block.ops[index])?;
            // Catch scratch leaks between instructions.
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
            index += 1;
        }
        self.core.current_op_index = None;

        let result = if let Some(cond) = fused_fcmp_cond {
            match &block.terminator {
                MachineTerminator::Branch { then_edge, else_edge, .. } => {
                    self.lower_fused_cond_branch(cond, then_edge, else_edge, fallthrough)
                }
                _ => unreachable!(),
            }
        } else {
            self.lower_terminator(&block.terminator, fallthrough)
        };
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
        self.lower_b(label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self.core.labels
                .get(fixup.label)
                .and_then(|v| *v)
                .ok_or_else(|| {
                    WasmError::internal("arm64 branch target label is unresolved".into())
                })?;
            let delta_words = ((target as isize) - (fixup.inst_offset as isize)) / 4;
            let patched = match fixup.kind {
                BranchFixupKind::B => enc::b(delta_words as i32),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta_words as i32),
                BranchFixupKind::Cbz(reg) => enc::cbz_64(reg, delta_words as i32),
                BranchFixupKind::Cbnz(reg) => enc::cbnz_64(reg, delta_words as i32),
            };
            self.core.text.patch_u32(fixup.inst_offset, patched);
        }
        Ok(())
    }

    fn alloc_gp_scratch(&mut self) -> u8 { self.gp_scratch.alloc() }
    fn free_gp_scratch(&mut self, id: u8) { self.gp_scratch.free_index(id) }
    fn alloc_fp_scratch(&mut self) -> u8 { self.fp_scratch.alloc() }
    fn free_fp_scratch(&mut self, id: u8) { self.fp_scratch.free_index(id) }

    fn lower_source_move(
        &mut self, dst: MachineBlockParam, src: ParallelSource,
    ) -> Result<(), WasmError> {
        self.lower_source_move_dispatch(dst, src)
    }

    fn lower_gp_cycle_break(
        &mut self, dst: MachineReg, src: MachineReg, scratch_id: u8,
    ) -> Result<(), WasmError> {
        let temp = self.gp_scratch.reg(scratch_id);
        let dst_gp = self.map_gp_reg(dst)?;
        let src_gp = self.map_gp_reg(src)?;
        self.core.text.emit_u32(enc::mov_reg_64(temp, dst_gp));
        self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
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
        let dst_fp = self.map_fp_reg(dst.reg)?;
        let width = dst.ty.float_width().expect("FP param width");
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmov_s(temp, dst_fp),
            MachineFloatWidth::F64 => enc::fmov_d(temp, dst_fp),
        });
        let src_fp = self.map_fp_reg(src)?;
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::fmov_s(dst_fp, src_fp),
            MachineFloatWidth::F64 => enc::fmov_d(dst_fp, src_fp),
        });
        self.core.set_fp_reg_width(dst.reg, width)?;
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        debug_assert!(bytes % 4 == 0, "ARM64 NOP padding must be 4-byte aligned");
        const ARM64_NOP: [u8; 4] = 0xd503201f_u32.to_le_bytes();
        for _ in 0..bytes / 4 {
            buf.emit_bytes(&ARM64_NOP);
        }
    }

    type CompiledEntry = CompiledArm64Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::arch::common::pipeline::EmittedFunction,
    ) -> Self::CompiledEntry {
        let entry = unsafe { buf.fn_ptr::<NativeRootEntry>(emitted.text_offset) };
        let root_return = unsafe { buf.ptr(emitted.text_offset + emitted.root_return_offset) };
        CompiledArm64Entry {
            entry,
            text_len: emitted.text_len,
            debug_regions: emitted.debug_regions.clone(),
            root_return,
        }
    }
}
