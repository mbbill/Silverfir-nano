//! Backend state and `ArchBackend` trait glue.
//!
//! This is the bridge between the common pipeline and the example-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs`.

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
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth,
            MachineFunction, MachineInst, MachineReg, MachineTerminator, MachineTrapKind,
        },
        runtime::{
            code::{CompiledNativeModule, NativeCodePtr, NativeRootEntry},
            code_buf::CodeBuffer,
        },
    },
};

use super::{
    abi::RegisterPlan,
    enc::{self, Cond},
    reg::{FpReg, GpReg},
};

// ── Frame constants ──────────────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = 8;
const GP_SAVE_BYTES: u32 = RegisterPlan::CALLEE_SAVED_GP.len() as u32 * STACK_SLOT_BYTES;
const FP_SAVE_BYTES: u32 = RegisterPlan::CALLEE_SAVED_FP.len() as u32 * STACK_SLOT_BYTES;
const FRAME_BYTES: u32 = {
    let total = GP_SAVE_BYTES + FP_SAVE_BYTES;
    let align = RegisterPlan::STACK_ALIGNMENT_BYTES;
    total.div_ceil(align) * align
};

// ── Branch fixup types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    B,
    BCond(Cond),
    Cbz(GpReg),
    Cbnz(GpReg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BranchFixup {
    pub inst_offset: usize,
    pub label: usize,
    pub kind: BranchFixupKind,
}

// ── Compiled entry ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct CompiledExampleEntry {
    pub entry: NativeRootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: NativeCodePtr,
}

// ── ExampleBackend ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct ExampleBackend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<GpReg, 2>,
    pub(super) fp_scratch: ScratchPool<FpReg, 2>,
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for ExampleBackend<'a> {
    const NAME: &'static str = "example";

    fn max_total_regs() -> usize { RegisterPlan::max_total_machine_regs() }
    fn max_fp_regs() -> usize { RegisterPlan::max_fp_machine_regs() }

    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        Self {
            core: CompilerCore::new(compiled, function, Self::max_fp_regs()),
            fixups: Vec::new(),
            gp_scratch: ScratchPool::new(RegisterPlan::GP_SCRATCHES),
            fp_scratch: ScratchPool::new(RegisterPlan::FP_SCRATCHES),
        }
    }

    fn core(&self) -> &CompilerCore<'a> { &self.core }
    fn core_mut(&mut self) -> &mut CompilerCore<'a> { &mut self.core }
    fn into_core(self) -> CompilerCore<'a> { self.core }

    fn emit_prologue(&mut self) {
        // Allocate frame and save callee-saved GP registers.
        self.core.text.emit_u32(enc::sub_imm_64(GpReg::SP, GpReg::SP, FRAME_BYTES as u16));
        for (i, reg) in RegisterPlan::CALLEE_SAVED_GP.iter().copied().enumerate() {
            self.core.text.emit_u32(enc::str_64(reg, GpReg::SP, i as u16));
        }
        // A real backend would also save CALLEE_SAVED_FP here.

        // Move entry arguments into pinned roles.
        self.core.text.emit_u32(enc::mov_reg_64(
            RegisterPlan::FIXED.ctx, RegisterPlan::ENTRY.ctx_arg,
        ));
        self.core.text.emit_u32(enc::mov_reg_64(
            RegisterPlan::FIXED.frame, RegisterPlan::ENTRY.frame_arg,
        ));
    }

    fn emit_epilogue(&mut self) {
        // Restore callee-saved GP registers and deallocate frame.
        for (i, reg) in RegisterPlan::CALLEE_SAVED_GP.iter().copied().enumerate().rev() {
            self.core.text.emit_u32(enc::ldr_64(reg, GpReg::SP, i as u16));
        }
        self.core.text.emit_u32(enc::add_imm_64(GpReg::SP, GpReg::SP, FRAME_BYTES as u16));
        self.core.text.emit_u32(enc::ret());
    }

    fn emit_return_ok_status(&mut self) {
        self.materialize_u64(RegisterPlan::RETURN.status, 0);
    }

    fn emit_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core.current_block = Some(block.id);
        self.core.current_edge_target = None;
        self.core.reset_block_fp_state(block)?;
        for (index, inst) in block.ops.iter().enumerate() {
            self.core.current_op_index = Some(index);
            self.emit_inst(inst)?;
            // Catch scratch leaks between instructions.
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
        }
        self.core.current_op_index = None;
        let result = self.emit_terminator(&block.terminator, fallthrough);
        self.core.current_block = None;
        result
    }

    fn emit_inst(&mut self, inst: &MachineInst) -> Result<(), WasmError> {
        self.lower_inst(inst)
    }

    fn emit_terminator(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.lower_terminator(term, fallthrough)
    }

    fn emit_trap(&mut self, _kind: MachineTrapKind) {
        // A real backend would emit a trap stub here.
    }

    fn emit_unconditional_branch(&mut self, label: usize) {
        self.lower_b(label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self.core.labels
                .get(fixup.label)
                .and_then(|v| *v)
                .ok_or_else(|| WasmError::internal("example branch label unresolved".into()))?;
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

    fn emit_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        self.lower_source_move(dst, src)
    }

    fn emit_gp_cycle_break(
        &mut self,
        dst: MachineReg,
        src: MachineReg,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let src_gp = self.map_gp_reg(src)?;
        let temp = RegisterPlan::TEMPS.gp_cycle_break;
        self.core.text.emit_u32(enc::mov_reg_64(temp, dst_gp));
        self.core.text.emit_u32(enc::mov_reg_64(dst_gp, src_gp));
        Ok(())
    }

    fn emit_fp_cycle_break(
        &mut self,
        dst: MachineBlockParam,
        src: MachineReg,
        _float_width: Option<MachineFloatWidth>,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst.reg)?;
        let src_fp = self.map_fp_reg(src)?;
        let temp = RegisterPlan::TEMPS.fp_cycle_break;
        self.core.text.emit_u32(enc::fmov_d(temp, dst_fp));
        self.core.text.emit_u32(enc::fmov_d(dst_fp, src_fp));
        Ok(())
    }

    fn emit_nop_padding(buf: &mut CodeBuffer, bytes: usize) {
        for _ in 0..bytes / 4 {
            buf.emit_bytes(&0u32.to_le_bytes());
        }
    }

    type CompiledEntry = CompiledExampleEntry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::arch::common::pipeline::EmittedFunction,
    ) -> Self::CompiledEntry {
        let entry = unsafe { buf.fn_ptr::<NativeRootEntry>(emitted.text_offset) };
        let root_return = unsafe { buf.ptr(emitted.text_offset + emitted.root_return_offset) };
        CompiledExampleEntry {
            entry,
            text_len: emitted.text_len,
            debug_regions: emitted.debug_regions.clone(),
            root_return,
        }
    }
}
