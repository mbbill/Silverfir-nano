//! Backend state and `ArchBackend` trait glue.
//!
//! This is the bridge between the common pipeline and the example-specific
//! instruction lowering. It contains only type definitions and trait glue —
//! all lowering logic lives in `inst.rs` and `control.rs`.

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
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFunction,
            MachineInst, MachineReg, MachineTerminator, MachineTrapKind, MACHINE_CTX_REG,
            MACHINE_FP_REG,
        },
        runtime::{
            code::{CompiledNativeModule, NativeRootEntry},
            code_buf::CodeBuffer,
        },
    },
};

use super::{
    abi,
    enc::{self, Cond},
    reg::{FpReg, GpReg},
};

// ── Frame constants ──────────────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = 8;
const GP_SAVE_BYTES: u32 = abi::CALLEE_SAVED_GP.len() as u32 * STACK_SLOT_BYTES;
const FP_SAVE_BYTES: u32 = abi::CALLEE_SAVED_FP.len() as u32 * STACK_SLOT_BYTES;
const FRAME_BYTES: u32 = {
    let total = GP_SAVE_BYTES + FP_SAVE_BYTES;
    let align = abi::STACK_ALIGNMENT_BYTES;
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
}

// ── ExampleBackend ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct ExampleBackend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<GpReg, 2>,
    pub(super) fp_scratch: ScratchPool<FpReg, 3>,
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for ExampleBackend<'a> {
    const NAME: &'static str = "example";

    fn max_total_regs() -> usize {
        abi::max_total_machine_regs()
    }
    fn max_fp_regs() -> usize {
        abi::max_fp_machine_regs()
    }

    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        Self {
            core: CompilerCore::new(compiled, function, Self::max_fp_regs()),
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
        // Allocate frame and save callee-saved GP registers.
        self.core
            .text
            .emit_u32(enc::sub_imm_64(GpReg::SP, GpReg::SP, FRAME_BYTES as u16));
        for (i, reg) in abi::CALLEE_SAVED_GP.iter().copied().enumerate() {
            self.core
                .text
                .emit_u32(enc::str_64(reg, GpReg::SP, i as u16));
        }
        // A real backend would also save CALLEE_SAVED_FP here.

        // Move C entry arguments into MachineIR fixed roles.
        self.core.text.emit_u32(enc::mov_reg_64(
            abi::map_fixed_gp(MACHINE_CTX_REG),
            abi::C_ARG0,
        ));
        self.core.text.emit_u32(enc::mov_reg_64(
            abi::map_fixed_gp(MACHINE_FP_REG),
            abi::C_ARG1,
        ));
    }

    fn lower_epilogue(&mut self) {
        for (i, reg) in abi::CALLEE_SAVED_GP.iter().copied().enumerate().rev() {
            self.core
                .text
                .emit_u32(enc::ldr_64(reg, GpReg::SP, i as u16));
        }
        self.core
            .text
            .emit_u32(enc::add_imm_64(GpReg::SP, GpReg::SP, FRAME_BYTES as u16));
        self.core.text.emit_u32(enc::ret());
    }

    fn lower_root_caller_stub(&mut self) {
        // Example backend is documentation-only; the call ABI rewrite (1A)
        // never executes this path. A real backend would push a root call
        // record onto the host stack and `call/bl internal_entry_label`.
    }

    fn lower_body_prelude(&mut self) {
        // A real backend would push x29/x30 (or equivalent) when the body
        // makes any host calls — see `body_emits_native_call`.
    }

    fn lower_body_local_error_tail(&mut self) {
        // A real backend would: pop body prelude link save, pop call
        // record, restore fp_reg, native return without touching C_RET0.
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

    fn lower_trap(&mut self, _kind: MachineTrapKind) {}

    fn lower_unconditional_branch(&mut self, label: usize) {
        self.lower_b(label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target = self
                .core
                .labels
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

    // ── Scratch allocation for parallel-move protocol ────────────────────

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
        let dst_gp = self.map_gp_reg(dst)?;
        let src_gp = self.map_gp_reg(src)?;
        let temp = self.gp_scratch.reg(scratch_id);
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
        let dst_fp = self.map_fp_reg(dst.reg)?;
        let src_fp = self.map_fp_reg(src)?;
        let temp = self.fp_scratch.reg(scratch_id);
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
        CompiledExampleEntry {
            entry,
            text_len: emitted.text_len,
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}
