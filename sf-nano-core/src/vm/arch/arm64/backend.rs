//! ARM64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the arm64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId,
            MachineFunction, MachineInst, MachineReg, MachineTerminator, MachineTrapKind,
            MACHINE_CTX_REG, MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{CodegenModuleView, NativeRootEntry},
            code_buf::CodeBuffer,
            context::ctx_offset,
        },
    },
};

use super::abi::{max_fp_machine_regs, max_total_machine_regs};
use super::{
    abi, enc,
    reg::{Arm64FpReg, Arm64Reg},
};
#[cfg(sf_ir_dump)]
use crate::vm::arch::common::types::DebugRegion;
use crate::vm::arch::common::{
    backend::ArchBackend, core::CompilerCore, scratch_pool::ScratchPool, types::ParallelSource,
};

// ── Frame layout constants ───────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;
const CALLEE_SAVED_GP_FRAME_SIZE: u32 =
    abi::callee_saved_gp_pair_count() as u32 * (2 * STACK_SLOT_BYTES);
const CALLEE_SAVED_FP_FRAME_OFFSET: u32 = CALLEE_SAVED_GP_FRAME_SIZE;
// Match the preserved-helper ABI: scalar builds only need the low 64-bit D
// payload of callee-saved FP regs, while SIMD builds must preserve the full
// 128-bit Q contents because the FP bank also carries v128 values.
#[cfg(sf_has_simd)]
const CALLEE_SAVED_FP_SLOT_BYTES: u32 = 16;
#[cfg(not(sf_has_simd))]
const CALLEE_SAVED_FP_SLOT_BYTES: u32 = STACK_SLOT_BYTES;
const CALLEE_SAVED_FP_FRAME_SIZE: u32 =
    abi::callee_saved_fp_count() as u32 * CALLEE_SAVED_FP_SLOT_BYTES;
const CALLEE_SAVED_FRAME_SIZE: u32 = {
    let total = CALLEE_SAVED_FP_FRAME_OFFSET + CALLEE_SAVED_FP_FRAME_SIZE;
    total.div_ceil(abi::stack_alignment_bytes()) * abi::stack_alignment_bytes()
};

#[cfg(not(sf_has_simd))]
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
    /// `bl <label>` — branch with link, populates LR.
    Bl,
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
    #[cfg(sf_ir_dump)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

// ── Arm64Backend ─────────────────────────────────────────────────────────────

/// Per-call deferred literal: the patchable callee-address word for one
/// `lower_call_direct` site, queued so the literal lives in the
/// end-of-body literal pool instead of inline after the BLR. Lets the
/// caller elide its trailing `b continuation` when the next emitted block
/// is the continuation block.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingCallLiteral {
    /// Offset of the `ldr_lit_64` instruction inside `core.text`. Patched
    /// during literal-pool flush so its pc-relative offset points at the
    /// emitted literal.
    pub ldr_offset: usize,
    /// Scratch register the LDR loads into. Re-encoded as part of the
    /// patched LDR instruction.
    pub scratch_reg: Arm64Reg,
    /// Local callee id; recorded into `direct_call_patches` once the
    /// literal slot has a real offset, so module-link patching can write
    /// the resolved address into the literal word.
    pub callee: MachineFuncId,
}

#[derive(Debug)]
pub(crate) struct Arm64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: collections::Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<Arm64Reg, 2>,
    pub(super) fp_scratch: ScratchPool<Arm64FpReg, 2>,
    /// Deferred per-call patchable literals. Flushed by
    /// `lower_function_literal_pool` (which the pipeline calls between
    /// edge stubs and the body-local error tail).
    pub(super) pending_call_literals: collections::Vec<PendingCallLiteral>,
    /// 1-slot peephole lookahead buffer. Holds the previously seen op so
    /// `emit_inst_at` can attempt a 2-op fusion (zero-store pair, FP
    /// store/load pair) when the next op arrives. Drained on `end_block`.
    /// The inst is owned (cloned) because the pipeline driver moves the
    /// source block out of the function via `mem::replace` and frees it
    /// after `end_block` returns — references would dangle.
    /// `usize` is the block-local op index at the time it was buffered.
    pending_op: Option<(MachineInst, usize)>,
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for Arm64Backend<'a> {
    const NAME: &'static str = "arm64";

    fn max_total_regs() -> usize {
        max_total_machine_regs()
    }
    fn max_fp_regs() -> usize {
        max_fp_machine_regs()
    }

    fn new(compiled: &'a dyn CodegenModuleView, function: MachineFunction) -> Self {
        Self {
            core: CompilerCore::new(compiled, function),
            fixups: collections::Vec::new(),
            gp_scratch: abi::new_gp_scratch_pool(),
            fp_scratch: abi::new_fp_scratch_pool(),
            pending_call_literals: collections::Vec::new(),
            pending_op: None,
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
        // Allocate frame and save callee-saved registers.
        self.core.text.emit_u32(enc::sub_imm_64(
            abi::stack_reg(),
            abi::stack_reg(),
            CALLEE_SAVED_FRAME_SIZE,
        ));
        for (index, (lhs, rhs)) in abi::callee_saved_gp_pairs().iter().copied().enumerate() {
            self.core.text.emit_u32(enc::stp_64(
                lhs,
                rhs,
                abi::stack_reg(),
                stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
            ));
        }
        // Save callee-saved FP regs.
        let fp_regs = abi::callee_saved_fp_regs();
        let mut fp_idx = 0usize;
        while fp_idx < fp_regs.len() {
            let byte_off =
                CALLEE_SAVED_FP_FRAME_OFFSET + (fp_idx as u32) * CALLEE_SAVED_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::str_q(fp_regs[fp_idx], abi::stack_reg(), byte_off / 16));
            #[cfg(not(sf_has_simd))]
            self.core.text.emit_u32(enc::str_d(
                fp_regs[fp_idx],
                abi::stack_reg(),
                stack_u64_slot(byte_off),
            ));
            fp_idx += 1;
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
        let fp_regs = abi::callee_saved_fp_regs();
        let mut fp_idx = 0usize;
        while fp_idx < fp_regs.len() {
            let byte_off =
                CALLEE_SAVED_FP_FRAME_OFFSET + (fp_idx as u32) * CALLEE_SAVED_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::ldr_q(fp_regs[fp_idx], abi::stack_reg(), byte_off / 16));
            #[cfg(not(sf_has_simd))]
            self.core.text.emit_u32(enc::ldr_d(
                fp_regs[fp_idx],
                abi::stack_reg(),
                stack_u64_slot(byte_off),
            ));
            fp_idx += 1;
        }
        // Restore callee-saved GP registers and deallocate frame.
        for (index, (lhs, rhs)) in abi::callee_saved_gp_pairs().iter().copied().enumerate() {
            self.core.text.emit_u32(enc::ldp_64(
                lhs,
                rhs,
                abi::stack_reg(),
                stack_pair_imm((index as u32) * 2 * STACK_SLOT_BYTES),
            ));
        }
        self.core.text.emit_u32(enc::add_imm_64(
            abi::stack_reg(),
            abi::stack_reg(),
            CALLEE_SAVED_FRAME_SIZE,
        ));
        self.core.text.emit_u32(enc::ret());
    }

    /// Public-entry caller stub. Pushes a root call record onto the host
    /// stack (caller_result_base = stack_base, caller_fp = stack_base) and
    /// `bl`s the internal entry. After the body's unified Return rets, the
    /// stack pointer is back to the post-prologue value (the body's Return
    /// pops the call record we pushed here), and `C_RET0` already holds 0
    /// or a trap kind. Falls through to `lower_epilogue`.
    fn lower_root_caller_stub(&mut self) {
        // x20 = MACHINE_FP_REG = the root frame base after the prologue.
        // For the root call, both caller_fp and caller_result_base point at
        // the root frame so the unified Return copies results into the
        // bytes that `eval.rs::collect_native_results_from_stack` reads.
        let fp_reg = abi::map_fixed_reg(MACHINE_FP_REG);
        // stp fp_reg, fp_reg, [sp, #-16]!
        self.core
            .text
            .emit_u32(enc::stp_64_pre_index(fp_reg, fp_reg, abi::stack_reg(), -16));
        // bl internal_entry_label  (resolved by patch_fixups)
        let internal_entry_label = self.core.internal_entry_label;
        self.lower_bl(internal_entry_label);
    }

    /// Body entry prelude. Push x29/x30 onto the host stack so the body can
    /// freely make nested `bl`s without losing the LR set by the caller's
    /// `bl` into this function. The body's unified Return and
    /// `body_local_error_label` both pop this pair before the native `ret`.
    ///
    /// Native backends emit this prelude unconditionally today; a future leaf
    /// optimization may skip it when the body makes no native calls.
    fn lower_body_prelude(&mut self) {
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();
        // stp x29, x30, [sp, #-16]!
        self.core
            .text
            .emit_u32(enc::stp_64_pre_index(x29, x30, abi::stack_reg(), -16));
    }

    /// Body-local error tail (`body_local_error_label`). Reached from trap
    /// stubs and post-BL status checks. Pops the body prelude link save
    /// and the caller's call record, restores `fp_reg`, and `ret`s without
    /// touching `C_RET0` (which already holds the trap kind set by the
    /// trap stub or inherited from a trapped descendant's BL).
    /// Flush deferred per-call literals into a pool at end-of-body. Each
    /// literal is an 8-byte zero word (patched at module-link time via the
    /// `direct_call_patches` entry we push here) and the corresponding
    /// `ldr_lit_64` instruction back in the body block is patched to point
    /// at it. This is what makes the trailing-`b`-elision in
    /// `lower_call_direct` correct: by the time the call site falls
    /// through to the next emitted block, no inline literal sits in the
    /// fall-through path because every literal lives in this pool.
    ///
    /// `ldr_lit_64`'s pc-relative immediate is a 19-bit signed
    /// instruction-word offset (±1 MiB byte reach). The pool is placed at
    /// the end of the body region so the LDR-to-literal distance is
    /// `(pool_offset - call_site_offset)` — bounded by body+edges size.
    /// We validate the delta here and fail compilation if it ever exceeds
    /// range, rather than letting `enc::ldr_lit_64`'s `& 0x0007_FFFF`
    /// silently truncate the immediate and produce a wrong call target.
    fn lower_function_literal_pool(&mut self) -> Result<(), WasmError> {
        let pending = core::mem::take(&mut self.pending_call_literals);
        for literal in pending {
            let literal_offset = self.core.text.emit_u64(0);
            let delta_bytes = literal_offset as isize - literal.ldr_offset as isize;
            // ldr_lit_64 encodes a signed 19-bit instruction-word offset.
            // Word range: [-2^18, 2^18 - 1]. Byte range: ±1 MiB.
            // The literal pool always sits *after* the LDR (delta is
            // positive in practice), but the bound is checked symmetric.
            const LDR_LIT_BYTE_MAX: isize = (1 << 20) - 4;
            if delta_bytes & 0b11 != 0 {
                return Err(WasmError::internal(
                    "arm64 deferred call literal is not 4-byte aligned",
                ));
            }
            if !(-LDR_LIT_BYTE_MAX..=LDR_LIT_BYTE_MAX).contains(&delta_bytes) {
                return Err(WasmError::internal(
                    "arm64 deferred call literal pool is out of range",
                ));
            }
            let delta_words = (delta_bytes / 4) as i32;
            self.core.text.patch_u32(
                literal.ldr_offset,
                enc::ldr_lit_64(literal.scratch_reg, delta_words),
            );
            self.core
                .direct_call_patches
                .push(crate::vm::arch::common::types::DirectCallPatch {
                    literal_offset,
                    callee: literal.callee,
                });
        }
        Ok(())
    }

    fn lower_body_local_error_tail(&mut self) {
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();
        // ldp x29, x30, [sp], #16    ; pop body prelude link save
        self.core
            .text
            .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));
        // Pop the call record. The error path does not copy results, so
        // we only need to pull caller_fp out (slot 1) and discard
        // caller_result_base (slot 0).
        let scratch_idx = self.gp_scratch.alloc();
        let scratch_fp = self.gp_scratch.reg(scratch_idx);
        self.core
            .text
            .emit_u32(enc::ldr_64(scratch_fp, abi::stack_reg(), 1));
        self.core
            .text
            .emit_u32(enc::add_imm_64(abi::stack_reg(), abi::stack_reg(), 16));
        // Restore caller fp_reg.
        let fp_reg = abi::map_fixed_reg(MACHINE_FP_REG);
        self.core.text.emit_u32(enc::mov_reg_64(fp_reg, scratch_fp));
        // C_RET0 untouched — preserves the error code.
        self.core.text.emit_u32(enc::ret());
        self.gp_scratch.free_index(scratch_idx);
    }

    fn begin_block(&mut self, block: &MachineBlock) -> Result<(), WasmError> {
        // Streaming entry: clear the lookahead and reset block state.
        self.pending_op = None;
        self.core.current_block = Some(block.id);
        self.core.current_edge_target = None;
        self.core.reset_block_fp_state(block)?;
        Ok(())
    }

    // Note on dropped burst fusion: a previous experiment grouped consecutive
    // IndexedLoad/IndexedStore ops sharing (base, index, extend) into one
    // shared `add Xs, Xb, Wi, UXTW` plus N immediate-offset loads/stores.
    // It looks like an obvious win but is measurably slower on Apple Silicon
    // (M-series) because that core macro-fuses `add x, x, #imm` with the
    // following `ldr w, [base, x]` into a single AGU op, and `mov w, w` is a
    // zero-latency rename. The 3-instruction sequence `mov + add + ldr-reg`
    // effectively executes as a single load there; the burst form's
    // `add x, base, w_idx, UXTW` is not AGU-fusable so its N dependent loads
    // pay an extra cycle of critical-path latency. See `try_lower_indexed_burst`
    // in inst.rs (kept under `#[allow(dead_code)]`) for the experiment and the
    // measured numbers.
    fn emit_inst_at(&mut self, inst: &MachineInst, index: usize) -> Result<(), WasmError> {
        // Try to fuse the previously buffered op with this incoming op. On a
        // hit, both are consumed by a single emitted instruction. On a miss,
        // emit the buffered op solo and buffer the new one for the next call.
        if let Some((prev, prev_index)) = self.pending_op.take() {
            self.core.current_op_index = Some(prev_index);
            if let Some((base, imm7)) = super::fusion::zero_store_pair_fusion(&prev, inst) {
                let base_reg = self.map_gp_reg(base)?;
                self.core.text.emit_u32(enc::stp_zero_64(base_reg, imm7));
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                return Ok(());
            }
            if self.try_lower_fp_pair(&prev, inst)? {
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                return Ok(());
            }
            // No fusion — emit prev solo before buffering the new op.
            self.lower_inst(&prev)?;
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
        }
        self.pending_op = Some((inst.clone(), index));
        Ok(())
    }

    fn end_block(
        &mut self,
        term: &MachineTerminator,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        // Drain any final buffered op before the terminator; nothing else can
        // arrive to fuse with it.
        if let Some((prev, prev_index)) = self.pending_op.take() {
            self.core.current_op_index = Some(prev_index);
            self.lower_inst(&prev)?;
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
        }
        self.core.current_op_index = None;
        let result = self.lower_terminator(term, fallthrough);
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
        // arm64 PC-relative branch reach in instruction words:
        //   b / bl              imm26 signed → ±2^25 words = ±128 MiB bytes
        //   b.cond / cbz / cbnz imm19 signed → ±2^18 words = ±1 MiB bytes
        // The encoders mask the immediate silently (`& 0x...`), so we
        // must validate the delta here or out-of-range targets become
        // wrong addresses at runtime. None of the current fixups should
        // ever exceed these bounds (functions are far smaller than
        // 1 MiB), but failing fast is the only safe option until a
        // veneer/island pass is added.
        const IMM19_WORD_MIN: isize = -(1 << 18);
        const IMM19_WORD_MAX: isize = (1 << 18) - 1;
        const IMM26_WORD_MIN: isize = -(1 << 25);
        const IMM26_WORD_MAX: isize = (1 << 25) - 1;
        for fixup in &self.fixups {
            let target = self
                .core
                .labels
                .get(fixup.label)
                .and_then(|v| *v)
                .ok_or_else(|| WasmError::internal("arm64 branch target label is unresolved"))?;
            let delta_bytes = (target as isize) - (fixup.inst_offset as isize);
            if delta_bytes & 0b11 != 0 {
                return Err(WasmError::internal(
                    "arm64 branch fixup target is not 4-byte aligned (inst at )",
                ));
            }
            let delta_words = delta_bytes / 4;
            let (_kind_name, in_range) = match fixup.kind {
                BranchFixupKind::B => (
                    "b",
                    (IMM26_WORD_MIN..=IMM26_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::Bl => (
                    "bl",
                    (IMM26_WORD_MIN..=IMM26_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::BCond(_) => (
                    "b.cond",
                    (IMM19_WORD_MIN..=IMM19_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::Cbz(_) => (
                    "cbz",
                    (IMM19_WORD_MIN..=IMM19_WORD_MAX).contains(&delta_words),
                ),
                BranchFixupKind::Cbnz(_) => (
                    "cbnz",
                    (IMM19_WORD_MIN..=IMM19_WORD_MAX).contains(&delta_words),
                ),
            };
            if !in_range {
                return Err(WasmError::internal(
                    "arm64 branch fixup is out of pc-relative range",
                ));
            }
            let delta_i32 = delta_words as i32;
            let patched = match fixup.kind {
                BranchFixupKind::B => enc::b(delta_i32),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta_i32),
                BranchFixupKind::Cbz(reg) => enc::cbz_64(reg, delta_i32),
                BranchFixupKind::Cbnz(reg) => enc::cbnz_64(reg, delta_i32),
                BranchFixupKind::Bl => enc::bl(delta_i32),
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
}

impl<'a> crate::vm::arch::shared_64::ModuleLinkBackend64<'a> for Arm64Backend<'a> {
    type CompiledEntry = CompiledArm64Entry;

    fn make_entry(
        buf: &CodeBuffer,
        emitted: &crate::vm::arch::shared_64::EmittedFunction64,
    ) -> Self::CompiledEntry {
        let entry =
            unsafe { buf.fn_ptr::<crate::vm::runtime::code::NativeRootEntry>(emitted.text_offset) };
        CompiledArm64Entry {
            entry,
            text_len: emitted.text_len,
            #[cfg(sf_ir_dump)]
            debug_regions: emitted.debug_regions.clone(),
        }
    }
}
