//! ARM64 backend: struct definitions and `ArchBackend` trait implementation.
//!
//! This file is the bridge between the common pipeline and the arm64-specific
//! instruction emission. It contains only type definitions and trait glue —
//! all emission logic lives in `inst.rs` and `control.rs` as inherent methods.

use crate::collections;

#[cfg(sf_has_guard_pages)]
use crate::vm::machine::machine_ir::MachineAddr;
use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId,
            MachineFunction, MachineInst, MachineInstKind, MachineReg, MachineReturnAbi,
            MachineTerminator, MachineTrapKind, MACHINE_CTX_REG, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::{code::NativeRootEntry, code_buf::CodeBuffer, context::ctx_offset},
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
    backend::ArchBackend,
    core::{CompilerCore, FunctionBody},
    scratch_pool::ScratchPool,
    types::ParallelSource,
};

// ── Frame layout constants ───────────────────────────────────────────────────

const STACK_SLOT_BYTES: u32 = core::mem::size_of::<u64>() as u32;
#[cfg(sf_has_simd)]
const PRESERVED_DYNAMIC_FP_SLOT_BYTES: u32 = 16;
#[cfg(not(sf_has_simd))]
const PRESERVED_DYNAMIC_FP_SLOT_BYTES: u32 = STACK_SLOT_BYTES;
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

/// Deferred direct-call patch site. The hot path is a single `bl` patched at
/// module link time. A local veneer is emitted in the end-of-body pool as a
/// correctness fallback for modules whose callee lands outside arm64's ±128MiB
/// branch range.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingDirectCall {
    pub inst_offset: usize,
    pub fallback_scratch_reg: Arm64Reg,
    pub callee: MachineFuncId,
    pub link: bool,
}

#[derive(Clone, Copy, Debug)]
struct DirectCallFallbackVeneer {
    callee: MachineFuncId,
    scratch_reg: Arm64Reg,
    veneer_offset: usize,
    literal_offset: usize,
}

#[derive(Debug)]
pub(crate) struct Arm64Backend<'a> {
    pub core: CompilerCore<'a>,
    pub(super) fixups: collections::Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<Arm64Reg, 2>,
    pub(super) fp_scratch: ScratchPool<Arm64FpReg, 2>,
    /// Deferred direct-call veneers. Flushed by
    /// `lower_function_literal_pool` (which the pipeline calls between
    /// edge stubs and the body-local error tail).
    pub(super) pending_direct_calls: collections::Vec<PendingDirectCall>,
}

impl<'a> Arm64Backend<'a> {
    pub(super) fn has_body_host_frame(&self) -> bool {
        match self.core.body {
            FunctionBody::Mir(function) => arm64_function_needs_body_host_frame(function),
            FunctionBody::Template { .. } => true,
        }
    }

    pub(super) fn emit_body_returning_blr(&mut self, target: Arm64Reg) -> Result<(), WasmError> {
        if !self.has_body_host_frame() {
            return Err(WasmError::internal(
                "arm64 body-returning native call requires a body link save",
            ));
        }
        self.core.text.emit_u32(enc::blr(target));
        Ok(())
    }

    fn preserved_dynamic_body_regs(
        &self,
    ) -> (collections::Vec<Arm64Reg>, collections::Vec<Arm64FpReg>) {
        let mut gp_regs = collections::Vec::new();
        let mut fp_regs = collections::Vec::new();
        for reg in self.core.preserved_clobbers() {
            if self.core.is_fp_reg(*reg) {
                fp_regs.push(
                    self.map_fp_reg(*reg)
                        .expect("validated arm64 preserved FP clobber must map"),
                );
            } else {
                gp_regs.push(
                    abi::map_reg(*reg).expect("validated arm64 preserved GP clobber must map"),
                );
            }
        }
        (gp_regs, fp_regs)
    }

    fn lower_root_result_copy_to_public_slots(&mut self) {
        let Some(runtime) = self.core.current_runtime() else {
            return;
        };
        match &runtime.return_abi {
            MachineReturnAbi::ScalarGp { .. } => {
                self.emit_root_frame_store_slot(
                    abi::W2W_GP_RET0,
                    abi::map_fixed_reg(MACHINE_FP_REG),
                    0,
                );
                return;
            }
            MachineReturnAbi::ScalarFp { ty } => {
                let Some(width) = ty.float_width() else {
                    return;
                };
                self.emit_root_frame_store_fp_slot(
                    abi::fp_zero_reg(),
                    abi::map_fixed_reg(MACHINE_FP_REG),
                    0,
                    width,
                );
                return;
            }
            MachineReturnAbi::ScalarGpPair => {
                return;
            }
            MachineReturnAbi::None | MachineReturnAbi::FrameFallback { .. } => {}
        }
        let Some(results) = runtime.return_results else {
            return;
        };
        if results.slots == 0 || results.base_slot == 0 {
            return;
        }

        let fp_reg = abi::map_fixed_reg(MACHINE_FP_REG);
        let value_idx = self.gp_scratch.alloc();
        let value = self.gp_scratch.reg(value_idx);
        for index in 0..u32::from(results.slots) {
            let src_slot = u32::from(results.base_slot).saturating_add(index);
            self.emit_root_frame_load_slot(value, fp_reg, src_slot);
            self.emit_root_frame_store_slot(value, fp_reg, index);
        }
        self.gp_scratch.free_index(value_idx);
    }

    fn emit_root_frame_load_slot(&mut self, dst: Arm64Reg, base: Arm64Reg, slot: u32) {
        if slot < 4096 {
            self.core.text.emit_u32(enc::ldr_64(dst, base, slot));
            return;
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_u64(addr, u64::from(slot) * u64::from(STACK_SLOT_BYTES));
        self.core.text.emit_u32(enc::add_reg_64(addr, base, addr));
        self.core.text.emit_u32(enc::ldr_64(dst, addr, 0));
        self.gp_scratch.free_index(addr_idx);
    }

    fn emit_root_frame_store_slot(&mut self, src: Arm64Reg, base: Arm64Reg, slot: u32) {
        if slot < 4096 {
            self.core.text.emit_u32(enc::str_64(src, base, slot));
            return;
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_u64(addr, u64::from(slot) * u64::from(STACK_SLOT_BYTES));
        self.core.text.emit_u32(enc::add_reg_64(addr, base, addr));
        self.core.text.emit_u32(enc::str_64(src, addr, 0));
        self.gp_scratch.free_index(addr_idx);
    }

    fn emit_root_frame_store_fp_slot(
        &mut self,
        src: Arm64FpReg,
        base: Arm64Reg,
        slot: u32,
        width: MachineFloatWidth,
    ) {
        match width {
            MachineFloatWidth::F32 => {
                let scaled = slot.saturating_mul(2);
                if scaled < 4096 {
                    self.core.text.emit_u32(enc::str_s(src, base, scaled));
                    return;
                }
            }
            MachineFloatWidth::F64 => {
                if slot < 4096 {
                    self.core.text.emit_u32(enc::str_d(src, base, slot));
                    return;
                }
            }
        }
        let addr_idx = self.gp_scratch.alloc();
        let addr = self.gp_scratch.reg(addr_idx);
        self.materialize_u64(addr, u64::from(slot) * u64::from(STACK_SLOT_BYTES));
        self.core.text.emit_u32(enc::add_reg_64(addr, base, addr));
        self.core.text.emit_u32(match width {
            MachineFloatWidth::F32 => enc::str_s(src, addr, 0),
            MachineFloatWidth::F64 => enc::str_d(src, addr, 0),
        });
        self.gp_scratch.free_index(addr_idx);
    }

    fn lower_preserved_dynamic_body_save(&mut self) {
        let (gp_regs, fp_regs) = self.preserved_dynamic_body_regs();
        let gp_bytes = gp_regs.len() as u32 * STACK_SLOT_BYTES;
        let fp_offset =
            gp_bytes.div_ceil(PRESERVED_DYNAMIC_FP_SLOT_BYTES) * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let used = fp_offset + fp_regs.len() as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let total = used.div_ceil(abi::stack_alignment_bytes()) * abi::stack_alignment_bytes();
        if total == 0 {
            return;
        }
        self.core
            .text
            .emit_u32(enc::sub_imm_64(abi::stack_reg(), abi::stack_reg(), total));
        for (index, reg) in gp_regs.iter().copied().enumerate() {
            self.core
                .text
                .emit_u32(enc::str_64(reg, abi::stack_reg(), index as u32));
        }
        for (index, reg) in fp_regs.iter().copied().enumerate() {
            let byte_offset = fp_offset + index as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::str_q(reg, abi::stack_reg(), byte_offset / 16));
            #[cfg(not(sf_has_simd))]
            self.core.text.emit_u32(enc::str_d(
                reg,
                abi::stack_reg(),
                byte_offset / STACK_SLOT_BYTES,
            ));
        }
    }

    pub(super) fn lower_preserved_dynamic_body_restore(&mut self) {
        let (gp_regs, fp_regs) = self.preserved_dynamic_body_regs();
        let gp_bytes = gp_regs.len() as u32 * STACK_SLOT_BYTES;
        let fp_offset =
            gp_bytes.div_ceil(PRESERVED_DYNAMIC_FP_SLOT_BYTES) * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let used = fp_offset + fp_regs.len() as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
        let total = used.div_ceil(abi::stack_alignment_bytes()) * abi::stack_alignment_bytes();
        if total == 0 {
            return;
        }
        for (index, reg) in gp_regs.iter().copied().enumerate() {
            self.core
                .text
                .emit_u32(enc::ldr_64(reg, abi::stack_reg(), index as u32));
        }
        for (index, reg) in fp_regs.iter().copied().enumerate() {
            let byte_offset = fp_offset + index as u32 * PRESERVED_DYNAMIC_FP_SLOT_BYTES;
            #[cfg(sf_has_simd)]
            self.core
                .text
                .emit_u32(enc::ldr_q(reg, abi::stack_reg(), byte_offset / 16));
            #[cfg(not(sf_has_simd))]
            self.core.text.emit_u32(enc::ldr_d(
                reg,
                abi::stack_reg(),
                byte_offset / STACK_SLOT_BYTES,
            ));
        }
        self.core
            .text
            .emit_u32(enc::add_imm_64(abi::stack_reg(), abi::stack_reg(), total));
    }

    fn lower_direct_call_fallback_veneer(
        &mut self,
        scratch_reg: Arm64Reg,
    ) -> Result<(usize, usize), WasmError> {
        let veneer_offset = self.core.text.len();
        let ldr_offset = self.core.text.emit_u32(enc::ldr_lit_64(scratch_reg, 0));
        self.core.text.emit_u32(enc::br(scratch_reg));
        let literal_offset = self.core.text.emit_u64(0);
        let delta_bytes = literal_offset as isize - ldr_offset as isize;
        // ldr_lit_64 encodes a signed 19-bit instruction-word offset.
        // Word range: [-2^18, 2^18 - 1]. Byte range: ±1 MiB.
        // The literal pool always sits after the LDR here, but the bound is
        // checked symmetric because the encoding itself is signed.
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
        self.core
            .text
            .patch_u32(ldr_offset, enc::ldr_lit_64(scratch_reg, delta_words));
        Ok((veneer_offset, literal_offset))
    }
}

fn arm64_function_needs_body_host_frame(function: &MachineFunction) -> bool {
    if !function.preserved_clobbers.is_empty() {
        return true;
    }

    function.program.blocks.iter().any(|block| {
        block
            .ops
            .iter()
            .any(|inst| arm64_inst_lowers_to_inline_native_call(&inst.kind))
            || matches!(block.terminator, MachineTerminator::Call { .. })
    })
}

/// ARM64-only lowering effect: true when this MachineIR instruction emits a
/// native call and then resumes inside the same wasm body. Shared MachineIR
/// deliberately does not own this predicate because other backends may lower
/// the same instruction without clobbering a link register.
fn arm64_inst_lowers_to_inline_native_call(kind: &MachineInstKind) -> bool {
    match kind {
        MachineInstKind::CallRuntime(_)
        | MachineInstKind::EhThrow { .. }
        | MachineInstKind::EhThrowRef { .. }
        | MachineInstKind::EhAllocExnRef { .. }
        | MachineInstKind::MemoryGrow { .. }
        | MachineInstKind::MemoryFill { .. }
        | MachineInstKind::MemoryCopy { .. }
        | MachineInstKind::MemoryInit { .. }
        | MachineInstKind::DataDrop { .. }
        | MachineInstKind::TableGrow { .. }
        | MachineInstKind::TableFill { .. }
        | MachineInstKind::TableCopy { .. }
        | MachineInstKind::TableInit { .. }
        | MachineInstKind::ElemDrop { .. }
        | MachineInstKind::RefFunc { .. }
        | MachineInstKind::RefAsNonNull { .. }
        | MachineInstKind::RefEq { .. }
        | MachineInstKind::RefI31 { .. }
        | MachineInstKind::I31GetS { .. }
        | MachineInstKind::I31GetU { .. }
        | MachineInstKind::AnyConvertExtern { .. }
        | MachineInstKind::ExternConvertAny { .. }
        | MachineInstKind::RefTest { .. }
        | MachineInstKind::RefCast { .. }
        | MachineInstKind::StructNew { .. }
        | MachineInstKind::StructNewDefault { .. }
        | MachineInstKind::StructGet { .. }
        | MachineInstKind::StructSet { .. }
        | MachineInstKind::ArrayNew { .. }
        | MachineInstKind::ArrayNewDefault { .. }
        | MachineInstKind::ArrayNewFixed { .. }
        | MachineInstKind::ArrayNewData { .. }
        | MachineInstKind::ArrayNewElem { .. }
        | MachineInstKind::ArrayGet { .. }
        | MachineInstKind::ArraySet { .. }
        | MachineInstKind::ArrayFill { .. }
        | MachineInstKind::ArrayCopy { .. }
        | MachineInstKind::ArrayInitData { .. }
        | MachineInstKind::ArrayInitElem { .. }
        | MachineInstKind::ArrayLen { .. } => true,
        #[cfg(sf_has_simd)]
        MachineInstKind::V128FromRaw { .. } | MachineInstKind::V128ToRaw { .. } => true,
        _ => false,
    }
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

    fn new(core: CompilerCore<'a>) -> Self {
        Self {
            core,
            fixups: collections::Vec::new(),
            gp_scratch: abi::new_gp_scratch_pool(),
            fp_scratch: abi::new_fp_scratch_pool(),
            pending_direct_calls: collections::Vec::new(),
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

    /// Public-entry caller stub. Loads root register parameters and `bl`s the
    /// internal entry. The root function uses the root wasm frame directly, so
    /// successful root returns copy their return-result frame region into the
    /// public result slots at `stack_base[0..]`. `C_RET0` holds 0 or a trap
    /// kind on return.
    fn lower_root_caller_stub(&mut self) {
        self.lower_root_param_lanes_from_frame();
        // bl internal_entry_label  (resolved by patch_fixups)
        let internal_entry_label = self.core.internal_entry_label;
        self.lower_bl(internal_entry_label);

        let done = self.core.new_label();
        self.lower_cbnz(abi::C_RET0, done);
        self.lower_root_result_copy_to_public_slots();
        self.core.bind_label(done);
    }

    /// Body entry prelude. Non-leaf bodies push x29/x30 onto the host stack so
    /// nested native calls can freely clobber LR. Leaf bodies with no preserved
    /// dynamic save area skip this prelude and return using the incoming LR.
    fn lower_body_prelude(&mut self) {
        if !self.has_body_host_frame() {
            return;
        }
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();
        // stp x29, x30, [sp, #-16]!
        self.core
            .text
            .emit_u32(enc::stp_64_pre_index(x29, x30, abi::stack_reg(), -16));
        self.lower_preserved_dynamic_body_save();
    }

    /// Flush direct-call fallback veneers into the end-of-body pool. The
    /// normal direct-call patch rewrites the call-site instruction to
    /// `bl callee`; when the module linker finds that target out of range, it
    /// patches the same call-site instruction to `bl veneer` instead. The
    /// veneer loads the absolute callee address and tail-branches to it,
    /// preserving the LR produced by the original `bl`.
    fn lower_function_literal_pool(&mut self) -> Result<(), WasmError> {
        let pending = core::mem::take(&mut self.pending_direct_calls);
        let mut veneers: collections::Vec<DirectCallFallbackVeneer> = collections::Vec::new();
        for call in pending {
            let (veneer_offset, literal_offset) = if let Some(veneer) =
                veneers.iter().copied().find(|veneer| {
                    veneer.callee == call.callee && veneer.scratch_reg == call.fallback_scratch_reg
                }) {
                (veneer.veneer_offset, veneer.literal_offset)
            } else {
                let (veneer_offset, literal_offset) =
                    self.lower_direct_call_fallback_veneer(call.fallback_scratch_reg)?;
                veneers.push(DirectCallFallbackVeneer {
                    callee: call.callee,
                    scratch_reg: call.fallback_scratch_reg,
                    veneer_offset,
                    literal_offset,
                });
                (veneer_offset, literal_offset)
            };
            let patch = if call.link {
                crate::vm::arch::common::types::DirectCallPatch::arm64_bl(
                    call.inst_offset,
                    veneer_offset,
                    literal_offset,
                    call.callee,
                )
            } else {
                crate::vm::arch::common::types::DirectCallPatch::arm64_b(
                    call.inst_offset,
                    veneer_offset,
                    literal_offset,
                    call.callee,
                )
            };
            self.core.direct_call_patches.push(patch);
        }
        Ok(())
    }

    fn lower_body_local_error_tail(&mut self) {
        if !self.has_body_host_frame() {
            self.core.text.emit_u32(enc::ret());
            return;
        }
        self.lower_preserved_dynamic_body_restore();
        let x29 = abi::host_fp_reg();
        let x30 = abi::host_lr_reg();
        // ldp x29, x30, [sp], #16    ; pop body prelude link save
        self.core
            .text
            .emit_u32(enc::ldp_64_post_index(x29, x30, abi::stack_reg(), 16));
        // C_RET0 untouched — preserves the error code.
        self.core.text.emit_u32(enc::ret());
    }

    #[cfg(sf_has_guard_pages)]
    fn lower_stack_probe(&mut self, addr: MachineAddr) -> Result<(), WasmError> {
        self.emit_stack_probe(addr)
    }

    fn lower_block(
        &mut self,
        block: &MachineBlock,
        fallthrough: Option<MachineBlockId>,
    ) -> Result<(), WasmError> {
        self.core.current_block = Some(block.id);
        self.core.current_edge_target = None;
        self.core.reset_block_fp_state(block)?;

        let mut index = 0;
        while index < block.ops.len() {
            self.core.current_op_index = Some(index);
            if let Some((base, imm7)) = super::fusion::zero_store_pair_fusion(block, index) {
                let base_reg = self.map_gp_reg(base)?;
                self.core.text.emit_u32(enc::stp_zero_64(base_reg, imm7));
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                index += 2;
                continue;
            }
            // ── DO NOT re-enable a burst-fuse pass here that groups consecutive
            // IndexedLoad/Store ops sharing (base, index, extend) into a single
            // shared `add Xs, Xb, Wi, UXTW` followed by N immediate-offset
            // loads/stores. It looks like an obvious win — N loads instead of N
            // (mov + add + reg-indexed load) — but it is *measurably slower* on
            // Apple Silicon (M-series) for the same reason described in the
            // long comment in `lower_indexed_load_with_offset`.
            //
            // Measured (2026-04, M-series):
            //
            //   benchmark   burst on   burst off
            //   coremark    32044      34048    (+6.3% with burst off)
            //   c-ray       2820 ms    2789 ms  (−1.1%, lower=faster)
            //   sha256      272 MB/s   272 MB/s (neutral)
            //   bzip2       17.95 MB/s 17.94 MB/s (neutral)
            //
            // Why: M-series can macro-fuse `add x, x, #imm` with the following
            // `ldr w, [base, x]` into a single AGU op, and `mov w, w` is a
            // zero-latency rename. So the "old" 3-instruction sequence
            // `mov + add + ldr-reg` effectively executes as a single load.
            //
            // The burst form emits `add x, base, w_idx, UXTW` (not fusable with
            // the load AGU on M-series) followed by N `ldr [x, #imm]`. The N
            // loads can issue in parallel after the add, but they all wait for
            // it — adding 1 cycle of dependent latency to the whole group. The
            // raw instruction count goes down but the per-group critical-path
            // length goes up.
            //
            // The independent inner LDP/STP D pair fusion that lived inside
            // try_emit_burst_pair is gone with this change. It was the only
            // path that produced LDP/STP D for c-ray's hot loops, but the
            // measured loss on c-ray (~30 ms) is smaller than the gain on
            // coremark (~6%). For workloads that are FP-pair-heavy and not
            // integer-load-bottlenecked, a future redesign that emits LDP/STP
            // D *without* the shared add might recover that gain. Don't put it
            // back as a "burst" pattern.
            //
            // try_lower_indexed_burst and try_emit_burst_pair are kept in
            // inst.rs as dead code with `#[allow(dead_code)]` so the work is
            // documented and easy to reanimate behind a `cfg` for
            // microarchitectures where the tradeoff inverts (e.g. cores
            // without macro-op fusion / move elimination).
            // Pair-fuse consecutive Store/Load ops at adjacent offsets with
            // consecutive d-registers into stp_d / ldp_d.
            if let Some(pair_count) = self.try_lower_fp_pair(block, index)? {
                self.gp_scratch.assert_all_free();
                self.fp_scratch.assert_all_free();
                index += pair_count;
                continue;
            }
            self.lower_inst(&block.ops[index])?;
            // Catch scratch leaks between instructions.
            self.gp_scratch.assert_all_free();
            self.fp_scratch.assert_all_free();
            index += 1;
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
