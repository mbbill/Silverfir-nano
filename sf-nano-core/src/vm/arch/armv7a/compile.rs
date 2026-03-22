//! ARMv7-A backend: compile MachineIR to ARM32 (ARM mode) machine code.

use alloc::{vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        entities::ModuleInst,
        machine::machine_ir::{
            MachineBlockId, MachineBlockParam, MachineFuncId, MachineFunction,
            MachineFunctionRuntime, MachineReg, MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        runtime::{
            code::{Armv7aCodePtr, Armv7aRootEntry, CompiledNativeModule},
            context::ctx_offset,
        },
    },
};

use super::{
    abi::{
        emit_shared_epilogue, emit_shared_prologue, fp_machine_reg, map_fixed_reg, map_reg,
        max_total_machine_regs, FP_SCRATCH0, SCRATCH0, SCRATCH1,
    },
    armv7a_raise_trap,
    emit::Arm32TextEmitter,
    enc::{self, Cond},
    reg::Arm32Reg,
};

pub(crate) use crate::vm::debug::ir_dump::DebugRegion;

/// Patch a MOVW/MOVT pair at `movw_offset` with a 32-bit address.
/// The MOVW is at `movw_offset`, MOVT is at `movw_offset + 4`.
/// We need to know the destination register — extract it from the existing MOVW.
fn patch_movw_movt(text: &mut Arm32TextEmitter, movw_offset: usize, addr: u32) {
    // Extract Rd from existing MOVW: bits [15:12]
    let existing = u32::from_le_bytes([
        text.byte(movw_offset),
        text.byte(movw_offset + 1),
        text.byte(movw_offset + 2),
        text.byte(movw_offset + 3),
    ]);
    let rd_bits = (existing >> 12) & 0xF;
    let rd = Arm32Reg::from_idx(rd_bits);
    text.patch_u32(movw_offset, enc::movw(rd, addr as u16));
    text.patch_u32(movw_offset + 4, enc::movt(rd, (addr >> 16) as u16));
}

// ─── Label & fixup types ────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LabelKind {
    Block,
    Edge,
    StackOverflow,
    ReturnOk,
    ReturnError,
}

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

#[derive(Clone, Copy, Debug)]
struct Label {
    kind: LabelKind,
    offset: Option<usize>,
}

// ─── Patch types ────────────────────────────────────────────────────────────

/// A resolved address patch: MOVW/MOVT pair at `movw_offset` should be patched
/// with the absolute address of `target_offset` within this function's text.
#[derive(Clone, Copy, Debug)]
struct LocalPtrPatch {
    movw_offset: usize,
    target_offset: usize,
}

/// An unresolved address patch: MOVW/MOVT at `movw_offset`, target is a label.
#[derive(Clone, Copy, Debug)]
struct PendingLocalPtrPatch {
    movw_offset: usize,
    target_label: usize,
}

/// Direct call patch: MOVW/MOVT at `movw_offset` should be patched with the
/// callee's internal entry address.
#[derive(Clone, Copy, Debug)]
struct DirectCallPatch {
    movw_offset: usize,
    callee: MachineFuncId,
}

// ─── Edge stubs ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct EdgeStub {
    label: usize,
    target: MachineBlockId,
    params: Vec<MachineBlockParam>,
    args: Vec<MachineValue>,
}

#[derive(Clone, Copy)]
enum ParallelSource {
    Reg(MachineReg),
    Imm(u64),
    GpTemp,
    FpTemp,
}

// ─── Arm32FunctionInfo ──────────────────────────────────────────────────────

/// Read-only per-function metadata embedded in the code stream.
/// The entry stub loads its own address (via PC-relative) and reads these
/// fields to set up the frame.
#[repr(C)]
struct Arm32FunctionInfo {
    entry: u32,
    total_frame_bytes: u32,
    frame_prefix_slots: u32,
    call_scratch_base_slot: u32,
}

const ARM32_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<Arm32FunctionInfo>();

/// Intermediate result of compiling one function (before final CodeBuffer write).
#[derive(Debug)]
struct FunctionArtifact {
    text: Arm32TextEmitter,
    local_ptr_patches: Vec<LocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    root_return_offset: usize,
    #[cfg(has_guard_pages)]
    return_error_offset: usize,
    internal_entry_offset: usize,
    debug_regions: Vec<DebugRegion>,
}

/// Result of compiling one function to ARM32 machine code.
#[derive(Clone, Debug)]
pub struct CompiledArm32Entry {
    pub entry: Armv7aRootEntry,
    pub text_len: usize,
    pub debug_regions: Vec<DebugRegion>,
    pub root_return: Armv7aCodePtr,
    #[cfg(has_guard_pages)]
    pub return_error: Armv7aCodePtr,
}

// ─── FunctionCompiler ───────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct FunctionCompiler<'a> {
    pub(super) text: Arm32TextEmitter,
    pub(super) compiled: &'a CompiledNativeModule,
    pub(super) function: &'a MachineFunction,
    labels: Vec<Label>,
    fixups: Vec<BranchFixup>,
    block_labels: Vec<usize>,
    edge_stubs: Vec<EdgeStub>,
    resolved_ptr_patches: Vec<LocalPtrPatch>,
    local_ptr_patches: Vec<PendingLocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    debug_regions: Vec<DebugRegion>,
    pub(super) return_ok_label: usize,
    pub(super) return_error_label: usize,
    pub(super) stack_overflow_label: usize,
    pub(super) current_block_index: Option<u32>,
}

impl<'a> FunctionCompiler<'a> {
    fn new(compiled: &'a CompiledNativeModule, function: &'a MachineFunction) -> Self {
        let mut fc = Self {
            text: Arm32TextEmitter::new(),
            compiled,
            function,
            labels: Vec::new(),
            fixups: Vec::new(),
            block_labels: Vec::new(),
            edge_stubs: Vec::new(),
            resolved_ptr_patches: Vec::new(),
            local_ptr_patches: Vec::new(),
            direct_call_patches: Vec::new(),
            function_table_patches: Vec::new(),
            debug_regions: Vec::new(),
            return_ok_label: 0,
            return_error_label: 0,
            stack_overflow_label: 0,
            current_block_index: None,
        };
        fc.return_ok_label = fc.alloc_label(LabelKind::ReturnOk);
        fc.return_error_label = fc.alloc_label(LabelKind::ReturnError);
        fc.stack_overflow_label = fc.alloc_label(LabelKind::StackOverflow);
        fc
    }

    #[inline]
    pub(super) fn current_trap_site(&self) -> u32 {
        super::compile_helpers::encode_trap_site(self.function.id.0, self.current_block_index)
    }

    #[inline]
    pub(super) fn is_fp_machine_reg(&self, reg: MachineReg) -> bool {
        self.function.program.is_fp_reg(reg)
    }

    #[inline]
    pub(super) fn map_fp_dreg(&self, reg: MachineReg) -> Result<u32, WasmError> {
        let fp_idx = (reg.0 as usize)
            .checked_sub(self.function.program.first_fp_reg as usize)
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

    pub(super) fn alloc_label(&mut self, kind: LabelKind) -> usize {
        let id = self.labels.len();
        self.labels.push(Label { kind, offset: None });
        id
    }

    pub(super) fn bind_label(&mut self, id: usize) {
        self.labels[id].offset = Some(self.text.len());
    }

    fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.as_usize())
            .copied()
            .ok_or_else(|| WasmError::internal("armv7a block label is out of range".into()))
    }

    pub(super) fn emit_branch(&mut self, kind: BranchFixupKind, target: usize) {
        let offset = self.text.len();
        self.text.emit_u32(enc::nop());
        self.fixups.push(BranchFixup {
            offset,
            kind,
            target,
        });
    }

    fn resolve_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target_offset = self.labels[fixup.target]
                .offset
                .ok_or_else(|| WasmError::internal("unresolved branch label".into()))?;
            let delta = target_offset as i32 - fixup.offset as i32;
            let inst = match fixup.kind {
                BranchFixupKind::B => enc::b(delta),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta),
            };
            self.text.patch_u32(fixup.offset, inst);
        }
        Ok(())
    }

    /// Load a 32-bit immediate into a register using MOVW/MOVT.
    pub(super) fn emit_load_u32(&mut self, dst: Arm32Reg, value: u32) {
        if let Some((imm8, rot)) = enc::encode_arm_imm(value) {
            self.text.emit_u32(enc::mov_imm(dst, imm8, rot));
        } else {
            self.text.emit_u32(enc::movw(dst, value as u16));
            let hi = (value >> 16) as u16;
            if hi != 0 {
                self.text.emit_u32(enc::movt(dst, hi));
            }
        }
    }

    /// Load a pointer-sized absolute address into a register.
    pub(super) fn emit_load_addr(&mut self, dst: Arm32Reg, addr: usize) {
        self.emit_load_u32(dst, addr as u32);
    }

    /// Emit a MOVW/MOVT pair with placeholder zeros. Returns the offset of the
    /// MOVW instruction — used later for patching the actual address.
    fn emit_patchable_addr(&mut self, dst: Arm32Reg) -> usize {
        let offset = self.text.len();
        self.text.emit_u32(enc::movw(dst, 0));
        self.text.emit_u32(enc::movt(dst, 0));
        offset
    }

    /// Retrieve runtime metadata for a given function.
    fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionRuntime, WasmError> {
        self.compiled
            .runtime()
            .functions
            .get(func_id.0 as usize)
            .ok_or_else(|| {
                WasmError::internal(alloc::format!(
                    "armv7a runtime metadata missing for machine function {}",
                    func_id.0
                ))
            })
    }

    // ─── Edge stubs ─────────────────────────────────────────────────────

    fn is_identity_edge(&self, target: MachineBlockId, args: &[MachineValue]) -> bool {
        let Some(block) = self.function.program.blocks.get(target.as_usize()) else {
            return false;
        };
        if block.params.len() != args.len() {
            return false;
        }
        block
            .params
            .iter()
            .zip(args.iter())
            .all(|(param, arg)| matches!(arg, MachineValue::Reg(r) if *r == param.reg))
    }

    pub(super) fn emit_edge(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        if self.is_identity_edge(target, args) {
            return self.block_label(target);
        }
        self.add_edge_stub(target, args)
    }

    fn add_edge_stub(
        &mut self,
        target: MachineBlockId,
        args: &[MachineValue],
    ) -> Result<usize, WasmError> {
        let block = self
            .function
            .program
            .blocks
            .get(target.as_usize())
            .ok_or_else(|| {
                WasmError::internal("armv7a edge target block is out of range".into())
            })?;
        let label = self.alloc_label(LabelKind::Edge);
        self.edge_stubs.push(EdgeStub {
            label,
            target,
            params: block.params.clone(),
            args: args.to_vec(),
        });
        Ok(label)
    }

    fn emit_parallel_moves(
        &mut self,
        params: &[MachineBlockParam],
        args: &[MachineValue],
    ) -> Result<(), WasmError> {
        let mut pending: Vec<(MachineBlockParam, ParallelSource)> = Vec::new();
        for (&dst, &arg) in params.iter().zip(args.iter()) {
            let src = match arg {
                MachineValue::Reg(reg) => ParallelSource::Reg(reg),
                MachineValue::Imm64(value) => ParallelSource::Imm(value),
            };
            if matches!(src, ParallelSource::Reg(r) if r == dst.reg) {
                continue;
            }
            pending.push((dst, src));
        }

        while !pending.is_empty() {
            // Find a ready move (destination not used as source by others)
            let mut ready = None;
            for index in 0..pending.len() {
                let dst = pending[index].0.reg;
                let blocked = pending.iter().enumerate().any(|(other, (_, src))| {
                    other != index && matches!(src, ParallelSource::Reg(r) if *r == dst)
                });
                if !blocked {
                    ready = Some(index);
                    break;
                }
            }

            if let Some(index) = ready {
                let (dst, src) = pending.remove(index);
                self.emit_source_move(dst, src)?;
                continue;
            }

            // Cycle: save first destination to temp, break the cycle
            let (dst, src) = pending.remove(0);
            let ParallelSource::Reg(src_reg) = src else {
                self.emit_source_move(dst, src)?;
                continue;
            };

            if self.is_fp_machine_reg(dst.reg) {
                // FP cycle: save dst D-reg to FP_SCRATCH0
                let dd = self.map_fp_dreg(dst.reg)?;
                let sd = self.map_fp_dreg(src_reg)?;
                self.text.emit_u32(enc::vmov_d(FP_SCRATCH0, dd));
                self.text.emit_u32(enc::vmov_d(dd, sd));
                for (_, source) in pending.iter_mut() {
                    if matches!(*source, ParallelSource::Reg(r) if r == dst.reg) {
                        *source = ParallelSource::FpTemp;
                    }
                }
            } else {
                // GP cycle: save dst to SCRATCH0 (R12)
                let dst_gp = map_reg(dst.reg)?;
                let src_gp = map_reg(src_reg)?;
                self.text.emit_u32(enc::mov_reg(SCRATCH0, dst_gp));
                self.text.emit_u32(enc::mov_reg(dst_gp, src_gp));
                for (_, source) in pending.iter_mut() {
                    if matches!(*source, ParallelSource::Reg(r) if r == dst.reg) {
                        *source = ParallelSource::GpTemp;
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_source_move(
        &mut self,
        dst: MachineBlockParam,
        src: ParallelSource,
    ) -> Result<(), WasmError> {
        if self.is_fp_machine_reg(dst.reg) {
            let dd = self.map_fp_dreg(dst.reg)?;
            match src {
                ParallelSource::Reg(src_reg) => {
                    if self.is_fp_machine_reg(src_reg) {
                        let sd = self.map_fp_dreg(src_reg)?;
                        if dd != sd {
                            self.text.emit_u32(enc::vmov_d(dd, sd));
                        }
                    } else {
                        // GP → FP
                        let src_gp = map_reg(src_reg)?;
                        self.emit_load_u32(Arm32Reg::R1, 0);
                        self.text.emit_u32(enc::vmov_d_rr(dd, src_gp, Arm32Reg::R1));
                    }
                }
                ParallelSource::Imm(value) => {
                    let lo = value as u32;
                    let hi = (value >> 32) as u32;
                    self.emit_load_u32(Arm32Reg::R0, lo);
                    self.emit_load_u32(Arm32Reg::R1, hi);
                    self.text
                        .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                }
                ParallelSource::GpTemp => {
                    // SCRATCH0 (GP) → FP D-reg
                    self.emit_load_u32(Arm32Reg::R1, 0);
                    self.text
                        .emit_u32(enc::vmov_d_rr(dd, SCRATCH0, Arm32Reg::R1));
                }
                ParallelSource::FpTemp => {
                    self.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
                }
            }
        } else {
            let dst_gp = map_reg(dst.reg)?;
            match src {
                ParallelSource::Reg(src_reg) => {
                    if self.is_fp_machine_reg(src_reg) {
                        // FP → GP: extract low 32 bits
                        let sd = self.map_fp_dreg(src_reg)?;
                        self.text.emit_u32(enc::vmov_rr_d(dst_gp, Arm32Reg::R1, sd));
                    } else {
                        let src_gp = map_reg(src_reg)?;
                        if dst_gp != src_gp {
                            self.text.emit_u32(enc::mov_reg(dst_gp, src_gp));
                        }
                    }
                }
                ParallelSource::Imm(value) => {
                    self.emit_load_u32(dst_gp, value as u32);
                }
                ParallelSource::GpTemp => {
                    self.text.emit_u32(enc::mov_reg(dst_gp, SCRATCH0));
                }
                ParallelSource::FpTemp => {
                    // FP temp → GP: extract low 32 bits from FP_SCRATCH0
                    self.text
                        .emit_u32(enc::vmov_rr_d(dst_gp, Arm32Reg::R1, FP_SCRATCH0));
                }
            }
        }
        Ok(())
    }

    fn emit_branch_to_block(&mut self, target: MachineBlockId) -> Result<(), WasmError> {
        let label = self.block_label(target)?;
        self.emit_branch(BranchFixupKind::B, label);
        Ok(())
    }

    // ─── Call infrastructure ────────────────────────────────────────────

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
        let call_link = self.compiled.runtime().call_link;
        let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;

        let callee_fp_orig = map_reg(callee_frame_base)?;
        // Direct local calls need the callee frame base to survive stack checks
        // and call-link materialization, both of which freely clobber caller-
        // saved GP regs. Preserve it in our call-local scratch register.
        let callee_fp = SCRATCH1;
        self.text.emit_u32(enc::mov_reg(callee_fp, callee_fp_orig));

        // Store continuation address (patchable) into callee frame
        let cont_patch = self.emit_patchable_addr(SCRATCH0);
        let cont_byte_offset = (continuation_slot as i32) * 8;
        self.text
            .emit_u32(enc::str_imm(SCRATCH0, callee_fp, cont_byte_offset));
        // Also store high word as zero (continuation is a 32-bit ptr in a 64-bit slot)
        self.emit_load_u32(Arm32Reg::R3, 0);
        self.text
            .emit_u32(enc::str_imm(Arm32Reg::R3, callee_fp, cont_byte_offset + 4));

        // Load callee entry (patchable) and jump
        let callee_patch = self.emit_patchable_addr(SCRATCH0);
        self.text
            .emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.text.emit_u32(enc::bx(SCRATCH0));

        // Record patches
        let continuation_label = self.block_label(continuation)?;
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            movw_offset: cont_patch,
            target_label: continuation_label,
        });
        self.direct_call_patches.push(DirectCallPatch {
            movw_offset: callee_patch,
            callee,
        });
        Ok(())
    }

    pub(super) fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = *self.runtime_for(self.function.id)?;
        let call_scratch = runtime.call_scratch.ok_or_else(|| {
            WasmError::internal("armv7a local return requires call scratch".into())
        })?;
        let call_link = self.compiled.runtime().call_link;
        let continuation_slot = call_scratch.base_slot + (call_link.continuation_offset / 8) as u16;
        let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
        let caller_result_base_slot =
            call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;

        let fp_reg = map_fixed_reg(MACHINE_FP_REG);

        // Load continuation address
        self.text.emit_u32(enc::ldr_imm(
            SCRATCH0,
            fp_reg,
            (continuation_slot as i32) * 8,
        ));
        // Load caller FP
        self.text.emit_u32(enc::ldr_imm(
            Arm32Reg::R3,
            fp_reg,
            (caller_frame_slot as i32) * 8,
        ));
        // Load caller result base (byte offset within caller frame)
        self.text.emit_u32(enc::ldr_imm(
            Arm32Reg::R0,
            fp_reg,
            (caller_result_base_slot as i32) * 8,
        ));
        // Compute absolute result address: caller_fp + result_base
        self.text
            .emit_u32(enc::add_reg(Arm32Reg::R0, Arm32Reg::R3, Arm32Reg::R0));

        // Copy return results to caller frame
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                // Load 8-byte slot from current frame
                self.text.emit_u32(enc::ldr_imm(
                    Arm32Reg::R1,
                    fp_reg,
                    (results.base_slot as i32 + index) * 8,
                ));
                self.text
                    .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::R0, index * 8));
                // Copy high word too (64-bit slots)
                self.text.emit_u32(enc::ldr_imm(
                    Arm32Reg::R1,
                    fp_reg,
                    (results.base_slot as i32 + index) * 8 + 4,
                ));
                self.text
                    .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::R0, index * 8 + 4));
            }
        }

        // Restore FP to caller's FP and jump to continuation. Fixed mem0 regs
        // stay live across local calls; helper-time refresh is modeled by
        // shared MachineIR loads after `CallHelper`, not by return fixups here.
        self.text.emit_u32(enc::mov_reg(fp_reg, Arm32Reg::R3));
        self.text.emit_u32(enc::bx(SCRATCH0));
        Ok(())
    }

    pub(super) fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_fp_orig = map_reg(callee_frame_base)?;
        // The dynamic dispatch sequence freely reuses caller-saved GP regs
        // while resolving the table entry, so preserve the precomputed callee
        // frame base before materializing the callee id or reading the
        // function-info record.
        let callee_fp = SCRATCH1;
        self.text.emit_u32(enc::mov_reg(callee_fp, callee_fp_orig));

        // Materialize callee ID into R3
        let callee_id_reg = match callee_target {
            MachineValue::Reg(r) => map_reg(r)?,
            MachineValue::Imm64(v) => {
                self.emit_load_u32(Arm32Reg::R3, v as u32);
                Arm32Reg::R3
            }
        };
        if callee_id_reg != Arm32Reg::R3 {
            self.text
                .emit_u32(enc::mov_reg(Arm32Reg::R3, callee_id_reg));
        }

        // Load function info table base (patchable)
        let table_patch = self.emit_patchable_addr(SCRATCH0);
        self.function_table_patches.push(table_patch);

        // Each Arm32FunctionInfo is 16 bytes (4 x u32).
        // Compute entry address: table_base + callee_id * 16
        self.text
            .emit_u32(enc::lsl_imm(Arm32Reg::R3, Arm32Reg::R3, 4));
        self.text
            .emit_u32(enc::add_reg(SCRATCH0, SCRATCH0, Arm32Reg::R3));

        // Load function info fields:
        // [+0] entry (u32), [+4] total_frame_bytes (u32),
        // [+8] frame_prefix_slots (u32), [+12] call_scratch_base_slot (u32)
        self.text.emit_u32(enc::ldr_imm(Arm32Reg::R0, SCRATCH0, 0)); // entry
        self.text.emit_u32(enc::ldr_imm(Arm32Reg::R1, SCRATCH0, 4)); // total_frame_bytes
                                                                     // R2 = frame_prefix_slots (not needed for zero-out in current impl)
        self.text.emit_u32(enc::ldr_imm(Arm32Reg::R2, SCRATCH0, 12)); // call_scratch_base_slot

        // Keep the callee frame base in the call-local scratch register for
        // the whole sequence, and preserve the resolved entry address in R0
        // until the final BX.

        // Stack overflow check: callee_fp + total_frame_bytes > stack_end?
        self.text
            .emit_u32(enc::add_reg(SCRATCH0, callee_fp, Arm32Reg::R1));
        self.text.emit_u32(enc::ldr_imm(
            Arm32Reg::R3,
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::STACK_END as i32,
        ));
        self.text.emit_u32(enc::cmp_reg(SCRATCH0, Arm32Reg::R3));
        self.emit_branch(BranchFixupKind::BCond(Cond::Hi), self.stack_overflow_label);

        // Compute call_scratch absolute byte offset: call_scratch_base_slot * 8
        self.text
            .emit_u32(enc::lsl_imm(Arm32Reg::R2, Arm32Reg::R2, 3));
        self.text
            .emit_u32(enc::add_reg(Arm32Reg::R2, callee_fp, Arm32Reg::R2));

        // Store continuation address (patchable)
        let call_link = self.compiled.runtime().call_link;
        let cont_patch = self.emit_patchable_addr(SCRATCH0);
        self.text.emit_u32(enc::str_imm(
            SCRATCH0,
            Arm32Reg::R2,
            call_link.continuation_offset,
        ));
        // Zero high word of continuation (32-bit ptr in 64-bit slot)
        self.emit_load_u32(Arm32Reg::R3, 0);
        self.text.emit_u32(enc::str_imm(
            Arm32Reg::R3,
            Arm32Reg::R2,
            call_link.continuation_offset + 4,
        ));

        // Store caller FP
        self.text.emit_u32(enc::str_imm(
            map_fixed_reg(MACHINE_FP_REG),
            Arm32Reg::R2,
            call_link.caller_frame_offset,
        ));
        self.text.emit_u32(enc::str_imm(
            Arm32Reg::R3,
            Arm32Reg::R2,
            call_link.caller_frame_offset + 4,
        ));

        // Store caller result base (byte offset)
        self.emit_load_u32(SCRATCH0, u32::from(caller_result_base) * 8);
        self.text.emit_u32(enc::str_imm(
            SCRATCH0,
            Arm32Reg::R2,
            call_link.caller_result_base_offset,
        ));
        self.text.emit_u32(enc::str_imm(
            Arm32Reg::R3,
            Arm32Reg::R2,
            call_link.caller_result_base_offset + 4,
        ));

        // Set FP to callee frame base and jump to callee entry
        self.text
            .emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), callee_fp));
        self.text.emit_u32(enc::bx(Arm32Reg::R0));

        // Record continuation patch
        let continuation_label = self.block_label(continuation)?;
        self.local_ptr_patches.push(PendingLocalPtrPatch {
            movw_offset: cont_patch,
            target_label: continuation_label,
        });
        Ok(())
    }
}

// ─── Module compilation ─────────────────────────────────────────────────────

pub fn compile_module(
    module: &ModuleInst,
    compiled: &CompiledNativeModule,
) -> Result<Vec<Option<CompiledArm32Entry>>, WasmError> {
    // Pass 1: compile each function to an intermediate buffer
    let mut artifacts = Vec::with_capacity(compiled.module().functions.len());
    for function in &compiled.module().functions {
        match compile_function(compiled, function) {
            Ok(artifact) => artifacts.push(artifact),
            Err(err) => return Err(err),
        }
    }

    // Pass 2: compute base offsets in the shared CodeBuffer
    let mut base_offsets = Vec::with_capacity(artifacts.len());
    let mut running_offset = 0usize;
    for artifact in &artifacts {
        base_offsets.push(running_offset);
        running_offset = running_offset.saturating_add(artifact.text.len());
    }
    let function_info_table_offset = running_offset;

    // Build function info table
    let mut function_info_bytes =
        Vec::with_capacity(compiled.runtime().functions.len() * ARM32_FUNCTION_INFO_SIZE);
    let base_ptr = {
        let executable = module
            .native_code_buffer()
            .map_err(|err| WasmError::internal(err.into()))?;
        executable.as_ptr()
    };

    let mut internal_entry_addrs = Vec::with_capacity(artifacts.len());
    for (i, base_offset) in base_offsets.iter().enumerate() {
        internal_entry_addrs.push(unsafe {
            base_ptr.add(*base_offset + artifacts[i].internal_entry_offset)
        } as usize);
    }

    for (func_idx, runtime) in compiled.runtime().functions.iter().enumerate() {
        let info = Arm32FunctionInfo {
            entry: *internal_entry_addrs.get(func_idx).ok_or_else(|| {
                WasmError::internal("armv7a function entry is out of range".into())
            })? as u32,
            total_frame_bytes: u32::from(runtime.total_frame_slots) * 8,
            frame_prefix_slots: u32::from(runtime.frame_prefix_slots),
            call_scratch_base_slot: u32::from(
                runtime
                    .call_scratch
                    .map(|region| region.base_slot)
                    .unwrap_or(0),
            ),
        };
        function_info_bytes.extend_from_slice(&info.entry.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.call_scratch_base_slot.to_le_bytes());
    }

    // Pass 2.5: patch addresses in artifacts
    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        // Patch local pointers (continuation addresses, jump table entries)
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u32;
            patch_movw_movt(&mut artifact.text, patch.movw_offset, target_addr);
        }
        // Patch direct call targets (callee internal entry addresses)
        for patch in &artifact.direct_call_patches {
            let callee_addr = *internal_entry_addrs
                .get(patch.callee.0 as usize)
                .ok_or_else(|| {
                    WasmError::internal("armv7a direct callee address is out of range".into())
                })? as u32;
            patch_movw_movt(&mut artifact.text, patch.movw_offset, callee_addr);
        }
        // Patch function table references
        for &movw_offset in &artifact.function_table_patches {
            let table_addr = unsafe { base_ptr.add(function_info_table_offset) } as u32;
            patch_movw_movt(&mut artifact.text, movw_offset, table_addr);
        }
    }

    // Pass 3: write everything to the shared CodeBuffer
    let mut executable = module
        .native_code_buffer()
        .map_err(|err| WasmError::internal(err.into()))?;
    executable.begin_write();
    executable.reset();

    let written_start = executable.len();
    let mut entries = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        let text_bytes = artifact.text.finish();
        let text_len = text_bytes.len();
        let debug_regions = artifact.debug_regions;
        let offset = executable.emit_bytes(&text_bytes);
        let entry = unsafe { executable.fn_ptr::<Armv7aRootEntry>(offset) };
        let root_return = unsafe { executable.ptr(offset + artifact.root_return_offset) };
        #[cfg(has_guard_pages)]
        let return_error = unsafe { executable.ptr(offset + artifact.return_error_offset) };
        entries.push(Some(CompiledArm32Entry {
            entry,
            root_return,
            #[cfg(has_guard_pages)]
            return_error,
            text_len,
            debug_regions,
        }));
    }
    executable.emit_bytes(&function_info_bytes);
    let written_len = executable.len().saturating_sub(written_start);
    executable.finish_write(written_start, written_len);

    // Record profiler symbols
    let module_name = &module.name;
    for (func_idx, entry) in entries.iter().enumerate() {
        if let Some(entry) = entry {
            let func_base = entry.entry as *const u8;
            for region in &entry.debug_regions {
                if region.len > 0 {
                    let region_start = unsafe { func_base.add(region.offset) };
                    let code_bytes =
                        unsafe { core::slice::from_raw_parts(region_start, region.len) };
                    let symbol =
                        alloc::format!("jit::{}::func{}::{}", module_name, func_idx, region.label);
                    crate::vm::runtime::profiler::record_function(region_start, code_bytes, &symbol);
                }
            }
        }
    }

    // Register guard-pages JIT ranges
    #[cfg(has_guard_pages)]
    {
        let ranges: Vec<_> = entries
            .iter()
            .flatten()
            .map(|e| {
                (
                    e.entry as usize,
                    e.entry as usize + e.text_len,
                    e.return_error as usize,
                )
            })
            .collect();
        crate::vm::runtime::trap_signal::register_jit_ranges(&ranges);
    }

    Ok(entries)
}

fn compile_function(
    compiled: &CompiledNativeModule,
    function: &MachineFunction,
) -> Result<FunctionArtifact, WasmError> {
    let program = &function.program;

    let max_total_reg = max_total_machine_regs();
    if program.reg_count as usize > max_total_reg {
        return Err(WasmError::invalid(alloc::format!(
            "armv7a MachineIR backend supports at most {} machine regs, got {} in function {}",
            max_total_reg,
            program.reg_count,
            function.id.0
        )));
    }

    let mut fc = FunctionCompiler::new(compiled, function);

    // Allocate block labels
    for _ in &program.blocks {
        let label = fc.alloc_label(LabelKind::Block);
        fc.block_labels.push(label);
    }

    // ─── Shared prologue ────────────────────────────────────────────────
    let prologue_start = fc.text.len();
    emit_shared_prologue(&mut fc.text);

    // Move args: entry is `fn(ctx: *mut NativeContext, fp: *mut u64) -> u32`
    // R0 = ctx, R1 = fp → fixed CTX / FP machine regs
    fc.text
        .emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_CTX_REG), Arm32Reg::R0));
    fc.text
        .emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), Arm32Reg::R1));

    // Fixed mem0 regs are part of the backend contract, like arm64/x86_64.
    // Internal local-call entries jump past this root prologue, so callees
    // inherit the caller's live fixed regs instead of re-initializing them.
    fc.text.emit_u32(enc::ldr_imm(
        map_fixed_reg(MACHINE_MEM0_BASE_REG),
        map_fixed_reg(MACHINE_CTX_REG),
        ctx_offset::MEM0_BASE as i32,
    ));
    fc.text.emit_u32(enc::ldr_imm(
        map_fixed_reg(MACHINE_MEM0_SIZE_REG),
        map_fixed_reg(MACHINE_CTX_REG),
        ctx_offset::MEM0_SIZE as i32,
    ));

    let internal_entry_offset = fc.text.len();
    fc.debug_regions.push(DebugRegion {
        offset: prologue_start,
        len: internal_entry_offset - prologue_start,
        label: alloc::string::String::from("prologue"),
    });

    // ─── Compile blocks ─────────────────────────────────────────────────
    let block_labels_snapshot = fc.block_labels.clone();
    for (block_idx, block) in program.blocks.iter().enumerate() {
        fc.current_block_index = Some(block_idx as u32);
        fc.bind_label(block_labels_snapshot[block_idx]);

        let block_start = fc.text.len();
        for inst in &block.ops {
            super::compile_inst::compile_inst(&mut fc, inst)?;
        }
        super::compile_control::compile_terminator(&mut fc, &block.terminator)?;
        let block_end = fc.text.len();

        fc.debug_regions.push(DebugRegion {
            offset: block_start,
            len: block_end - block_start,
            label: alloc::format!("block_{}", block_idx),
        });
    }
    fc.current_block_index = None;

    // ─── Edge stubs (parallel moves for block parameters) ───────────────
    let edges = fc.edge_stubs.clone();
    for edge in edges {
        fc.bind_label(edge.label);
        fc.emit_parallel_moves(&edge.params, &edge.args)?;
        fc.emit_branch_to_block(edge.target)?;
    }

    // ─── Return OK ──────────────────────────────────────────────────────
    fc.bind_label(fc.return_ok_label);
    let return_ok_start = fc.text.len();
    fc.emit_load_u32(Arm32Reg::R0, 0);
    emit_shared_epilogue(&mut fc.text);
    fc.debug_regions.push(DebugRegion {
        offset: return_ok_start,
        len: fc.text.len() - return_ok_start,
        label: alloc::string::String::from("return_ok"),
    });
    let root_return_offset = return_ok_start;

    // ─── Return Error ───────────────────────────────────────────────────
    fc.bind_label(fc.return_error_label);
    let return_error_start = fc.text.len();
    fc.emit_load_u32(Arm32Reg::R0, 1);
    emit_shared_epilogue(&mut fc.text);
    fc.debug_regions.push(DebugRegion {
        offset: return_error_start,
        len: fc.text.len() - return_error_start,
        label: alloc::string::String::from("return_error"),
    });

    // ─── Stack overflow trampoline ──────────────────────────────────────
    fc.bind_label(fc.stack_overflow_label);
    fc.text
        .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
    fc.emit_load_u32(Arm32Reg::R1, 7);
    fc.emit_load_u32(Arm32Reg::R2, fc.current_trap_site());
    super::compile_helpers::emit_host_call(&mut fc, armv7a_raise_trap as usize);
    fc.emit_branch(BranchFixupKind::B, fc.return_error_label);

    // ─── Resolve branch fixups ──────────────────────────────────────────
    fc.resolve_fixups()?;

    // ─── Resolve pending local ptr patches ──────────────────────────────
    let mut local_ptr_patches = fc.resolved_ptr_patches;
    local_ptr_patches.reserve(fc.local_ptr_patches.len());
    for patch in fc.local_ptr_patches {
        let target_offset = fc.labels[patch.target_label].offset.ok_or_else(|| {
            WasmError::internal("armv7a local continuation label is unresolved".into())
        })?;
        local_ptr_patches.push(LocalPtrPatch {
            movw_offset: patch.movw_offset,
            target_offset,
        });
    }

    Ok(FunctionArtifact {
        text: fc.text,
        local_ptr_patches,
        direct_call_patches: fc.direct_call_patches,
        function_table_patches: fc.function_table_patches,
        root_return_offset,
        #[cfg(has_guard_pages)]
        return_error_offset: return_error_start,
        internal_entry_offset,
        debug_regions: fc.debug_regions,
    })
}
