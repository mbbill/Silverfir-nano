//! ARMv7-A backend: compile MachineIR to ARM32 (ARM mode) machine code.

use alloc::{vec, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        entities::ModuleInst,
        native::{
            code::{Armv7aCodePtr, Armv7aRootEntry, CompiledNativeModule},
            ir::machine::{
                MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
                MachineCompareKind, MachineConvertOp, MachineFloatBinaryOp, MachineFloatUnaryOp,
                MachineFloatWidth, MachineFuncId, MachineFunction, MachineInst, MachineInstKind,
                MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension,
                MachineMemWidth, MachineReg, MachineSign, MachineTerminator, MachineTrapKind,
                MachineValue, MACHINE_CTX_REG, MACHINE_FP_REG,
            },
            runtime::context::ctx_offset,
            runtime::helpers::resolve_helper_entry,
        },
    },
};

use super::{
    abi::{
        emit_shared_epilogue, emit_shared_prologue, fp_machine_reg, is_fp_machine_reg,
        map_fixed_reg, map_reg, max_gp_mapped_regs, max_total_machine_regs, FP_SCRATCH0, SCRATCH0,
        SCRATCH1,
    },
    armv7a_f32_trunc_i64s, armv7a_f32_trunc_i64u, armv7a_f64_trunc_i64s, armv7a_f64_trunc_i64u,
    armv7a_i64s_to_f32, armv7a_i64s_to_f64, armv7a_i64u_to_f32, armv7a_i64u_to_f64,
    armv7a_raise_trap, armv7a_sdiv, armv7a_udiv,
    emit::Arm32TextEmitter,
    enc::{self, Cond},
    reg::Arm32Reg,
};

pub use crate::vm::native::ir_dump::DebugRegion;

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
enum LabelKind {
    Block,
    Edge,
    StackOverflow,
    CallDepthExhausted,
    ReturnOk,
    ReturnError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchFixupKind {
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
struct FunctionCompiler<'a> {
    text: Arm32TextEmitter,
    compiled: &'a CompiledNativeModule,
    function: &'a MachineFunction,
    labels: Vec<Label>,
    fixups: Vec<BranchFixup>,
    block_labels: Vec<usize>,
    edge_stubs: Vec<EdgeStub>,
    resolved_ptr_patches: Vec<LocalPtrPatch>,
    local_ptr_patches: Vec<PendingLocalPtrPatch>,
    direct_call_patches: Vec<DirectCallPatch>,
    function_table_patches: Vec<usize>,
    debug_regions: Vec<DebugRegion>,
    return_ok_label: usize,
    return_error_label: usize,
    stack_overflow_label: usize,
    call_depth_label: usize,
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
            call_depth_label: 0,
        };
        fc.return_ok_label = fc.alloc_label(LabelKind::ReturnOk);
        fc.return_error_label = fc.alloc_label(LabelKind::ReturnError);
        fc.stack_overflow_label = fc.alloc_label(LabelKind::StackOverflow);
        fc.call_depth_label = fc.alloc_label(LabelKind::CallDepthExhausted);
        fc
    }

    fn alloc_label(&mut self, kind: LabelKind) -> usize {
        let id = self.labels.len();
        self.labels.push(Label { kind, offset: None });
        id
    }

    fn bind_label(&mut self, id: usize) {
        self.labels[id].offset = Some(self.text.len());
    }

    fn block_label(&self, target: MachineBlockId) -> Result<usize, WasmError> {
        self.block_labels
            .get(target.as_usize())
            .copied()
            .ok_or_else(|| WasmError::internal("armv7a block label is out of range".into()))
    }

    fn emit_branch(&mut self, kind: BranchFixupKind, target: usize) {
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
    fn emit_load_u32(&mut self, dst: Arm32Reg, value: u32) {
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
    fn emit_load_addr(&mut self, dst: Arm32Reg, addr: usize) {
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
    ) -> Result<&crate::vm::native::ir::runtime::MachineFunctionRuntime, WasmError> {
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

    fn emit_edge(
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

            if is_fp_machine_reg(dst.reg) {
                // FP cycle: save dst D-reg to FP_SCRATCH0
                let dd = map_fp_dreg(dst.reg)?;
                let sd = map_fp_dreg(src_reg)?;
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
        if is_fp_machine_reg(dst.reg) {
            let dd = map_fp_dreg(dst.reg)?;
            match src {
                ParallelSource::Reg(src_reg) => {
                    if is_fp_machine_reg(src_reg) {
                        let sd = map_fp_dreg(src_reg)?;
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
                    if is_fp_machine_reg(src_reg) {
                        // FP → GP: extract low 32 bits
                        let sd = map_fp_dreg(src_reg)?;
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

    fn emit_call_direct(
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
        let caller_frame_slot = call_scratch.base_slot + (call_link.caller_frame_offset / 8) as u16;
        let caller_result_base_slot =
            call_scratch.base_slot + (call_link.caller_result_base_offset / 8) as u16;
        let callee_total = callee_runtime.total_frame_slots;

        let callee_fp = map_reg(callee_frame_base)?;

        // Store continuation address (patchable) into callee frame
        let cont_patch = self.emit_patchable_addr(SCRATCH0);
        let cont_byte_offset = (continuation_slot as i32) * 8;
        self.text
            .emit_u32(enc::str_imm(SCRATCH0, callee_fp, cont_byte_offset));
        // Also store high word as zero (continuation is a 32-bit ptr in a 64-bit slot)
        self.emit_load_u32(Arm32Reg::R3, 0);
        self.text
            .emit_u32(enc::str_imm(Arm32Reg::R3, callee_fp, cont_byte_offset + 4));

        // Store caller FP (current FP) into callee frame
        let caller_fp_byte_offset = (caller_frame_slot as i32) * 8;
        self.text.emit_u32(enc::str_imm(
            map_fixed_reg(MACHINE_FP_REG),
            callee_fp,
            caller_fp_byte_offset,
        ));
        self.text.emit_u32(enc::str_imm(
            Arm32Reg::R3,
            callee_fp,
            caller_fp_byte_offset + 4,
        ));

        // Store caller result base (0 for direct calls — result goes to slot 0)
        let result_base_byte_offset = (caller_result_base_slot as i32) * 8;
        self.text.emit_u32(enc::str_imm(
            Arm32Reg::R3,
            callee_fp,
            result_base_byte_offset,
        ));
        self.text.emit_u32(enc::str_imm(
            Arm32Reg::R3,
            callee_fp,
            result_base_byte_offset + 4,
        ));

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

    fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
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

        // Restore FP to caller's FP and jump to continuation
        self.text.emit_u32(enc::mov_reg(fp_reg, Arm32Reg::R3));
        // Reload mem0_base/mem0_size for the caller's context
        self.text.emit_u32(enc::ldr_imm(
            Arm32Reg::R11,
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_BASE as i32,
        ));
        self.text.emit_u32(enc::ldr_imm(
            Arm32Reg::R4,
            map_fixed_reg(MACHINE_CTX_REG),
            ctx_offset::MEM0_SIZE as i32,
        ));
        self.text.emit_u32(enc::bx(SCRATCH0));
        Ok(())
    }

    fn emit_call_indirect(
        &mut self,
        callee_target: MachineValue,
        callee_frame_base: MachineReg,
        arg_slots: u16,
        caller_result_base: u16,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
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

        let callee_fp = map_reg(callee_frame_base)?;

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
                    crate::vm::native::profiler::record_function(region_start, code_bytes, &symbol);
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
        crate::vm::native::trap_signal::register_jit_ranges(&ranges);
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
    // R0 = ctx, R1 = fp → R9=CTX, R10=FP
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R9, Arm32Reg::R0));
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R10, Arm32Reg::R1));

    // Load mem0_base and mem0_size from NativeContext
    fc.text.emit_u32(enc::ldr_imm(
        Arm32Reg::R11,
        Arm32Reg::R9,
        ctx_offset::MEM0_BASE as i32,
    ));
    fc.text.emit_u32(enc::ldr_imm(
        Arm32Reg::R4,
        Arm32Reg::R9,
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
        fc.bind_label(block_labels_snapshot[block_idx]);

        let block_start = fc.text.len();
        for inst in &block.ops {
            compile_inst(&mut fc, inst)?;
        }
        compile_terminator(&mut fc, &block.terminator)?;
        let block_end = fc.text.len();

        fc.debug_regions.push(DebugRegion {
            offset: block_start,
            len: block_end - block_start,
            label: alloc::format!("block_{}", block_idx),
        });
    }

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
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
    fc.emit_load_u32(Arm32Reg::R1, 8);
    fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
    fc.text.emit_u32(enc::blx_reg(SCRATCH0));
    fc.emit_branch(BranchFixupKind::B, fc.return_error_label);

    // ─── Call depth exhausted trampoline ─────────────────────────────────
    fc.bind_label(fc.call_depth_label);
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
    fc.emit_load_u32(Arm32Reg::R1, 7);
    fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
    fc.text.emit_u32(enc::blx_reg(SCRATCH0));
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

// ─── Instruction compilation ────────────────────────────────────────────────

/// Map a MachineValue that must be a register to its ARM32 physical register.
fn map_value_reg(val: &MachineValue) -> Result<Arm32Reg, WasmError> {
    match val {
        MachineValue::Reg(r) => map_reg(*r),
        MachineValue::Imm64(_) => Err(WasmError::internal(
            "armv7a: carry/borrow/wide-mul operand must be a register, not immediate".into(),
        )),
    }
}

fn compile_inst(fc: &mut FunctionCompiler<'_>, inst: &MachineInst) -> Result<(), WasmError> {
    match &inst.kind {
        MachineInstKind::Move { dst, src } => {
            let dst_is_fp = is_fp_machine_reg(*dst);
            let src_is_fp = match src {
                MachineValue::Reg(r) => is_fp_machine_reg(*r),
                MachineValue::Imm64(_) => false,
            };

            if dst_is_fp && src_is_fp {
                // FP → FP move (D-register)
                let dd = map_fp_dreg(*dst)?;
                let dm = map_fp_dreg(match src {
                    MachineValue::Reg(r) => *r,
                    _ => unreachable!(),
                })?;
                if dd != dm {
                    fc.text.emit_u32(enc::vmov_d(dd, dm));
                }
            } else if dst_is_fp {
                // GP/Imm → FP: load to GP scratch then VMOV to D-reg
                let dd = map_fp_dreg(*dst)?;
                match src {
                    MachineValue::Reg(r) => {
                        let src_hw = map_reg(*r)?;
                        // Move GP value to low half of D-register (as 64-bit with zero-extended high)
                        fc.emit_load_u32(Arm32Reg::R1, 0);
                        fc.text.emit_u32(enc::vmov_d_rr(dd, src_hw, Arm32Reg::R1));
                    }
                    MachineValue::Imm64(imm) => {
                        let lo = *imm as u32;
                        let hi = (*imm >> 32) as u32;
                        fc.emit_load_u32(Arm32Reg::R0, lo);
                        fc.emit_load_u32(Arm32Reg::R1, hi);
                        fc.text
                            .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                    }
                }
            } else if src_is_fp {
                // FP → GP: VMOV from D-reg low word to GP
                let dst_hw = map_reg(*dst)?;
                let dm = map_fp_dreg(match src {
                    MachineValue::Reg(r) => *r,
                    _ => unreachable!(),
                })?;
                fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, dm));
                // dst_hw now has the low 32 bits
            } else {
                // GP → GP or Imm → GP
                let dst_hw = map_reg(*dst)?;
                match src {
                    MachineValue::Reg(r) => {
                        let src_hw = map_reg(*r)?;
                        if dst_hw != src_hw {
                            fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                        }
                    }
                    MachineValue::Imm64(imm) => {
                        fc.emit_load_u32(dst_hw, *imm as u32);
                    }
                }
            }
        }

        MachineInstKind::FloatConst { width, dst, bits } => {
            // Load FP constant: put bits in GP scratch, then VMOV to FP reg
            let gp_base = max_gp_mapped_regs();
            let dst_idx = (dst.0 as usize).checked_sub(gp_base).ok_or_else(|| {
                WasmError::invalid("armv7a: FloatConst dst is not an FP register".into())
            })?;
            let dd = fp_machine_reg(dst_idx).unwrap();
            match width {
                MachineFloatWidth::F32 => {
                    let lo = *bits as u32;
                    fc.emit_load_u32(SCRATCH0, lo);
                    fc.text.emit_u32(enc::vmov_s_r(dd * 2, SCRATCH0));
                }
                MachineFloatWidth::F64 => {
                    let lo = *bits as u32;
                    let hi = (*bits >> 32) as u32;
                    fc.emit_load_u32(Arm32Reg::R0, lo);
                    fc.emit_load_u32(Arm32Reg::R1, hi);
                    fc.text
                        .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                }
            }
        }

        MachineInstKind::Lea { dst, addr } => {
            let dst_hw = map_reg(*dst)?;
            let base_hw = map_reg(addr.base)?;
            if addr.offset == 0 {
                if dst_hw != base_hw {
                    fc.text.emit_u32(enc::mov_reg(dst_hw, base_hw));
                }
            } else if let Some((imm8, rot)) = enc::encode_arm_imm(addr.offset as u32) {
                fc.text.emit_u32(enc::add_imm(dst_hw, base_hw, imm8, rot));
            } else {
                fc.emit_load_u32(SCRATCH0, addr.offset as u32);
                fc.text.emit_u32(enc::add_reg(dst_hw, base_hw, SCRATCH0));
            }
        }

        MachineInstKind::Load {
            dst,
            addr,
            width,
            extension,
        } => {
            compile_load(fc, *dst, addr, *width, *extension)?;
        }

        MachineInstKind::Store { addr, width, src } => {
            compile_store(fc, addr, *width, src)?;
        }

        MachineInstKind::IntBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        } => {
            compile_int_binary(fc, *width, *op, *dst, lhs, rhs)?;
        }

        MachineInstKind::IntUnary {
            width,
            op,
            dst,
            src,
        } => {
            compile_int_unary(fc, *width, *op, *dst, src)?;
        }

        MachineInstKind::IntCompare {
            width,
            kind,
            sign,
            dst,
            lhs,
            rhs,
        } => {
            compile_int_compare(fc, *width, *kind, *sign, *dst, lhs, rhs)?;
        }

        MachineInstKind::FloatBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        } => {
            compile_float_binary(fc, *width, *op, *dst, lhs, rhs)?;
        }

        MachineInstKind::FloatUnary {
            width,
            op,
            dst,
            src,
        } => {
            compile_float_unary(fc, *width, *op, *dst, src)?;
        }

        MachineInstKind::FloatCompare {
            width,
            kind,
            dst,
            lhs,
            rhs,
        } => {
            compile_float_compare(fc, *width, *kind, *dst, lhs, rhs)?;
        }

        MachineInstKind::Convert { op, dst, src } => {
            compile_convert(fc, *op, *dst, src)?;
        }

        MachineInstKind::Select {
            dst,
            on_true,
            on_false,
            cond,
        } => {
            compile_select(fc, *dst, cond, on_true, on_false)?;
        }

        MachineInstKind::TrapIf { kind, cond } => {
            compile_trap_if(fc, *kind, cond)?;
        }

        // ── 32-bit carry/borrow/wide-mul primitives ─────────────────────
        MachineInstKind::IntAddCarryOut { dst, carry_out, lhs, rhs } => {
            let dst_hw = map_reg(*dst)?;
            let carry_hw = map_reg(*carry_out)?;
            let lhs_hw = map_value_reg(lhs)?;
            let rhs_hw = map_value_reg(rhs)?;
            // ADDS dst, lhs, rhs  — sets C = carry
            fc.text.emit_u32(enc::adds_reg(dst_hw, lhs_hw, rhs_hw));
            // carry_out = C: MOV carry, #0 ; ADC carry, carry, #0
            fc.text.emit_u32(enc::mov_imm(carry_hw, 0, 0));
            fc.text.emit_u32(enc::adc_reg(carry_hw, carry_hw, carry_hw));
            // adc_reg(carry, carry, carry) = carry + carry + C = 0 + 0 + C = C
            // (carry was just set to 0, so carry + carry = 0, plus C flag)
        }
        MachineInstKind::IntAddWithCarry { dst, lhs, rhs, carry_in } => {
            let dst_hw = map_reg(*dst)?;
            let lhs_hw = map_value_reg(lhs)?;
            let rhs_hw = map_value_reg(rhs)?;
            let carry_hw = map_value_reg(carry_in)?;
            // CMP carry_in, #1  — sets C = (carry_in >= 1) = carry_in for 0/1
            fc.text.emit_u32(enc::cmp_imm(carry_hw, 1, 0));
            // ADC dst, lhs, rhs  — dst = lhs + rhs + C
            fc.text.emit_u32(enc::adc_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineInstKind::IntSubBorrowOut { dst, borrow_out, lhs, rhs } => {
            let dst_hw = map_reg(*dst)?;
            let borrow_hw = map_reg(*borrow_out)?;
            let lhs_hw = map_value_reg(lhs)?;
            let rhs_hw = map_value_reg(rhs)?;
            // SUBS dst, lhs, rhs  — C = NOT(borrow)
            fc.text.emit_u32(enc::subs_reg(dst_hw, lhs_hw, rhs_hw));
            // Extract borrow = NOT(C):
            // MOV borrow, #0; ADC borrow, borrow, borrow → borrow = NOT(borrow)
            fc.text.emit_u32(enc::mov_imm(borrow_hw, 0, 0));
            fc.text.emit_u32(enc::adc_reg(borrow_hw, borrow_hw, borrow_hw));
            // Now borrow_hw = C = NOT(borrow). Invert: borrow = 1 - borrow_hw
            fc.text.emit_u32(enc::rsb_imm(borrow_hw, borrow_hw, 1, 0));
        }
        MachineInstKind::IntSubWithBorrow { dst, lhs, rhs, borrow_in } => {
            let dst_hw = map_reg(*dst)?;
            let lhs_hw = map_value_reg(lhs)?;
            let rhs_hw = map_value_reg(rhs)?;
            let borrow_hw = map_value_reg(borrow_in)?;
            // dst = lhs - rhs - borrow_in (wrapping)
            // Use SCRATCH0 to avoid aliasing issues.
            fc.text.emit_u32(enc::sub_reg(SCRATCH0, lhs_hw, rhs_hw));
            fc.text.emit_u32(enc::sub_reg(dst_hw, SCRATCH0, borrow_hw));
        }
        MachineInstKind::IntMulWide { sign, dst_lo, dst_hi, lhs, rhs } => {
            let lo_hw = map_reg(*dst_lo)?;
            let hi_hw = map_reg(*dst_hi)?;
            let lhs_hw = map_value_reg(lhs)?;
            let rhs_hw = map_value_reg(rhs)?;
            match sign {
                MachineSign::Unsigned => fc.text.emit_u32(enc::umull(lo_hw, hi_hw, lhs_hw, rhs_hw)),
                MachineSign::Signed => fc.text.emit_u32(enc::smull(lo_hw, hi_hw, lhs_hw, rhs_hw)),
            }
        }

        MachineInstKind::CallHelper(call) => {
            compile_call_helper(fc, call)?;
        }
    }
    Ok(())
}

// ─── Load/Store helpers ─────────────────────────────────────────────────────

fn compile_load(
    fc: &mut FunctionCompiler<'_>,
    dst: MachineReg,
    addr: &MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
) -> Result<(), WasmError> {
    let base_hw = map_reg(addr.base)?;
    let offset = addr.offset;

    // FP destination: use VLDR
    if is_fp_machine_reg(dst) {
        let dd = map_fp_dreg(dst)?;
        match width {
            MachineMemWidth::U64 => {
                // VLDR.64 Dd, [base, #offset]
                // VLDR offset must be word-aligned and in range ±1020
                if offset >= 0 && (offset % 4) == 0 && offset <= 1020 {
                    fc.text.emit_u32(enc::vldr_d(dd, base_hw, offset));
                } else {
                    // Compute effective address in scratch, then VLDR with offset 0
                    fc.emit_load_u32(SCRATCH0, offset as u32);
                    fc.text.emit_u32(enc::add_reg(SCRATCH0, base_hw, SCRATCH0));
                    fc.text.emit_u32(enc::vldr_d(dd, SCRATCH0, 0));
                }
            }
            MachineMemWidth::U32 => {
                // VLDR.32 Sd, [base, #offset] — use S-register = dd*2
                if offset >= 0 && (offset % 4) == 0 && offset <= 1020 {
                    fc.text.emit_u32(enc::vldr_s(dd * 2, base_hw, offset));
                } else {
                    fc.emit_load_u32(SCRATCH0, offset as u32);
                    fc.text.emit_u32(enc::add_reg(SCRATCH0, base_hw, SCRATCH0));
                    fc.text.emit_u32(enc::vldr_s(dd * 2, SCRATCH0, 0));
                }
            }
            _ => {
                return Err(WasmError::invalid(alloc::format!(
                    "armv7a: unsupported FP load width {:?}",
                    width
                )));
            }
        }
        return Ok(());
    }

    // GP destination: use LDR/LDRB/LDRH etc.
    let dst_hw = map_reg(dst)?;
    match width {
        MachineMemWidth::U8 => match extension {
            MachineLoadExtension::SignExtend => {
                fc.text.emit_u32(enc::ldrsb_imm(dst_hw, base_hw, offset));
            }
            _ => {
                fc.text.emit_u32(enc::ldrb_imm(dst_hw, base_hw, offset));
            }
        },
        MachineMemWidth::U16 => match extension {
            MachineLoadExtension::SignExtend => {
                fc.text.emit_u32(enc::ldrsh_imm(dst_hw, base_hw, offset));
            }
            _ => {
                fc.text.emit_u32(enc::ldrh_imm(dst_hw, base_hw, offset));
            }
        },
        MachineMemWidth::U32 => {
            fc.text.emit_u32(enc::ldr_imm(dst_hw, base_hw, offset));
        }
        MachineMemWidth::U64 => {
            // 64-bit load to GP: load low 32 bits only
            fc.text.emit_u32(enc::ldr_imm(dst_hw, base_hw, offset));
        }
    }
    Ok(())
}

fn compile_store(
    fc: &mut FunctionCompiler<'_>,
    addr: &MachineAddr,
    width: MachineMemWidth,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let base_hw = map_reg(addr.base)?;
    let offset = addr.offset;

    // Check if source is an FP register
    if let MachineValue::Reg(r) = src {
        if is_fp_machine_reg(*r) {
            let dd = map_fp_dreg(*r)?;
            match width {
                MachineMemWidth::U64 => {
                    if offset >= 0 && (offset % 4) == 0 && offset <= 1020 {
                        fc.text.emit_u32(enc::vstr_d(dd, base_hw, offset));
                    } else {
                        fc.emit_load_u32(SCRATCH0, offset as u32);
                        fc.text.emit_u32(enc::add_reg(SCRATCH0, base_hw, SCRATCH0));
                        fc.text.emit_u32(enc::vstr_d(dd, SCRATCH0, 0));
                    }
                }
                MachineMemWidth::U32 => {
                    if offset >= 0 && (offset % 4) == 0 && offset <= 1020 {
                        fc.text.emit_u32(enc::vstr_s(dd * 2, base_hw, offset));
                    } else {
                        fc.emit_load_u32(SCRATCH0, offset as u32);
                        fc.text.emit_u32(enc::add_reg(SCRATCH0, base_hw, SCRATCH0));
                        fc.text.emit_u32(enc::vstr_s(dd * 2, SCRATCH0, 0));
                    }
                }
                _ => {
                    return Err(WasmError::invalid(alloc::format!(
                        "armv7a: unsupported FP store width {:?}",
                        width
                    )));
                }
            }
            return Ok(());
        }
    }

    // GP source
    let src_hw = match src {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(imm) => {
            fc.emit_load_u32(SCRATCH0, *imm as u32);
            SCRATCH0
        }
    };

    match width {
        MachineMemWidth::U8 => {
            fc.text.emit_u32(enc::strb_imm(src_hw, base_hw, offset));
        }
        MachineMemWidth::U16 => {
            fc.text.emit_u32(enc::strh_imm(src_hw, base_hw, offset));
        }
        MachineMemWidth::U32 => {
            fc.text.emit_u32(enc::str_imm(src_hw, base_hw, offset));
        }
        MachineMemWidth::U64 => {
            // Store low word, then zero high word
            fc.text.emit_u32(enc::str_imm(src_hw, base_hw, offset));
            fc.emit_load_u32(SCRATCH0, 0);
            fc.text
                .emit_u32(enc::str_imm(SCRATCH0, base_hw, offset + 4));
        }
    }
    Ok(())
}

// ─── Integer ALU ────────────────────────────────────────────────────────────

fn compile_int_binary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineIntWidth,
    op: MachineIntBinaryOp,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;

    let lhs_hw = match lhs {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(dst_hw, *v as u32);
            dst_hw
        }
    };

    match op {
        MachineIntBinaryOp::Add => match rhs {
            MachineValue::Imm64(imm) => {
                if let Some((imm8, rot)) = enc::encode_arm_imm(*imm as u32) {
                    fc.text.emit_u32(enc::add_imm(dst_hw, lhs_hw, imm8, rot));
                } else {
                    fc.emit_load_u32(SCRATCH0, *imm as u32);
                    fc.text.emit_u32(enc::add_reg(dst_hw, lhs_hw, SCRATCH0));
                }
            }
            MachineValue::Reg(r) => {
                fc.text.emit_u32(enc::add_reg(dst_hw, lhs_hw, map_reg(*r)?));
            }
        },
        MachineIntBinaryOp::Sub => match rhs {
            MachineValue::Imm64(imm) => {
                if let Some((imm8, rot)) = enc::encode_arm_imm(*imm as u32) {
                    fc.text.emit_u32(enc::sub_imm(dst_hw, lhs_hw, imm8, rot));
                } else {
                    fc.emit_load_u32(SCRATCH0, *imm as u32);
                    fc.text.emit_u32(enc::sub_reg(dst_hw, lhs_hw, SCRATCH0));
                }
            }
            MachineValue::Reg(r) => {
                fc.text.emit_u32(enc::sub_reg(dst_hw, lhs_hw, map_reg(*r)?));
            }
        },
        MachineIntBinaryOp::Mul => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::mul(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::And => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::and_imm(dst_hw, lhs_hw, imm8, rot));
                        return Ok(());
                    }
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::and_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::Or => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::orr_imm(dst_hw, lhs_hw, imm8, rot));
                        return Ok(());
                    }
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::orr_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::Xor => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::eor_imm(dst_hw, lhs_hw, imm8, rot));
                        return Ok(());
                    }
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::eor_reg(dst_hw, lhs_hw, rhs_hw));
        }
        MachineIntBinaryOp::Shl => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::lsl_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            // Mask shift amount to 5 bits (wasm i32 semantics)
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::lsl_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::ShrU => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::lsr_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::lsr_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::ShrS => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::asr_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::asr_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::Rotl => {
            // rotl(x, k) = rotr(x, 32-k)
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (32 - ((*v as u32) & 31)) & 31;
                    fc.text.emit_u32(enc::ror_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::rsb_imm(SCRATCH0, SCRATCH0, 32, 0));
            fc.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::Rotr => {
            let rhs_hw = match rhs {
                MachineValue::Imm64(v) => {
                    let shift = (*v as u32) & 31;
                    fc.text.emit_u32(enc::ror_imm(dst_hw, lhs_hw, shift));
                    return Ok(());
                }
                MachineValue::Reg(r) => map_reg(*r)?,
            };
            fc.text.emit_u32(enc::and_imm(SCRATCH0, rhs_hw, 31, 0));
            fc.text.emit_u32(enc::ror_reg(dst_hw, lhs_hw, SCRATCH0));
        }
        MachineIntBinaryOp::DivU => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(rhs_hw, 0, 0));
            let ok = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
            fc.emit_load_u32(Arm32Reg::R1, 5); // IntegerDivideByZero
            fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            fc.emit_load_u32(Arm32Reg::R0, 1);
            emit_shared_epilogue(&mut fc.text);
            fc.bind_label(ok);
            // Call armv7a_udiv(num, den) -> quotient in R0
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, lhs_hw));
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, rhs_hw));
            fc.emit_load_addr(SCRATCH0, armv7a_udiv as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
        }
        MachineIntBinaryOp::DivS => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(rhs_hw, 0, 0));
            let not_zero = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_zero);
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
            fc.emit_load_u32(Arm32Reg::R1, 5);
            fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            fc.emit_load_u32(Arm32Reg::R0, 1);
            emit_shared_epilogue(&mut fc.text);
            fc.bind_label(not_zero);
            // Trap on INT_MIN / -1 (integer overflow)
            fc.emit_load_u32(SCRATCH0, 0x80000000u32);
            fc.text.emit_u32(enc::cmp_reg(lhs_hw, SCRATCH0));
            let not_overflow = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_overflow);
            fc.text.emit_u32(enc::cmn_imm(rhs_hw, 1, 0)); // CMN rhs, #1 == CMP rhs, #-1
            let not_overflow2 = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), not_overflow2);
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
            fc.emit_load_u32(Arm32Reg::R1, 6); // IntegerOverflow
            fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            fc.emit_load_u32(Arm32Reg::R0, 1);
            emit_shared_epilogue(&mut fc.text);
            fc.bind_label(not_overflow);
            fc.bind_label(not_overflow2);
            // Call armv7a_sdiv(num, den)
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, lhs_hw));
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, rhs_hw));
            fc.emit_load_addr(SCRATCH0, armv7a_sdiv as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
        }
        MachineIntBinaryOp::RemU => {
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(rhs_hw, 0, 0));
            let ok = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
            fc.emit_load_u32(Arm32Reg::R1, 5);
            fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            fc.emit_load_u32(Arm32Reg::R0, 1);
            emit_shared_epilogue(&mut fc.text);
            fc.bind_label(ok);
            // rem = lhs - (lhs / rhs) * rhs
            fc.text
                .emit_u32(enc::push((1 << lhs_hw.idx()) | (1 << rhs_hw.idx())));
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, lhs_hw));
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, rhs_hw));
            fc.emit_load_addr(SCRATCH0, armv7a_udiv as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            // R0 = quotient. Restore lhs, rhs
            fc.text.emit_u32(enc::pop((1 << 2) | (1 << 3))); // R2=lhs, R3=rhs
            fc.text
                .emit_u32(enc::mul(SCRATCH0, Arm32Reg::R0, Arm32Reg::R3));
            fc.text
                .emit_u32(enc::sub_reg(dst_hw, Arm32Reg::R2, SCRATCH0));
        }
        MachineIntBinaryOp::RemS => {
            // Similar to RemU but signed, and no overflow trap for remainder
            let rhs_hw = match rhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // Trap on divide by zero
            fc.text.emit_u32(enc::cmp_imm(rhs_hw, 0, 0));
            let ok = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), ok);
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
            fc.emit_load_u32(Arm32Reg::R1, 5);
            fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            fc.emit_load_u32(Arm32Reg::R0, 1);
            emit_shared_epilogue(&mut fc.text);
            fc.bind_label(ok);
            // INT_MIN % -1 == 0 in wasm (no trap, just returns 0)
            // rem = num - (num / den) * den — this naturally gives 0
            // Save lhs and rhs, call sdiv, compute remainder
            fc.text
                .emit_u32(enc::push((1 << lhs_hw.idx()) | (1 << rhs_hw.idx())));
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, lhs_hw));
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, rhs_hw));
            fc.emit_load_addr(SCRATCH0, armv7a_sdiv as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            // R0 = quotient. Restore lhs, rhs
            fc.text.emit_u32(enc::pop((1 << 2) | (1 << 3))); // R2=lhs, R3=rhs
                                                             // rem = lhs - quotient * rhs: MLS dst, R0, R3, R2
            fc.text
                .emit_u32(enc::mul(SCRATCH0, Arm32Reg::R0, Arm32Reg::R3));
            fc.text
                .emit_u32(enc::sub_reg(dst_hw, Arm32Reg::R2, SCRATCH0));
        }
    }

    // For I32 width, mask to 32 bits (already natural on ARM32)
    // For I64 width, we only handle low 32 bits currently
    Ok(())
}

fn compile_int_unary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineIntWidth,
    op: MachineIntUnaryOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;
    let src_hw = match src {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(dst_hw, *v as u32);
            dst_hw
        }
    };

    match op {
        MachineIntUnaryOp::Eqz => {
            // dst = (src == 0) ? 1 : 0
            fc.emit_load_u32(dst_hw, 0);
            fc.text.emit_u32(enc::cmp_imm(src_hw, 0, 0));
            // MOV{EQ} dst, #1
            let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
            fc.text.emit_u32(enc::dp_imm_cond(
                Cond::Eq,
                0b1101,
                false,
                dst_hw,
                Arm32Reg::R0,
                imm8,
                rot,
            ));
        }
        MachineIntUnaryOp::Clz => {
            fc.text.emit_u32(enc::clz(dst_hw, src_hw));
        }
        MachineIntUnaryOp::Ctz => {
            // ctz(x) = 31 - clz(x & -x) when x != 0, else 32
            // RBIT + CLZ on ARMv7
            // Actually ARMv7 has RBIT: reverse bits, then CLZ
            fc.text.emit_u32(rbit(dst_hw, src_hw));
            fc.text.emit_u32(enc::clz(dst_hw, dst_hw));
        }
        MachineIntUnaryOp::Popcnt => {
            // Hamming weight using parallel bit counting
            // x = x - ((x >> 1) & 0x55555555)
            fc.text.emit_u32(enc::lsr_imm(SCRATCH0, src_hw, 1));
            fc.emit_load_u32(Arm32Reg::R3, 0x55555555);
            fc.text
                .emit_u32(enc::and_reg(SCRATCH0, SCRATCH0, Arm32Reg::R3));
            fc.text.emit_u32(enc::sub_reg(dst_hw, src_hw, SCRATCH0));
            // x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
            fc.emit_load_u32(Arm32Reg::R3, 0x33333333);
            fc.text.emit_u32(enc::lsr_imm(SCRATCH0, dst_hw, 2));
            fc.text
                .emit_u32(enc::and_reg(SCRATCH0, SCRATCH0, Arm32Reg::R3));
            fc.text.emit_u32(enc::and_reg(dst_hw, dst_hw, Arm32Reg::R3));
            fc.text.emit_u32(enc::add_reg(dst_hw, dst_hw, SCRATCH0));
            // x = (x + (x >> 4)) & 0x0F0F0F0F
            fc.text.emit_u32(enc::lsr_imm(SCRATCH0, dst_hw, 4));
            fc.text.emit_u32(enc::add_reg(dst_hw, dst_hw, SCRATCH0));
            fc.emit_load_u32(Arm32Reg::R3, 0x0F0F0F0F);
            fc.text.emit_u32(enc::and_reg(dst_hw, dst_hw, Arm32Reg::R3));
            // x = x * 0x01010101 >> 24
            fc.emit_load_u32(Arm32Reg::R3, 0x01010101);
            fc.text.emit_u32(enc::mul(dst_hw, dst_hw, Arm32Reg::R3));
            fc.text.emit_u32(enc::lsr_imm(dst_hw, dst_hw, 24));
        }
        MachineIntUnaryOp::Extend8S => {
            fc.text.emit_u32(enc::sxtb(dst_hw, src_hw));
        }
        MachineIntUnaryOp::Extend16S => {
            fc.text.emit_u32(enc::sxth(dst_hw, src_hw));
        }
        MachineIntUnaryOp::Extend32S => {
            // On 32-bit, this is a no-op (value is already 32 bits)
            if dst_hw != src_hw {
                fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
            }
        }
    }
    Ok(())
}

/// RBIT Rd, Rm (reverse bits, ARMv6T2+)
fn rbit(dst: Arm32Reg, src: Arm32Reg) -> u32 {
    // RBIT: cond 0110 1111 1111 Rd 1111 0011 Rm
    enc::cond_bits(Cond::Al)
        | (0b01101111 << 20)
        | (0b1111 << 16)
        | ((dst.idx()) << 12)
        | (0b11110011 << 4)
        | src.idx()
}

// ─── Float ALU ──────────────────────────────────────────────────────────────

fn fp_reg_from_value(val: &MachineValue) -> Result<MachineReg, WasmError> {
    match val {
        MachineValue::Reg(r) => Ok(*r),
        MachineValue::Imm64(_) => Err(WasmError::invalid(
            "armv7a: float op requires register, got immediate".into(),
        )),
    }
}

/// Map a machine FP register to its VFP D-register number.
fn map_fp_dreg(reg: MachineReg) -> Result<u32, WasmError> {
    let gp_base = max_gp_mapped_regs();
    let fp_idx = (reg.0 as usize).checked_sub(gp_base).ok_or_else(|| {
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

fn compile_float_binary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    op: MachineFloatBinaryOp,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let lhs_reg = fp_reg_from_value(lhs)?;
    let rhs_reg = fp_reg_from_value(rhs)?;

    let gp_base = max_gp_mapped_regs();
    let dst_idx = (dst.0 as usize).checked_sub(gp_base).ok_or_else(|| {
        WasmError::invalid("armv7a: float op destination is not an FP register".into())
    })?;
    let lhs_idx = (lhs_reg.0 as usize).checked_sub(gp_base).ok_or_else(|| {
        WasmError::invalid("armv7a: float op source is not an FP register".into())
    })?;
    let rhs_idx = (rhs_reg.0 as usize).checked_sub(gp_base).ok_or_else(|| {
        WasmError::invalid("armv7a: float op source is not an FP register".into())
    })?;

    let dd = fp_machine_reg(dst_idx).unwrap();
    let dn = fp_machine_reg(lhs_idx).unwrap();
    let dm = fp_machine_reg(rhs_idx).unwrap();

    match (width, op) {
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Add) => {
            fc.text.emit_u32(enc::vadd_d(dd, dn, dm));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Sub) => {
            fc.text.emit_u32(enc::vsub_d(dd, dn, dm));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Mul) => {
            fc.text.emit_u32(enc::vmul_d(dd, dn, dm));
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Div) => {
            fc.text.emit_u32(enc::vdiv_d(dd, dn, dm));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Add) => {
            fc.text.emit_u32(enc::vadd_s(dd * 2, dn * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Sub) => {
            fc.text.emit_u32(enc::vsub_s(dd * 2, dn * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Mul) => {
            fc.text.emit_u32(enc::vmul_s(dd * 2, dn * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Div) => {
            fc.text.emit_u32(enc::vdiv_s(dd * 2, dn * 2, dm * 2));
        }

        // Min/Max: compare, handle NaN, select
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Min) => {
            // wasm min: if either is NaN → NaN; min(-0,+0) → -0
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            // If unordered (NaN): result = lhs + rhs (propagates NaN)
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_d(dd, dn, dm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            // MI = lhs < rhs → lhs; GT = lhs > rhs → rhs; EQ = equal, pick rhs for -0 handling
            // Use VBSL or conditional: select lhs if MI, else rhs
            // Simplest: if lhs < rhs then dd=dn else dd=dm
            if dd != dm {
                fc.text.emit_u32(enc::vmov_d(dd, dm));
            }
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            // If MI (lhs < rhs), overwrite with lhs
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Pl), skip);
            fc.text.emit_u32(enc::vmov_d(dd, dn));
            fc.bind_label(skip);
            fc.bind_label(done);
        }
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Max) => {
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_d(dd, dn, dm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            if dd != dm {
                fc.text.emit_u32(enc::vmov_d(dd, dm));
            }
            fc.text.emit_u32(enc::vcmp_d(dn, dm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Mi), skip); // lhs < rhs → keep rhs
            fc.text.emit_u32(enc::vmov_d(dd, dn)); // lhs >= rhs → lhs
            fc.bind_label(skip);
            fc.bind_label(done);
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Min) => {
            let sdd = dd * 2;
            let sdn = dn * 2;
            let sdm = dm * 2;
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_s(sdd, sdn, sdm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            if sdd != sdm {
                fc.text.emit_u32(enc::vmov_s(sdd, sdm));
            }
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Pl), skip);
            fc.text.emit_u32(enc::vmov_s(sdd, sdn));
            fc.bind_label(skip);
            fc.bind_label(done);
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Max) => {
            let sdd = dd * 2;
            let sdn = dn * 2;
            let sdm = dm * 2;
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let no_nan = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Vc), no_nan);
            fc.text.emit_u32(enc::vadd_s(sdd, sdn, sdm));
            let done = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::B, done);
            fc.bind_label(no_nan);
            if sdd != sdm {
                fc.text.emit_u32(enc::vmov_s(sdd, sdm));
            }
            fc.text.emit_u32(enc::vcmp_s(sdn, sdm));
            fc.text.emit_u32(enc::vmrs_apsr());
            let skip = fc.alloc_label(LabelKind::Block);
            fc.emit_branch(BranchFixupKind::BCond(Cond::Mi), skip);
            fc.text.emit_u32(enc::vmov_s(sdd, sdn));
            fc.bind_label(skip);
            fc.bind_label(done);
        }

        // Copysign: take magnitude from lhs, sign from rhs
        (MachineFloatWidth::F64, MachineFloatBinaryOp::Copysign) => {
            // Extract sign bit of rhs (bit 63) into R0
            fc.text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R0, Arm32Reg::R1, dm));
            // R1 has the high word with the sign bit
            // Extract magnitude of lhs
            fc.text
                .emit_u32(enc::vmov_rr_d(Arm32Reg::R2, Arm32Reg::R3, dn));
            // Clear sign bit of lhs high word, insert sign bit from rhs
            fc.text
                .emit_u32(enc::bic_imm(Arm32Reg::R3, Arm32Reg::R3, 0x80, 2)); // BIC R3, R3, #0x80000000
            fc.text
                .emit_u32(enc::and_imm(Arm32Reg::R1, Arm32Reg::R1, 0x80, 2)); // AND R1, R1, #0x80000000
            fc.text
                .emit_u32(enc::orr_reg(Arm32Reg::R3, Arm32Reg::R3, Arm32Reg::R1));
            fc.text
                .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R2, Arm32Reg::R3));
        }
        (MachineFloatWidth::F32, MachineFloatBinaryOp::Copysign) => {
            let sdn = dn * 2;
            let sdm = dm * 2;
            let sdd = dd * 2;
            fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R0, sdn)); // lhs bits
            fc.text.emit_u32(enc::vmov_r_s(Arm32Reg::R1, sdm)); // rhs bits
            fc.text
                .emit_u32(enc::bic_imm(Arm32Reg::R0, Arm32Reg::R0, 0x80, 2)); // clear sign
            fc.text
                .emit_u32(enc::and_imm(Arm32Reg::R1, Arm32Reg::R1, 0x80, 2)); // extract sign
            fc.text
                .emit_u32(enc::orr_reg(Arm32Reg::R0, Arm32Reg::R0, Arm32Reg::R1));
            fc.text.emit_u32(enc::vmov_s_r(sdd, Arm32Reg::R0));
        }
    }
    Ok(())
}

fn compile_float_unary(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    op: MachineFloatUnaryOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    let src_reg = fp_reg_from_value(src)?;
    let gp_base = max_gp_mapped_regs();
    let dst_idx = (dst.0 as usize)
        .checked_sub(gp_base)
        .ok_or_else(|| WasmError::invalid("armv7a: float unary dst is not FP".into()))?;
    let src_idx = (src_reg.0 as usize)
        .checked_sub(gp_base)
        .ok_or_else(|| WasmError::invalid("armv7a: float unary src is not FP".into()))?;
    let dd = fp_machine_reg(dst_idx).unwrap();
    let dm = fp_machine_reg(src_idx).unwrap();

    match (width, op) {
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Abs) => {
            fc.text.emit_u32(enc::vabs_d(dd, dm));
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Neg) => {
            fc.text.emit_u32(enc::vneg_d(dd, dm));
        }
        (MachineFloatWidth::F64, MachineFloatUnaryOp::Sqrt) => {
            fc.text.emit_u32(enc::vsqrt_d(dd, dm));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Abs) => {
            fc.text.emit_u32(enc::vabs_s(dd * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Neg) => {
            fc.text.emit_u32(enc::vneg_s(dd * 2, dm * 2));
        }
        (MachineFloatWidth::F32, MachineFloatUnaryOp::Sqrt) => {
            fc.text.emit_u32(enc::vsqrt_s(dd * 2, dm * 2));
        }
        _ => {
            return Err(WasmError::invalid(alloc::format!(
                "armv7a: unsupported float unary op {:?} {:?}",
                width,
                op
            )));
        }
    }
    Ok(())
}

fn compile_float_compare(
    fc: &mut FunctionCompiler<'_>,
    width: MachineFloatWidth,
    kind: MachineCompareKind,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let lhs_reg = fp_reg_from_value(lhs)?;
    let rhs_reg = fp_reg_from_value(rhs)?;
    let dst_hw = map_reg(dst)?;
    let gp_base = max_gp_mapped_regs();
    let lhs_d = fp_machine_reg((lhs_reg.0 as usize).checked_sub(gp_base).unwrap()).unwrap();
    let rhs_d = fp_machine_reg((rhs_reg.0 as usize).checked_sub(gp_base).unwrap()).unwrap();

    match width {
        MachineFloatWidth::F64 => {
            fc.text.emit_u32(enc::vcmp_d(lhs_d, rhs_d));
        }
        MachineFloatWidth::F32 => {
            fc.text.emit_u32(enc::vcmp_s(lhs_d * 2, rhs_d * 2));
        }
    }
    fc.text.emit_u32(enc::vmrs_apsr());

    let cond = match kind {
        MachineCompareKind::Eq => Cond::Eq,
        MachineCompareKind::Ne => Cond::Ne,
        MachineCompareKind::Lt => Cond::Mi,
        MachineCompareKind::Gt => Cond::Gt,
        MachineCompareKind::Le => Cond::Ls,
        MachineCompareKind::Ge => Cond::Ge,
    };

    fc.emit_load_u32(dst_hw, 0);
    let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
    fc.text.emit_u32(enc::dp_imm_cond(
        cond,
        0b1101,
        false,
        dst_hw,
        Arm32Reg::R0,
        imm8,
        rot,
    ));
    Ok(())
}

// ─── Convert ────────────────────────────────────────────────────────────────

fn compile_convert(
    fc: &mut FunctionCompiler<'_>,
    op: MachineConvertOp,
    dst: MachineReg,
    src: &MachineValue,
) -> Result<(), WasmError> {
    match op {
        // ─── Integer wrapping/extending (GP → GP) ────────────────────────
        MachineConvertOp::I32WrapI64 => {
            let dst_hw = map_reg(dst)?;
            match src {
                MachineValue::Reg(r) => {
                    let src_hw = map_reg(*r)?;
                    if dst_hw != src_hw {
                        fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                    }
                }
                MachineValue::Imm64(v) => fc.emit_load_u32(dst_hw, *v as u32),
            }
        }
        MachineConvertOp::I64ExtendI32U | MachineConvertOp::I64ExtendI32S => {
            let dst_hw = map_reg(dst)?;
            match src {
                MachineValue::Reg(r) => {
                    let src_hw = map_reg(*r)?;
                    if dst_hw != src_hw {
                        fc.text.emit_u32(enc::mov_reg(dst_hw, src_hw));
                    }
                }
                MachineValue::Imm64(v) => fc.emit_load_u32(dst_hw, *v as u32),
            }
        }

        // ─── F64 → I32 (FP src → GP dst) ────────────────────────────────
        MachineConvertOp::I32TruncF64S => {
            // VCVT.S32.F64 Sd, Dm  then VMOV Rd, Sd
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            let dst_hw = map_reg(dst)?;
            // Use S0 (FP_SCRATCH0*2) as temp single-precision register
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_s32_d(sd_tmp, dm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }
        MachineConvertOp::I32TruncF64U => {
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_u32_d(sd_tmp, dm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }
        MachineConvertOp::I32TruncSatF64S => {
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_s32_d(sd_tmp, dm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }
        MachineConvertOp::I32TruncSatF64U => {
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_u32_d(sd_tmp, dm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }

        // ─── F32 → I32 (FP src → GP dst) ────────────────────────────────
        MachineConvertOp::I32TruncF32S => {
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2; // S-register
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_s32_s(sd_tmp, sm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }
        MachineConvertOp::I32TruncF32U => {
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_u32_s(sd_tmp, sm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }
        MachineConvertOp::I32TruncSatF32S => {
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_s32_s(sd_tmp, sm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }
        MachineConvertOp::I32TruncSatF32U => {
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            let dst_hw = map_reg(dst)?;
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vcvt_u32_s(sd_tmp, sm));
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sd_tmp));
        }

        // ─── F64/F32 → I64 (via helper call, returns low 32 bits) ──────
        MachineConvertOp::I64TruncF64S | MachineConvertOp::I64TruncSatF64S => {
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            let dst_hw = map_reg(dst)?;
            // EABI: f64 arg in D0
            if dm != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, dm));
            }
            fc.emit_load_addr(SCRATCH0, armv7a_f64_trunc_i64s as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
        }
        MachineConvertOp::I64TruncF64U | MachineConvertOp::I64TruncSatF64U => {
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            let dst_hw = map_reg(dst)?;
            if dm != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(FP_SCRATCH0, dm));
            }
            fc.emit_load_addr(SCRATCH0, armv7a_f64_trunc_i64u as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
        }
        MachineConvertOp::I64TruncF32S | MachineConvertOp::I64TruncSatF32S => {
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            let dst_hw = map_reg(dst)?;
            // EABI: f32 arg in S0
            let s0 = FP_SCRATCH0 * 2;
            if sm != s0 {
                fc.text.emit_u32(enc::vmov_s(s0, sm));
            }
            fc.emit_load_addr(SCRATCH0, armv7a_f32_trunc_i64s as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
        }
        MachineConvertOp::I64TruncF32U | MachineConvertOp::I64TruncSatF32U => {
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            let dst_hw = map_reg(dst)?;
            let s0 = FP_SCRATCH0 * 2;
            if sm != s0 {
                fc.text.emit_u32(enc::vmov_s(s0, sm));
            }
            fc.emit_load_addr(SCRATCH0, armv7a_f32_trunc_i64u as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dst_hw != Arm32Reg::R0 {
                fc.text.emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R0));
            }
        }

        // ─── I32 → F64 (GP src → FP dst) ────────────────────────────────
        MachineConvertOp::F64ConvertI32S => {
            let dd = map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_d_s32(dd, sd_tmp));
        }
        MachineConvertOp::F64ConvertI32U => {
            let dd = map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_d_u32(dd, sd_tmp));
        }

        // ─── I32 → F32 (GP src → FP dst) ────────────────────────────────
        MachineConvertOp::F32ConvertI32S => {
            let sd = map_fp_dreg(dst)? * 2; // S-register
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_s_s32(sd, sd_tmp));
        }
        MachineConvertOp::F32ConvertI32U => {
            let sd = map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let sd_tmp = FP_SCRATCH0 * 2;
            fc.text.emit_u32(enc::vmov_s_r(sd_tmp, src_hw));
            fc.text.emit_u32(enc::vcvt_s_u32(sd, sd_tmp));
        }

        // ─── F32 ↔ F64 (FP → FP) ───────────────────────────────────────
        MachineConvertOp::F64PromoteF32 => {
            let dd = map_fp_dreg(dst)?;
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            fc.text.emit_u32(enc::vcvt_d_s(dd, sm));
        }
        MachineConvertOp::F32DemoteF64 => {
            let sd = map_fp_dreg(dst)? * 2;
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            fc.text.emit_u32(enc::vcvt_s_d(sd, dm));
        }

        // ─── I64 → F64/F32 (via helper call) ─────────────────────────────
        // On ARM32, the GP register holds the low 32 bits of the i64.
        // We sign/zero-extend from the 32-bit value to form the full i64,
        // then call a helper that does the conversion.
        MachineConvertOp::F64ConvertI64S => {
            let dd = map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // R0 = lo, R1 = hi (sign-extend: hi = lo >> 31, arithmetic shift)
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.text.emit_u32(enc::asr_imm(Arm32Reg::R1, src_hw, 31));
            fc.emit_load_addr(SCRATCH0, armv7a_i64s_to_f64 as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            // Result is in D0 (EABI: f64 returned in D0)
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        MachineConvertOp::F64ConvertI64U => {
            let dd = map_fp_dreg(dst)?;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            // R0 = lo, R1 = 0 (zero-extend)
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.emit_load_u32(Arm32Reg::R1, 0);
            fc.emit_load_addr(SCRATCH0, armv7a_i64u_to_f64 as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            if dd != FP_SCRATCH0 {
                fc.text.emit_u32(enc::vmov_d(dd, FP_SCRATCH0));
            }
        }
        MachineConvertOp::F32ConvertI64S => {
            let sd = map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.text.emit_u32(enc::asr_imm(Arm32Reg::R1, src_hw, 31));
            fc.emit_load_addr(SCRATCH0, armv7a_i64s_to_f32 as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            // Result in S0 (EABI: f32 returned in S0)
            let s0 = FP_SCRATCH0 * 2;
            if sd != s0 {
                fc.text.emit_u32(enc::vmov_s(sd, s0));
            }
        }
        MachineConvertOp::F32ConvertI64U => {
            let sd = map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, src_hw));
            fc.emit_load_u32(Arm32Reg::R1, 0);
            fc.emit_load_addr(SCRATCH0, armv7a_i64u_to_f32 as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            let s0 = FP_SCRATCH0 * 2;
            if sd != s0 {
                fc.text.emit_u32(enc::vmov_s(sd, s0));
            }
        }

        // ─── Reinterpret (bit cast, no conversion) ──────────────────────
        MachineConvertOp::I32ReinterpretF32 => {
            let dst_hw = map_reg(dst)?;
            let src_reg = fp_reg_from_value(src)?;
            let sm = map_fp_dreg(src_reg)? * 2;
            fc.text.emit_u32(enc::vmov_r_s(dst_hw, sm));
        }
        MachineConvertOp::F32ReinterpretI32 => {
            let sd = map_fp_dreg(dst)? * 2;
            let src_hw = match src {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::vmov_s_r(sd, src_hw));
        }
        MachineConvertOp::I64ReinterpretF64 => {
            // F64 (D-reg) → I64 (GP low 32 bits on ARM32)
            let dst_hw = map_reg(dst)?;
            let src_reg = fp_reg_from_value(src)?;
            let dm = map_fp_dreg(src_reg)?;
            // VMOV Rlo, Rhi, Dm — extract low 32 bits to dst
            fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, dm));
        }
        MachineConvertOp::F64ReinterpretI64 => {
            // I64 (GP low 32 bits) → F64 (D-reg)
            let dd = map_fp_dreg(dst)?;
            match src {
                MachineValue::Reg(r) => {
                    let src_hw = map_reg(*r)?;
                    fc.emit_load_u32(Arm32Reg::R1, 0);
                    fc.text.emit_u32(enc::vmov_d_rr(dd, src_hw, Arm32Reg::R1));
                }
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(Arm32Reg::R0, *v as u32);
                    fc.emit_load_u32(Arm32Reg::R1, (*v >> 32) as u32);
                    fc.text
                        .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
                }
            }
        }

        _ => {
            return Err(WasmError::invalid(alloc::format!(
                "armv7a: unsupported convert op {:?}",
                op
            )));
        }
    }
    Ok(())
}

// ─── Select ─────────────────────────────────────────────────────────────────

fn compile_select(
    fc: &mut FunctionCompiler<'_>,
    dst: MachineReg,
    condition: &MachineValue,
    true_val: &MachineValue,
    false_val: &MachineValue,
) -> Result<(), WasmError> {
    if is_fp_machine_reg(dst) {
        // FP select: use branch-based approach since ARM32 has no conditional VMOV
        let dd = map_fp_dreg(dst)?;

        // Test condition first
        let cond_hw = match condition {
            MachineValue::Reg(r) => map_reg(*r)?,
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(SCRATCH0, *v as u32);
                SCRATCH0
            }
        };
        fc.text.emit_u32(enc::cmp_imm(cond_hw, 0, 0));

        let true_label = fc.alloc_label(LabelKind::Block);
        let done_label = fc.alloc_label(LabelKind::Block);
        fc.emit_branch(BranchFixupKind::BCond(Cond::Ne), true_label);

        // False path: load false_val to dd
        match false_val {
            MachineValue::Reg(r) if is_fp_machine_reg(*r) => {
                let sd = map_fp_dreg(*r)?;
                if dd != sd {
                    fc.text.emit_u32(enc::vmov_d(dd, sd));
                }
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                fc.emit_load_u32(Arm32Reg::R1, 0);
                fc.text.emit_u32(enc::vmov_d_rr(dd, src, Arm32Reg::R1));
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(Arm32Reg::R0, *v as u32);
                fc.emit_load_u32(Arm32Reg::R1, (*v >> 32) as u32);
                fc.text
                    .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
            }
        }
        fc.emit_branch(BranchFixupKind::B, done_label);

        // True path: load true_val to dd
        fc.bind_label(true_label);
        match true_val {
            MachineValue::Reg(r) if is_fp_machine_reg(*r) => {
                let sd = map_fp_dreg(*r)?;
                if dd != sd {
                    fc.text.emit_u32(enc::vmov_d(dd, sd));
                }
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                fc.emit_load_u32(Arm32Reg::R1, 0);
                fc.text.emit_u32(enc::vmov_d_rr(dd, src, Arm32Reg::R1));
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(Arm32Reg::R0, *v as u32);
                fc.emit_load_u32(Arm32Reg::R1, (*v >> 32) as u32);
                fc.text
                    .emit_u32(enc::vmov_d_rr(dd, Arm32Reg::R0, Arm32Reg::R1));
            }
        }
        fc.bind_label(done_label);
        return Ok(());
    }

    fn gp_value_aliases_dst(
        value: &MachineValue,
        dst_hw: Arm32Reg,
    ) -> Result<bool, WasmError> {
        match value {
            MachineValue::Reg(r) if !is_fp_machine_reg(*r) => Ok(map_reg(*r)? == dst_hw),
            _ => Ok(false),
        }
    }

    fn emit_gp_select_value(
        fc: &mut FunctionCompiler<'_>,
        dst_hw: Arm32Reg,
        value: &MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) if is_fp_machine_reg(*r) => {
                let sd = map_fp_dreg(*r)?;
                fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, sd));
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                if dst_hw != src {
                    fc.text.emit_u32(enc::mov_reg(dst_hw, src));
                }
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(dst_hw, *v as u32);
            }
        }
        Ok(())
    }

    fn emit_gp_select_value_cond(
        fc: &mut FunctionCompiler<'_>,
        dst_hw: Arm32Reg,
        value: &MachineValue,
        cond: Cond,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) if is_fp_machine_reg(*r) => {
                let skip = fc.alloc_label(LabelKind::Block);
                fc.emit_branch(BranchFixupKind::BCond(cond.invert()), skip);
                let sd = map_fp_dreg(*r)?;
                fc.text.emit_u32(enc::vmov_rr_d(dst_hw, Arm32Reg::R1, sd));
                fc.bind_label(skip);
            }
            MachineValue::Reg(r) => {
                let src = map_reg(*r)?;
                fc.text.emit_u32(enc::mov_reg_cond(cond, dst_hw, src));
            }
            MachineValue::Imm64(v) => {
                fc.emit_load_u32(SCRATCH0, *v as u32);
                fc.text.emit_u32(enc::mov_reg_cond(cond, dst_hw, SCRATCH0));
            }
        }
        Ok(())
    }

    // GP select
    let dst_hw = map_reg(dst)?;

    // Test condition before touching dst so dst == cond is safe.
    let cond_hw = match condition {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(SCRATCH0, *v as u32);
            SCRATCH0
        }
    };
    fc.text.emit_u32(enc::cmp_imm(cond_hw, 0, 0));

    if gp_value_aliases_dst(true_val, dst_hw)? {
        // Loading the false arm first would clobber the live true source when
        // `dst` reuses that register. Seed `dst` with the true arm, then
        // overwrite it on the false path.
        emit_gp_select_value(fc, dst_hw, true_val)?;
        emit_gp_select_value_cond(fc, dst_hw, false_val, Cond::Eq)?;
    } else {
        emit_gp_select_value(fc, dst_hw, false_val)?;
        emit_gp_select_value_cond(fc, dst_hw, true_val, Cond::Ne)?;
    }

    Ok(())
}

// ─── IntCompare ─────────────────────────────────────────────────────────

fn compile_int_compare(
    fc: &mut FunctionCompiler<'_>,
    _width: MachineIntWidth,
    kind: MachineCompareKind,
    sign: MachineSign,
    dst: MachineReg,
    lhs: &MachineValue,
    rhs: &MachineValue,
) -> Result<(), WasmError> {
    let dst_hw = map_reg(dst)?;
    let lhs_hw = match lhs {
        MachineValue::Reg(r) => map_reg(*r)?,
        MachineValue::Imm64(v) => {
            fc.emit_load_u32(SCRATCH0, *v as u32);
            SCRATCH0
        }
    };

    match rhs {
        MachineValue::Reg(r) => {
            fc.text.emit_u32(enc::cmp_reg(lhs_hw, map_reg(*r)?));
        }
        MachineValue::Imm64(v) => {
            if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                fc.text.emit_u32(enc::cmp_imm(lhs_hw, imm8, rot));
            } else {
                let tmp = if lhs_hw == SCRATCH0 {
                    Arm32Reg::R3
                } else {
                    SCRATCH0
                };
                fc.emit_load_u32(tmp, *v as u32);
                fc.text.emit_u32(enc::cmp_reg(lhs_hw, tmp));
            }
        }
    }

    let cond = match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cond::Eq,
        (MachineCompareKind::Ne, _) => Cond::Ne,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Cc,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
        (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Cs,
    };

    fc.emit_load_u32(dst_hw, 0);
    let (imm8, rot) = enc::encode_arm_imm(1).unwrap();
    fc.text.emit_u32(enc::dp_imm_cond(
        cond,
        0b1101,
        false,
        dst_hw,
        Arm32Reg::R0,
        imm8,
        rot,
    ));
    Ok(())
}

// ─── TrapIf ─────────────────────────────────────────────────────────────

fn compile_trap_if(
    fc: &mut FunctionCompiler<'_>,
    kind: MachineTrapKind,
    cond: &MachineBranchCond,
) -> Result<(), WasmError> {
    let arm_cond = compile_branch_condition(fc, cond)?;
    // Skip trap if condition is NOT met
    let skip_label = fc.alloc_label(LabelKind::Block);
    let inv_cond = arm_cond.invert();
    fc.emit_branch(BranchFixupKind::BCond(inv_cond), skip_label);

    // Emit trap inline
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
    let trap_code = trap_kind_to_u32(kind);
    fc.emit_load_u32(Arm32Reg::R1, trap_code);
    fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
    fc.text.emit_u32(enc::blx_reg(SCRATCH0));
    fc.emit_load_u32(Arm32Reg::R0, 1);
    emit_shared_epilogue(&mut fc.text);

    fc.bind_label(skip_label);
    Ok(())
}

// ─── CallHelper ─────────────────────────────────────────────────────────

fn compile_call_helper(
    fc: &mut FunctionCompiler<'_>,
    call: &crate::vm::native::ir::machine::MachineHelperCall,
) -> Result<(), WasmError> {
    let binding = fc
        .compiled
        .module()
        .externs
        .get(call.target.0 as usize)
        .ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "armv7a: extern id {} not found",
                call.target.0
            ))
        })?;
    let metadata = fc
        .compiled
        .const_ptr(call.metadata)
        .ok_or_else(|| WasmError::internal("armv7a: helper metadata is out of range".into()))?;

    let helper_ptr = resolve_helper_entry(binding.symbol) as usize;

    // EABI: fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
    fc.text.emit_u32(enc::mov_reg(Arm32Reg::R1, Arm32Reg::R10));
    fc.emit_load_addr(Arm32Reg::R2, metadata as usize);

    fc.emit_load_addr(SCRATCH0, helper_ptr);
    fc.text.emit_u32(enc::blx_reg(SCRATCH0));

    // Check return value: if non-zero, return error
    fc.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
    let ok_label = fc.alloc_label(LabelKind::Block);
    fc.emit_branch(BranchFixupKind::BCond(Cond::Eq), ok_label);
    emit_shared_epilogue(&mut fc.text);
    fc.bind_label(ok_label);

    // Reload cached registers clobbered by the call
    fc.text.emit_u32(enc::ldr_imm(
        Arm32Reg::R11,
        Arm32Reg::R9,
        ctx_offset::MEM0_BASE as i32,
    ));
    fc.text.emit_u32(enc::ldr_imm(
        Arm32Reg::R4,
        Arm32Reg::R9,
        ctx_offset::MEM0_SIZE as i32,
    ));

    Ok(())
}

// ─── Terminator compilation ─────────────────────────────────────────────────

fn compile_terminator(
    fc: &mut FunctionCompiler<'_>,
    terminator: &MachineTerminator,
) -> Result<(), WasmError> {
    match terminator {
        MachineTerminator::Return => {
            fc.emit_return_sequence()?;
        }

        MachineTerminator::Jump(edge) => {
            let label = fc.emit_edge(edge.target, &edge.args)?;
            fc.emit_branch(BranchFixupKind::B, label);
        }

        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            let then_label = fc.emit_edge(then_edge.target, &then_edge.args)?;
            let else_label = fc.emit_edge(else_edge.target, &else_edge.args)?;

            let arm_cond = compile_branch_condition(fc, cond)?;
            fc.emit_branch(BranchFixupKind::BCond(arm_cond), then_label);
            fc.emit_branch(BranchFixupKind::B, else_label);
        }

        MachineTerminator::Trap { kind } => {
            let return_error_label = fc.return_error_label;
            fc.text.emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R9));
            let trap_code = trap_kind_to_u32(*kind);
            fc.emit_load_u32(Arm32Reg::R1, trap_code);
            fc.emit_load_addr(SCRATCH0, armv7a_raise_trap as usize);
            fc.text.emit_u32(enc::blx_reg(SCRATCH0));
            fc.emit_branch(BranchFixupKind::B, return_error_label);
        }

        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            continuation,
        } => {
            fc.emit_call_direct(*callee, *callee_frame_base, *continuation)?;
        }

        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            arg_slots,
            caller_result_base,
            continuation,
        } => {
            fc.emit_call_indirect(
                *callee_target,
                *callee_frame_base,
                *arg_slots,
                *caller_result_base,
                *continuation,
            )?;
        }

        MachineTerminator::JumpTable { index, entries } => {
            if entries.is_empty() {
                return Err(WasmError::internal(
                    "armv7a jump table requires at least one entry".into(),
                ));
            }
            if entries.len() == 1 {
                let label = fc.emit_edge(entries[0].target, &entries[0].args)?;
                fc.emit_branch(BranchFixupKind::B, label);
                return Ok(());
            }

            // Clamp index to entries.len()-1
            let index_hw = match index {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            let max_idx = (entries.len() - 1) as u32;
            let clamp_hw = if index_hw == SCRATCH0 {
                SCRATCH1
            } else {
                SCRATCH0
            };
            fc.emit_load_u32(clamp_hw, max_idx);
            fc.text.emit_u32(enc::cmp_reg(index_hw, clamp_hw));
            // If index > max, use max (conditional move)
            fc.text
                .emit_u32(enc::mov_reg_cond(Cond::Hi, index_hw, clamp_hw));

            // Emit edge stubs and collect their labels
            let mut edge_label_ids = Vec::with_capacity(entries.len());
            for entry in entries {
                let label = fc.emit_edge(entry.target, &entry.args)?;
                edge_label_ids.push(label);
            }

            // ARM reads PC as current+8 for data-processing instructions, so
            // keep one 4-byte padding slot between the dispatch ADD and the
            // first branch-table entry.
            fc.text.emit_u32(enc::add_reg_lsl_imm(
                Arm32Reg::R15,
                Arm32Reg::R15,
                index_hw,
                2,
            ));
            fc.text.emit_u32(enc::nop());

            // Emit branch table entries (will be patched by resolve_fixups)
            for &label_id in &edge_label_ids {
                fc.emit_branch(BranchFixupKind::B, label_id);
            }
        }
    }
    Ok(())
}

fn compile_branch_condition(
    fc: &mut FunctionCompiler<'_>,
    cond: &MachineBranchCond,
) -> Result<Cond, WasmError> {
    match cond {
        MachineBranchCond::Value(value) => {
            // Branch taken if value != 0
            let hw = match value {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };
            fc.text.emit_u32(enc::cmp_imm(hw, 0, 0));
            Ok(Cond::Ne)
        }

        MachineBranchCond::IntCompare {
            width,
            kind,
            sign,
            lhs,
            rhs,
        } => {
            let lhs_hw = match lhs {
                MachineValue::Reg(r) => map_reg(*r)?,
                MachineValue::Imm64(v) => {
                    fc.emit_load_u32(SCRATCH0, *v as u32);
                    SCRATCH0
                }
            };

            match rhs {
                MachineValue::Reg(r) => {
                    fc.text.emit_u32(enc::cmp_reg(lhs_hw, map_reg(*r)?));
                }
                MachineValue::Imm64(v) => {
                    if let Some((imm8, rot)) = enc::encode_arm_imm(*v as u32) {
                        fc.text.emit_u32(enc::cmp_imm(lhs_hw, imm8, rot));
                    } else {
                        let tmp = if lhs_hw == SCRATCH0 {
                            Arm32Reg::R3
                        } else {
                            SCRATCH0
                        };
                        fc.emit_load_u32(tmp, *v as u32);
                        fc.text.emit_u32(enc::cmp_reg(lhs_hw, tmp));
                    }
                }
            }

            Ok(match (kind, sign) {
                (MachineCompareKind::Eq, _) => Cond::Eq,
                (MachineCompareKind::Ne, _) => Cond::Ne,
                (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
                (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Cc,
                (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
                (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
                (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
                (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
                (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
                (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Cs,
            })
        }

        MachineBranchCond::FloatCompare {
            width,
            kind,
            lhs,
            rhs,
        } => {
            let lhs_reg = fp_reg_from_value(lhs)?;
            let rhs_reg = fp_reg_from_value(rhs)?;
            let gp_base = max_gp_mapped_regs();
            let lhs_d = fp_machine_reg((lhs_reg.0 as usize).checked_sub(gp_base).unwrap()).unwrap();
            let rhs_d = fp_machine_reg((rhs_reg.0 as usize).checked_sub(gp_base).unwrap()).unwrap();

            match width {
                MachineFloatWidth::F64 => {
                    fc.text.emit_u32(enc::vcmp_d(lhs_d, rhs_d));
                }
                MachineFloatWidth::F32 => {
                    fc.text.emit_u32(enc::vcmp_s(lhs_d * 2, rhs_d * 2));
                }
            }
            fc.text.emit_u32(enc::vmrs_apsr());

            Ok(match kind {
                MachineCompareKind::Eq => Cond::Eq,
                MachineCompareKind::Ne => Cond::Ne,
                MachineCompareKind::Lt => Cond::Mi,
                MachineCompareKind::Gt => Cond::Gt,
                MachineCompareKind::Le => Cond::Ls,
                MachineCompareKind::Ge => Cond::Ge,
            })
        }
    }
}

// ─── Utility ────────────────────────────────────────────────────────────────

fn trap_kind_to_u32(kind: MachineTrapKind) -> u32 {
    match kind {
        MachineTrapKind::Unreachable => 0,
        MachineTrapKind::MemoryOutOfBounds => 1,
        MachineTrapKind::TableOutOfBounds => 2,
        MachineTrapKind::InvalidFunctionReference => 3,
        MachineTrapKind::IndirectCallTypeMismatch => 4,
        MachineTrapKind::IntegerDivideByZero => 5,
        MachineTrapKind::IntegerOverflow => 6,
        MachineTrapKind::StackOverflow => 7,
        MachineTrapKind::HelperFailure => 8,
    }
}
