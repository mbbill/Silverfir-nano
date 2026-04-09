//! Backend state and `ArchBackend` trait glue for ARMv7-A.
//!
//! This is the bridge between the common pipeline and the ARM32-specific
//! instruction lowering. All lowering logic lives in `inst.rs` and `control.rs`.

use alloc::vec::Vec;

#[cfg(sf_has_debug_regions)]
use crate::vm::arch::common::types::DebugRegion;
use crate::{
    error::WasmError,
    vm::{
        arch::common::{
            backend::ArchBackend,
            core::CompilerCore,
            helpers::trap_code,
            scratch_pool::ScratchPool,
            types::ParallelSource,
        },
        machine::machine_ir::{
            MachineBlock, MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId,
            MachineFunction, MachineFunctionAbi, MachineInst, MachineReg, MachineTerminator,
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
        self, emit_shared_epilogue, emit_shared_prologue, fp_machine_reg, map_fixed_reg, map_reg,
        SCRATCH0,
    },
    armv7a_raise_trap,
    enc::{self, Cond},
    inst::{emit_load_u32_into, emit_patchable_addr_into},
    reg::Arm32Reg,
};

// ── Branch fixup types ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BranchFixupKind {
    B,
    /// `bl <label>` — branch with link, populates LR. Used by the root caller
    /// stub to call into the function's internal entry.
    Bl,
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
    #[cfg(sf_has_debug_regions)]
    pub debug_regions: Vec<DebugRegion>,
}

// ── Arm32Backend ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct Arm32Backend<'a> {
    pub core: CompilerCore<'a>,
    fixups: Vec<BranchFixup>,
    pub(super) gp_scratch: ScratchPool<Arm32Reg, 2>,
    pub(super) fp_scratch: ScratchPool<u32, 3>,
}

fn emit_zero_dynamic_callee_prefix(
    core: &mut CompilerCore<'_>,
    fixups: &mut Vec<BranchFixup>,
    gp_scratch: &ScratchPool<Arm32Reg, 2>,
    callee_fp: Arm32Reg,
    arg_slots: u16,
    frame_prefix_slots_reg: Arm32Reg,
    zero_reg: Arm32Reg,
) {
    let start_guard = gp_scratch.scoped_alloc();
    let start = *start_guard;
    emit_load_u32_into(&mut core.text, start, u32::from(arg_slots) * 8);
    core.text.emit_u32(enc::add_reg(start, callee_fp, start));
    core.text.emit_u32(enc::lsl_imm(
        frame_prefix_slots_reg,
        frame_prefix_slots_reg,
        3,
    ));
    core.text.emit_u32(enc::add_reg(
        frame_prefix_slots_reg,
        callee_fp,
        frame_prefix_slots_reg,
    ));
    core.text
        .emit_u32(enc::cmp_reg(start, frame_prefix_slots_reg));

    let done = core.new_label();
    let loop_label = core.new_label();
    let done_fixup = core.text.len();
    core.text.emit_u32(enc::nop());
    fixups.push(BranchFixup {
        offset: done_fixup,
        kind: BranchFixupKind::BCond(Cond::Cs),
        target: done,
    });

    emit_load_u32_into(&mut core.text, zero_reg, 0);
    core.bind_label(loop_label);
    core.text.emit_u32(enc::str_imm(zero_reg, start, 0));
    core.text.emit_u32(enc::str_imm(zero_reg, start, 4));
    core.text.emit_u32(enc::add_imm(start, start, 8, 0));
    core.text
        .emit_u32(enc::cmp_reg(start, frame_prefix_slots_reg));
    let loop_fixup = core.text.len();
    core.text.emit_u32(enc::nop());
    fixups.push(BranchFixup {
        offset: loop_fixup,
        kind: BranchFixupKind::BCond(Cond::Cc),
        target: loop_label,
    });
    core.bind_label(done);
}

// ── ArchBackend trait implementation ─────────────────────────────────────────

impl<'a> ArchBackend<'a> for Arm32Backend<'a> {
    const NAME: &'static str = "armv7a";

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
        emit_shared_prologue(&mut self.core.text);
        // Move args: entry is `fn(ctx: *mut NativeContext, fp: *mut u64) -> u32`
        // R0 = ctx, R1 = fp → fixed CTX / FP machine regs
        self.core
            .text
            .emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_CTX_REG), Arm32Reg::R0));
        self.core
            .text
            .emit_u32(enc::mov_reg(map_fixed_reg(MACHINE_FP_REG), Arm32Reg::R1));
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

    /// Public-entry caller stub. Pushes a root call record onto the host
    /// stack:
    ///   [sp + 0] = caller_result_base = MACHINE_FP_REG (the root frame)
    ///   [sp + 4] = caller_fp           = MACHINE_FP_REG (the root frame)
    /// Then `bl`s the internal entry. After the body's unified Return rets,
    /// SP is back to where it was, and R0 holds 0 (success) or a trap kind.
    /// Falls through to `lower_epilogue`.
    fn lower_root_caller_stub(&mut self) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // sub sp, sp, #8
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        // str fp_reg, [sp, #0]   ; caller_result_base = root frame
        self.core
            .text
            .emit_u32(enc::str_imm(fp_reg, Arm32Reg::SP, 0));
        // str fp_reg, [sp, #4]   ; caller_fp = root frame
        self.core
            .text
            .emit_u32(enc::str_imm(fp_reg, Arm32Reg::SP, 4));
        // bl internal_entry_label  (resolved by patch_fixups)
        let internal_entry_label = self.core.internal_entry_label;
        self.emit_branch(BranchFixupKind::Bl, internal_entry_label);
    }

    fn lower_epilogue(&mut self) {
        emit_shared_epilogue(&mut self.core.text);
    }

    /// Body entry prelude. Reached via `bl` from either the public stub or
    /// another SF function. Pushes the live LR onto the host stack so the
    /// body can freely make nested `bl`s without losing the BL return
    /// address that brought us here. The body's unified Return and
    /// `body_local_error_label` both pop this save before the native return.
    fn lower_body_prelude(&mut self) {
        // sub sp, sp, #8           ; preserve 8-byte alignment
        // str lr, [sp, #0]         ; save the BL return address
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R14, Arm32Reg::SP, 0));
    }

    /// Body-local error tail (`body_local_error_label`). Reached from trap
    /// stubs and post-`bl` status checks. Pops the body prelude link save
    /// and the caller's call record, restores `MACHINE_FP_REG`, and `bx`s
    /// to LR without touching R0 (which already holds the trap kind set by
    /// the trap stub or inherited from a trapped descendant's `bl`).
    fn lower_body_local_error_tail(&mut self) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // ldr lr, [sp, #0]
        self.core
            .text
            .emit_u32(enc::ldr_imm(Arm32Reg::R14, Arm32Reg::SP, 0));
        // add sp, sp, #8           ; pop body prelude link save
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        // ldr fp_reg, [sp, #4]     ; recover caller_fp (skip caller_result_base)
        self.core
            .text
            .emit_u32(enc::ldr_imm(fp_reg, Arm32Reg::SP, 4));
        // add sp, sp, #8           ; pop call record
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        // bx lr  (R0 untouched — preserves the error code)
        self.core.text.emit_u32(enc::bx(Arm32Reg::R14));
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
            // Catch scratch leaks between instructions.
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

    fn lower_trap(&mut self, kind: MachineTrapKind) {
        // Shared trap handler: called once per trap kind, shared by all
        // TrapIf sites. The AAPCS shim takes (ctx, kind: u32) so the args
        // fit cleanly in R0/R1 without u64 alignment skipping.
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R0, map_fixed_reg(MACHINE_CTX_REG)));
        self.emit_load_u32(Arm32Reg::R1, trap_code(kind) as u32);
        self.emit_host_call(armv7a_raise_trap as usize);
        // The shim returns NativeCallStatus::Error (= 1) in R0. Branch to
        // body_local_error_label, which preserves R0 and propagates upward
        // through the unified Return tail.
        let body_local_error = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::B, body_local_error);
    }

    fn lower_unconditional_branch(&mut self, label: usize) {
        self.emit_branch(BranchFixupKind::B, label);
    }

    fn patch_fixups(&mut self) -> Result<(), WasmError> {
        for fixup in &self.fixups {
            let target_offset = self
                .core
                .labels
                .get(fixup.target)
                .and_then(|v| *v)
                .ok_or_else(|| WasmError::internal("armv7a branch label unresolved".into()))?;
            let delta = target_offset as i32 - fixup.offset as i32;
            let inst = match fixup.kind {
                BranchFixupKind::B => enc::b(delta),
                BranchFixupKind::Bl => enc::bl(delta),
                BranchFixupKind::BCond(cond) => enc::b_cond(cond, delta),
            };
            self.core.text.patch_u32(fixup.offset, inst);
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
        CompiledArm32Entry {
            entry,
            text_len: emitted.text_len,
            #[cfg(sf_has_debug_regions)]
            debug_regions: emitted.debug_regions.clone(),
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
        let fp_idx =
            crate::vm::machine::machine_ir::fp_reg_index(reg, self.core.compiled.backend())
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
        // ARM hard-float helpers may clobber D0-D7. D0-D2 are reserved as
        // scratch/return registers in this backend; preserve the caller-
        // clobbered FP dynamic subset D3-D7 around every host call without
        // perturbing the native stack layout used for helper stack arguments.
        let helper_scratch = self
            .core
            .compiled
            .abi()
            .functions
            .get(self.core.function.id.0 as usize)
            .and_then(|runtime| runtime.helper_scratch);
        if let Some(helper_scratch) = helper_scratch {
            debug_assert!(
                helper_scratch.slots >= 5,
                "armv7a helper scratch must reserve five 64-bit slots for D3-D7"
            );
            self.emit_helper_scratch_save(helper_scratch);
        }
        emit_load_u32_into(&mut self.core.text, SCRATCH0, target as u32);
        self.core.text.emit_u32(enc::blx_reg(SCRATCH0));
        if let Some(helper_scratch) = helper_scratch {
            self.emit_helper_scratch_restore(helper_scratch);
        }
    }

    /// Save D3-D7 into the function's `helper_scratch` slots. Uses LDR/STR
    /// immediate addressing when the slot offsets fit in the 12-bit
    /// unsigned immediate field; otherwise falls back to a register-base
    /// path that materializes `fp_reg + base` into a temporary GP register
    /// (spilled to the host stack so the JIT register file is preserved).
    fn emit_helper_scratch_save(
        &mut self,
        helper_scratch: crate::vm::machine::machine_ir::MachineFrameRegion,
    ) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let gp_lo = self.gp_scratch.reg(0);
        let gp_hi = self.gp_scratch.reg(1);
        let base_byte_offset = i32::from(helper_scratch.base_slot) * 8;
        let last_byte_offset = base_byte_offset + 4 * 8 + 4; // D7 high half
        let direct = base_byte_offset >= 0 && last_byte_offset <= 4095;

        if direct {
            // Fast path: every offset fits the 12-bit immediate.
            let mut d_reg = 3u32;
            let mut slot = 0i32;
            while d_reg <= 7 {
                let byte_offset = base_byte_offset + slot * 8;
                self.core.text.emit_u32(enc::vmov_rr_d(gp_lo, gp_hi, d_reg));
                self.core
                    .text
                    .emit_u32(enc::str_imm(gp_lo, fp_reg, byte_offset));
                self.core
                    .text
                    .emit_u32(enc::str_imm(gp_hi, fp_reg, byte_offset + 4));
                d_reg += 1;
                slot += 1;
            }
        } else {
            // Slow path: spill R5 (a callee-saved dynamic GP reg) to the
            // host stack, materialize `fp_reg + base` into R5, then use R5
            // as the LDR/STR base for the loop. R5 is restored from the
            // stack before we leave this function so the JIT register
            // file is unchanged.
            self.spill_helper_base_reg();
            self.materialize_helper_base(Arm32Reg::R5, fp_reg, base_byte_offset);
            let mut d_reg = 3u32;
            let mut slot = 0i32;
            while d_reg <= 7 {
                let byte_offset = slot * 8;
                self.core.text.emit_u32(enc::vmov_rr_d(gp_lo, gp_hi, d_reg));
                self.core
                    .text
                    .emit_u32(enc::str_imm(gp_lo, Arm32Reg::R5, byte_offset));
                self.core
                    .text
                    .emit_u32(enc::str_imm(gp_hi, Arm32Reg::R5, byte_offset + 4));
                d_reg += 1;
                slot += 1;
            }
            self.restore_helper_base_reg();
        }
    }

    /// Restore D3-D7 from the function's `helper_scratch` slots. Mirrors
    /// `emit_helper_scratch_save`.
    fn emit_helper_scratch_restore(
        &mut self,
        helper_scratch: crate::vm::machine::machine_ir::MachineFrameRegion,
    ) {
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let gp_lo = self.gp_scratch.reg(0);
        let gp_hi = self.gp_scratch.reg(1);
        let base_byte_offset = i32::from(helper_scratch.base_slot) * 8;
        let last_byte_offset = base_byte_offset + 4 * 8 + 4;
        let direct = base_byte_offset >= 0 && last_byte_offset <= 4095;

        if direct {
            let mut d_reg = 3u32;
            let mut slot = 0i32;
            while d_reg <= 7 {
                let byte_offset = base_byte_offset + slot * 8;
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(gp_lo, fp_reg, byte_offset));
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(gp_hi, fp_reg, byte_offset + 4));
                self.core.text.emit_u32(enc::vmov_d_rr(d_reg, gp_lo, gp_hi));
                d_reg += 1;
                slot += 1;
            }
        } else {
            self.spill_helper_base_reg();
            self.materialize_helper_base(Arm32Reg::R5, fp_reg, base_byte_offset);
            let mut d_reg = 3u32;
            let mut slot = 0i32;
            while d_reg <= 7 {
                let byte_offset = slot * 8;
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(gp_lo, Arm32Reg::R5, byte_offset));
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(gp_hi, Arm32Reg::R5, byte_offset + 4));
                self.core.text.emit_u32(enc::vmov_d_rr(d_reg, gp_lo, gp_hi));
                d_reg += 1;
                slot += 1;
            }
            self.restore_helper_base_reg();
        }
    }

    /// `push {r5}` — save the dynamic GP register we're about to use as a
    /// helper-scratch base pointer. The JIT may have a live SSA value in
    /// R5; we restore it before returning to JIT-managed code.
    fn spill_helper_base_reg(&mut self) {
        // sub sp, sp, #8  (8-byte alignment for AAPCS)
        // str r5, [sp, #0]
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R5, Arm32Reg::SP, 0));
    }

    /// `pop {r5}` — counterpart to `spill_helper_base_reg`.
    fn restore_helper_base_reg(&mut self) {
        // ldr r5, [sp, #0]
        // add sp, sp, #8
        self.core
            .text
            .emit_u32(enc::ldr_imm(Arm32Reg::R5, Arm32Reg::SP, 0));
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
    }

    /// Materialize `fp_reg + base_byte_offset` into `dst` using a
    /// MOVW/MOVT pair plus an ADD. `base_byte_offset` may be any 32-bit
    /// signed value; we widen it via the standard 16+16 pair.
    fn materialize_helper_base(&mut self, dst: Arm32Reg, base_reg: Arm32Reg, byte_offset: i32) {
        let off = byte_offset as u32;
        self.core.text.emit_u32(enc::movw(dst, off as u16));
        let hi = (off >> 16) as u16;
        if hi != 0 {
            self.core.text.emit_u32(enc::movt(dst, hi));
        }
        self.core.text.emit_u32(enc::add_reg(dst, base_reg, dst));
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
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a emit_move_gp_value cannot consume reserved cache register {} as a real value",
                    reg.0
                )));
            }
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
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a materialize_gp_into cannot consume reserved cache register {} as a real value",
                    reg.0
                )));
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
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a emit_pair_args_to_r0_r1 cannot consume reserved cache register {} as lo half",
                    reg.0
                )));
            }
        };
        let src_hi_reg = match src_hi {
            MachineValue::Reg(r) => Some(map_reg(*r)?),
            MachineValue::Imm64(_) => None,
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a emit_pair_args_to_r0_r1 cannot consume reserved cache register {} as hi half",
                    reg.0
                )));
            }
        };
        if matches!(src_lo_reg, Some(Arm32Reg::R1)) && matches!(src_hi_reg, Some(Arm32Reg::R0)) {
            let s = self.gp_scratch.scoped_alloc();
            self.core.text.emit_u32(enc::mov_reg(*s, Arm32Reg::R0));
            self.core
                .text
                .emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R1));
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R1, *s));
            return Ok(());
        }
        let moved_hi =
            matches!(src_hi_reg, Some(Arm32Reg::R0)) && !matches!(src_lo_reg, Some(Arm32Reg::R0));
        if moved_hi {
            self.core
                .text
                .emit_u32(enc::mov_reg(Arm32Reg::R1, Arm32Reg::R0));
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
            self.core
                .text
                .emit_u32(enc::mov_reg(Arm32Reg::R0, Arm32Reg::R1));
            self.core.text.emit_u32(enc::mov_reg(Arm32Reg::R1, *s));
            return Ok(());
        }
        let moved_lo = dst_hi_hw == Arm32Reg::R0 && dst_lo_hw != Arm32Reg::R0;
        if moved_lo {
            self.core
                .text
                .emit_u32(enc::mov_reg(dst_lo_hw, Arm32Reg::R0));
        }
        let moved_hi = dst_lo_hw == Arm32Reg::R1 && dst_hi_hw != Arm32Reg::R1;
        if moved_hi {
            self.core
                .text
                .emit_u32(enc::mov_reg(dst_hi_hw, Arm32Reg::R1));
        }
        if !moved_lo && dst_lo_hw != Arm32Reg::R0 {
            self.core
                .text
                .emit_u32(enc::mov_reg(dst_lo_hw, Arm32Reg::R0));
        }
        if !moved_hi && dst_hi_hw != Arm32Reg::R1 {
            self.core
                .text
                .emit_u32(enc::mov_reg(dst_hi_hw, Arm32Reg::R1));
        }
        Ok(())
    }

    // ── Caller-saved register spill / restore ────────────────────────────

    pub(super) fn spill_caller_saved_gp_regs(&mut self) {
        // Keep SP 8-byte aligned across helper calls while preserving the
        // caller-clobbered GP dynamic subset. R9 is platform-defined under
        // AAPCS, so the JIT cannot rely on host helpers preserving it.
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 24, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R2, Arm32Reg::SP, 8));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R3, Arm32Reg::SP, 12));
        self.core
            .text
            .emit_u32(enc::str_imm(Arm32Reg::R9, Arm32Reg::SP, 16));
    }

    pub(super) fn restore_caller_saved_gp_regs(&mut self, preserved: &[Arm32Reg]) {
        if !preserved.contains(&Arm32Reg::R0) {
            self.core
                .text
                .emit_u32(enc::ldr_imm(Arm32Reg::R0, Arm32Reg::SP, 0));
        }
        if !preserved.contains(&Arm32Reg::R1) {
            self.core
                .text
                .emit_u32(enc::ldr_imm(Arm32Reg::R1, Arm32Reg::SP, 4));
        }
        if !preserved.contains(&Arm32Reg::R2) {
            self.core
                .text
                .emit_u32(enc::ldr_imm(Arm32Reg::R2, Arm32Reg::SP, 8));
        }
        if !preserved.contains(&Arm32Reg::R3) {
            self.core
                .text
                .emit_u32(enc::ldr_imm(Arm32Reg::R3, Arm32Reg::SP, 12));
        }
        if !preserved.contains(&Arm32Reg::R9) {
            self.core
                .text
                .emit_u32(enc::ldr_imm(Arm32Reg::R9, Arm32Reg::SP, 16));
        }
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 24, 0));
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
                MachineValue::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a emit_values_to_regs_via_stack cannot consume reserved cache register {} as a real value",
                        reg.0
                    )));
                }
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
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, bytes, 0));
    }

    pub(super) fn emit_stack_temp_free(&mut self, bytes: u32) {
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, bytes, 0));
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
        // CRITICAL: this is the PARALLEL-MOVE handler. Multiple moves may
        // still be pending in the scheduler when this function runs, so we
        // must NEVER write to a JIT-mapped GP register (R0..R3, R5..R9) as
        // a temporary — those registers may hold the source for a later
        // pending move. All temporaries here must come from the scratch
        // pool (R12 / R14).
        if self.is_fp_machine_reg(dst.reg) {
            let dd = self.map_fp_dreg(dst.reg)?;
            match src {
                ParallelSource::Reg { reg, .. } if self.is_fp_machine_reg(reg) => {
                    let sd = self.map_fp_dreg(reg)?;
                    if dd != sd {
                        self.core.text.emit_u32(enc::vmov_d(dd, sd));
                    }
                }
                ParallelSource::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a received non-identity reserved cache edge move into {} from {}",
                        dst.reg.0,
                        reg.0
                    )));
                }
                ParallelSource::Reg { reg, .. } => {
                    // GP → FP. Use a JIT GP scratch (R12/R14) for the
                    // zero-extended high half so we don't clobber a
                    // pending parallel-move source.
                    let src_gp = map_reg(reg)?;
                    let zero_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *zero_s, 0);
                    self.core
                        .text
                        .emit_u32(enc::vmov_d_rr(dd, src_gp, *zero_s));
                }
                ParallelSource::Imm(value) => {
                    let lo = value as u32;
                    let hi = (value >> 32) as u32;
                    let lo_s = self.gp_scratch.scoped_alloc();
                    let hi_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *lo_s, lo);
                    emit_load_u32_into(&mut self.core.text, *hi_s, hi);
                    self.core
                        .text
                        .emit_u32(enc::vmov_d_rr(dd, *lo_s, *hi_s));
                }
                ParallelSource::GpTemp(id) => {
                    let temp = self.gp_scratch.reg(id);
                    let zero_s = self.gp_scratch.scoped_alloc();
                    emit_load_u32_into(&mut self.core.text, *zero_s, 0);
                    self.core
                        .text
                        .emit_u32(enc::vmov_d_rr(dd, temp, *zero_s));
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
                    // FP → GP: extract low 32 bits via vmov_rr_d. Use a
                    // JIT GP scratch as the high-half slot.
                    let sd = self.map_fp_dreg(reg)?;
                    let hi_scratch = self.gp_scratch.scoped_alloc();
                    self.core
                        .text
                        .emit_u32(enc::vmov_rr_d(dst_gp, *hi_scratch, sd));
                }
                ParallelSource::ReservedReg(reg) => {
                    return Err(WasmError::internal(alloc::format!(
                        "armv7a received non-identity reserved cache edge move into {} from {}",
                        dst.reg.0,
                        reg.0
                    )));
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
                    let hi_scratch = self.gp_scratch.scoped_alloc();
                    self.core
                        .text
                        .emit_u32(enc::vmov_rr_d(dst_gp, *hi_scratch, temp));
                }
            }
        }
        Ok(())
    }

    // ── Call infrastructure ──────────────────────────────────────────────

    pub(super) fn runtime_for(
        &self,
        func_id: MachineFuncId,
    ) -> Result<&MachineFunctionAbi, WasmError> {
        self.core.runtime_for(func_id)
    }

    /// Direct call lowering. Pushes the backend-private call record onto
    /// the host stack, switches `MACHINE_FP_REG` to the callee frame, and
    /// `bl`s through a patchable MOVW/MOVT-loaded callee address. After the
    /// callee returns, status is checked in R0; nonzero branches to the
    /// body-local error tail to propagate the trap upward.
    pub(super) fn emit_call_direct(
        &mut self,
        callee: MachineFuncId,
        callee_frame_base: MachineReg,
        caller_result_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_fp = map_reg(callee_frame_base)?;
        let caller_result_base = map_reg(caller_result_base)?;
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let body_local_error_label = self.core.body_local_error_label;

        // Push the backend-private call record onto the host stack:
        //   [sp + 0] = caller_result_base
        //   [sp + 4] = caller_fp
        // (matches the order popped by `lower_body_local_error_tail` and the
        // unified return sequence in control.rs.)
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(caller_result_base, Arm32Reg::SP, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(fp_reg, Arm32Reg::SP, 4));

        // Switch MACHINE_FP_REG to the callee frame.
        self.core.text.emit_u32(enc::mov_reg(fp_reg, callee_fp));

        // Load the callee's internal entry address into a scratch via a
        // patchable MOVW/MOVT pair, then `blx` through it.
        let s = self.gp_scratch.scoped_alloc();
        let callee_patch = emit_patchable_addr_into(&mut self.core.text, *s);
        self.core.text.emit_u32(enc::blx_reg(*s));
        drop(s);

        // Record the direct-call patch so module-link patching writes the
        // resolved internal-entry address into the MOVW/MOVT pair.
        self.core
            .direct_call_patches
            .push(crate::vm::arch::common::types::DirectCallPatch {
                literal_offset: callee_patch,
                callee,
            });

        // --- callee returns here. R0 holds 0 (success) or trap kind (error).
        // Status check propagates errors to body_local_error_label.
        self.emit_status_check_to(body_local_error_label);

        // Continuation: branch to the continuation block. The block layout
        // pass already prefers placing the continuation immediately after
        // the call site, so this branch is often elidable when the
        // continuation is the next emitted block.
        let continuation_label = self.core.block_label(continuation)?;
        self.emit_branch(BranchFixupKind::B, continuation_label);
        Ok(())
    }

    /// Unified return sequence. Inline at every `MachineTerminator::Return`.
    /// Pops the body prelude link save and the caller's call record from
    /// the host stack, copies the function's `return_results` region into
    /// `*caller_result_base`, restores `MACHINE_FP_REG`, sets R0 = 0, and
    /// `bx`s to LR.
    ///
    /// On entry the host stack looks like:
    ///   [sp +  0] = saved LR (body prelude link save)
    ///   [sp +  8] = caller_result_base (call record slot 0)
    ///   [sp + 12] = caller_fp           (call record slot 1)
    ///
    /// The copy loop uses R12 (SCRATCH0) as the result-base pointer and
    /// R14 (SCRATCH1, alias LR) as the per-iteration data temp. Because
    /// SCRATCH1 *is* LR, the prelude's saved LR has to stay on the stack
    /// until *after* the copy loop finishes — only then do we materialize
    /// LR for the final `bx lr`. We bypass the scratch pool here and pin
    /// the two physical registers directly so the ordering is explicit.
    pub(super) fn emit_return_sequence(&mut self) -> Result<(), WasmError> {
        let runtime = self.runtime_for(self.core.function.id)?.clone();
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        // R12 and R14 are the two GP scratches in this backend; both are
        // free at function-tail boundaries because every preceding insn
        // dropped its scratch guards.
        let result_base = Arm32Reg::R12;
        let temp = Arm32Reg::R14;

        // 1. Load caller_result_base from the call record (sp + 8). The
        //    body prelude link save still occupies sp[0..8]; we leave it
        //    there until step 3 so the copy loop can clobber R14 freely.
        self.core
            .text
            .emit_u32(enc::ldr_imm(result_base, Arm32Reg::SP, 8));

        // 2. Copy each return slot from the callee frame to *result_base.
        //    On 32-bit GP we copy 8 bytes per slot via two ldr/str pairs.
        if let Some(results) = runtime.return_results {
            for index in 0..results.slots as i32 {
                let src_off = (results.base_slot as i32 + index) * 8;
                let dst_off = index * 8;
                // lo half
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(temp, fp_reg, src_off));
                self.core
                    .text
                    .emit_u32(enc::str_imm(temp, result_base, dst_off));
                // hi half
                self.core
                    .text
                    .emit_u32(enc::ldr_imm(temp, fp_reg, src_off + 4));
                self.core
                    .text
                    .emit_u32(enc::str_imm(temp, result_base, dst_off + 4));
            }
        }

        // 3. Now the copy is done — restore LR from the body prelude link
        //    save, then advance sp past it. R14 / `temp` becomes LR again.
        self.core
            .text
            .emit_u32(enc::ldr_imm(Arm32Reg::R14, Arm32Reg::SP, 0));

        // 4. Load caller_fp from sp[12] into fp_reg, then drop both the
        //    body prelude link save (8 bytes) and the call record
        //    (another 8 bytes) in a single `add sp, #16`.
        self.core
            .text
            .emit_u32(enc::ldr_imm(fp_reg, Arm32Reg::SP, 12));
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 16, 0));

        // 5. Success status: R0 = 0.
        self.emit_load_u32(Arm32Reg::R0, 0);

        // 6. Native return (bx lr; LR holds the body prelude's saved LR).
        self.core.text.emit_u32(enc::bx(Arm32Reg::R14));
        Ok(())
    }

    /// Indirect call lowering. Same call-record / status-check protocol as
    /// `emit_call_direct`, but the callee entry address is supplied at
    /// runtime in a register (already resolved by call_indirect dispatch).
    pub(super) fn emit_call_indirect(
        &mut self,
        _callee_target: MachineReg,
        callee_entry: MachineReg,
        callee_frame_base: MachineReg,
        caller_result_base: MachineReg,
        continuation: MachineBlockId,
    ) -> Result<(), WasmError> {
        let callee_fp = map_reg(callee_frame_base)?;
        let caller_result_base = map_reg(caller_result_base)?;
        let callee_entry = map_reg(callee_entry)?;
        let fp_reg = map_fixed_reg(MACHINE_FP_REG);
        let body_local_error_label = self.core.body_local_error_label;

        // Push call record (same shape as emit_call_direct).
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, 8, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(caller_result_base, Arm32Reg::SP, 0));
        self.core
            .text
            .emit_u32(enc::str_imm(fp_reg, Arm32Reg::SP, 4));

        // Switch fp_reg to the callee frame.
        self.core.text.emit_u32(enc::mov_reg(fp_reg, callee_fp));

        // BLX to the runtime-resolved callee entry. Populates LR for the
        // callee's eventual `bx lr`.
        self.core.text.emit_u32(enc::blx_reg(callee_entry));

        // --- callee returns here. Status check + continuation branch.
        self.emit_status_check_to(body_local_error_label);
        let continuation_label = self.core.block_label(continuation)?;
        self.emit_branch(BranchFixupKind::B, continuation_label);
        Ok(())
    }

    /// Emit `cmp r0, #0; b.ne <label>` so the post-call status check
    /// branches to the body-local error tail when R0 != 0.
    fn emit_status_check_to(&mut self, label: usize) {
        self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R0, 0, 0));
        self.emit_branch(BranchFixupKind::BCond(Cond::Ne), label);
    }

    // ── Preserved-helper bridge ──────────────────────────────────────────
    //
    // The MachineIR `Memory*`/`Table*`/`*Drop` instruction kinds are lowered
    // by handing the operands to the shared `preserved_entry` runtime helper
    // (see `vm/runtime/preserved/`). The arm64 backend has a dedicated
    // preserved-helper frame that saves *all* caller-clobbered JIT registers
    // before the call. armv7a piggy-backs on the existing `emit_host_call`
    // path, which already saves D3-D7 via the per-function `helper_scratch`
    // slots; we additionally `spill_caller_saved_gp_regs` for R0-R3, R9.
    //
    // Layout while the helper runs:
    //
    //   sp +  0  ─┐  io[0..8]   (8 × 8-byte slots: IMM0, IMM1, ARG0, ARG1,
    //              │              ARG2, RET0, scratch, scratch)
    //   sp + 64  ─┘
    //   sp + 64  ─┐  R0..R3, R9 spill area (24 bytes)
    //   sp + 88  ─┘

    /// Encode a u32 immediate write into io[slot] (low + zero high half).
    fn emit_preserved_io_store_imm(&mut self, slot: usize, value: u32) {
        let s = self.gp_scratch.scoped_alloc();
        emit_load_u32_into(&mut self.core.text, *s, value);
        self.core
            .text
            .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8) as i32));
        emit_load_u32_into(&mut self.core.text, *s, 0);
        self.core
            .text
            .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8 + 4) as i32));
    }

    /// Encode a `MachineValue` write into io[slot]. Wasm-side values fit in
    /// 32 bits on this target, so the high half of the slot is zero.
    fn emit_preserved_io_store_value(
        &mut self,
        slot: usize,
        value: MachineValue,
    ) -> Result<(), WasmError> {
        match value {
            MachineValue::Reg(r) => {
                let src = map_reg(r)?;
                self.core
                    .text
                    .emit_u32(enc::str_imm(src, Arm32Reg::SP, (slot * 8) as i32));
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, 0);
                self.core
                    .text
                    .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8 + 4) as i32));
            }
            MachineValue::Imm64(v) => {
                let s = self.gp_scratch.scoped_alloc();
                emit_load_u32_into(&mut self.core.text, *s, v as u32);
                self.core
                    .text
                    .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8) as i32));
                emit_load_u32_into(&mut self.core.text, *s, (v >> 32) as u32);
                self.core
                    .text
                    .emit_u32(enc::str_imm(*s, Arm32Reg::SP, (slot * 8 + 4) as i32));
            }
            MachineValue::ReservedReg(reg) => {
                return Err(WasmError::internal(alloc::format!(
                    "armv7a preserved-helper cannot consume reserved cache register {} as a real value",
                    reg.0
                )));
            }
        }
        Ok(())
    }

    /// Lower a preserved-helper call. Allocates the I/O area, populates the
    /// IMM/ARG slots, dispatches `preserved_entry(ctx, op_code, io_ptr)`,
    /// and propagates errors via `body_local_error_label`. If `result_dst`
    /// is `Some`, the helper's RET0 slot is moved into that GP register on
    /// the success path.
    pub(super) fn emit_preserved_helper_call(
        &mut self,
        op_code: u32,
        imms: &[(usize, u32)],
        args: &[(usize, MachineValue)],
        result_dst: Option<MachineReg>,
    ) -> Result<(), WasmError> {
        use crate::vm::runtime::preserved::{io as preserved_io, preserved_entry};

        const IO_AREA_BYTES: u32 = 64;

        // Step 1: spill R0..R3, R9 onto the host stack so the JIT register
        // file survives the helper call. After this `sp -= 24` and the
        // saved area lives at sp[0..20].
        self.spill_caller_saved_gp_regs();

        // Step 2: allocate the I/O area below the spilled GP region. After
        // this, sp[0..64] = io area, sp[64..88] = saved GPs.
        self.core
            .text
            .emit_u32(enc::sub_imm(Arm32Reg::SP, Arm32Reg::SP, IO_AREA_BYTES, 0));

        // Step 3: populate IMM and ARG slots. R0..R3 / R9 are still live
        // here (the spill only copied them; it did not modify them), so
        // `map_reg` lookups for ARG MachineValues read the correct
        // physical registers.
        for &(slot, imm) in imms {
            self.emit_preserved_io_store_imm(slot, imm);
        }
        for &(slot, value) in args {
            self.emit_preserved_io_store_value(slot, value)?;
        }

        // Step 4: set up the C calling convention. r0 = ctx, r1 = op_code,
        // r2 = io_ptr (= sp). emit_host_call below will save/restore D3-D7
        // around the BLX via the per-function helper_scratch.
        self.core.text.emit_u32(enc::mov_reg(
            Arm32Reg::R0,
            map_fixed_reg(MACHINE_CTX_REG),
        ));
        self.emit_load_u32(Arm32Reg::R1, op_code);
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R2, Arm32Reg::SP));

        // Step 5: dispatch the helper.
        self.emit_host_call(preserved_entry as usize);

        // Step 6: stash status (R0) into SCRATCH0 (R12). After this we
        // commit to two diverging tails — success and error — that need to
        // restore the JIT register file in different ways. Note that R0
        // itself is *not* yet clobbered: the success-path restore will
        // overwrite it from the spill area, but the error-path restore
        // skips R0 so the helper's status code propagates back to the C
        // caller via the body-local error tail.
        self.core
            .text
            .emit_u32(enc::mov_reg(Arm32Reg::R12, Arm32Reg::R0));

        // Step 7: read the result (if any) into SCRATCH1 (R14 / LR). LR is
        // free at this point — the body prelude saved the body's real LR
        // onto the host stack at function entry, and the success path will
        // never need LR before the next instruction-level scratch alloc
        // overwrites it. (The error tail loads its own LR before `bx`.)
        if result_dst.is_some() {
            self.core.text.emit_u32(enc::ldr_imm(
                Arm32Reg::R14,
                Arm32Reg::SP,
                (preserved_io::RET0 * 8) as i32,
            ));
        }

        // Step 8: free the I/O area. Both tails need this.
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, IO_AREA_BYTES, 0));

        // Step 9: branch on status. The two tails differ in how they pop
        // the spill area: the success path runs `restore_caller_saved_gp_regs`
        // (which restores R0..R3, R9 from the spill and pops 24 bytes),
        // while the error path uses a bare `add sp, #24` so R0 retains the
        // helper's status code while it propagates to body_local_error_label.
        let error_path = self.core.new_label();
        self.core.text.emit_u32(enc::cmp_imm(Arm32Reg::R12, 0, 0));
        self.emit_branch(BranchFixupKind::BCond(Cond::Ne), error_path);

        // ── Success tail ─────────────────────────────────────────────────
        self.restore_caller_saved_gp_regs(&[]);
        if let Some(dst) = result_dst {
            let dst_hw = map_reg(dst)?;
            self.core
                .text
                .emit_u32(enc::mov_reg(dst_hw, Arm32Reg::R14));
        }
        let after = self.core.new_label();
        self.emit_branch(BranchFixupKind::B, after);

        // ── Error tail ───────────────────────────────────────────────────
        // Drop the spill area without touching R0..R3 / R9 so the helper's
        // status code (still in R0) flows into body_local_error_label.
        self.core.bind_label(error_path);
        self.core
            .text
            .emit_u32(enc::add_imm(Arm32Reg::SP, Arm32Reg::SP, 24, 0));
        let body_local_error_label = self.core.body_local_error_label;
        self.emit_branch(BranchFixupKind::B, body_local_error_label);

        // ── Continuation ─────────────────────────────────────────────────
        self.core.bind_label(after);
        Ok(())
    }
}
