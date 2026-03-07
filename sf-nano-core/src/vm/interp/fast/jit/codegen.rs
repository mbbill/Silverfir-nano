//! JIT codegen: per-opcode emission and group assembly.
//!
//! `JitEmitter` tracks TOS height and emits ARM64 instructions for each
//! supported Wasm opcode. Depth-variant register selection matches the
//! interpreter's TOS register window convention.

use alloc::vec::Vec;
use super::code_buf::CodeBuffer;
use super::reg::Reg;
use super::arm64_enc::{self, Cond};
use super::emit;

/// TOS registers in index order: T0=x26, T1=x27, T2=x28, T3=x0.
const TOS_REGS: [Reg; 4] = [Reg::T0, Reg::T1, Reg::T2, Reg::T3];

/// Hot local registers: L0=x23, L1=x24, L2=x25.
const LOCAL_REGS: [Reg; 3] = [Reg::L0, Reg::L1, Reg::L2];
const FP_ALIAS_REGS: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Static OOB trap message (null-terminated C string).
const OOB_MSG: &[u8; 28] = b"out of bounds memory access\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GprLoc {
    Canonical,
    Alias(Reg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FpWidth {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FpLoc {
    reg: u8,
    width: FpWidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotLoc {
    gpr: Option<GprLoc>,
    fp: Option<FpLoc>,
}

impl SlotLoc {
    const fn canonical() -> Self {
        Self {
            gpr: Some(GprLoc::Canonical),
            fp: None,
        }
    }

    const fn gpr_alias(reg: Reg) -> Self {
        Self {
            gpr: Some(GprLoc::Alias(reg)),
            fp: None,
        }
    }

    const fn with_fp(mut self, fp: Option<FpLoc>) -> Self {
        self.fp = fp;
        self
    }

    const fn fp_only(fp: FpLoc) -> Self {
        Self {
            gpr: None,
            fp: Some(fp),
        }
    }
}

/// Compute depth variant from stack height.
pub fn depth_variant(height: usize) -> u8 {
    if height == 0 { 1 } else { ((height - 1) % 4 + 1) as u8 }
}

/// Get TOS register at position `pos` (1=top) for depth variant `d` (1-4).
pub fn tos_reg(d: u8, pos: u8) -> Reg {
    TOS_REGS[(d as usize + 4 - pos as usize) % 4]
}

/// Emit a group of JIT instructions with automatic depth tracking.
pub struct JitEmitter<'a> {
    buf: &'a mut CodeBuffer,
    height: usize,
    /// Per-slot location metadata for the live TOS window.
    ///
    /// Index 0 is the current top of stack, index 1 is the next value down, etc.
    slots: [SlotLoc; 4],
    /// Current FP aliases for hot locals L0-L2, when they are known to still match.
    local_fp_aliases: [Option<FpLoc>; 3],
    /// Byte offset where this group's code starts.
    pub start_offset: usize,
    /// Byte offsets of B.HI instructions to patch to the OOB trap stub.
    trap_patches: Vec<usize>,
}

impl<'a> JitEmitter<'a> {
    pub fn new(buf: &'a mut CodeBuffer, initial_height: usize) -> Self {
        let start_offset = buf.len();
        Self {
            buf,
            height: initial_height,
            slots: [SlotLoc::canonical(); 4],
            local_fp_aliases: [None; 3],
            start_offset,
            trap_patches: Vec::new(),
        }
    }

    pub fn height(&self) -> usize { self.height }

    fn dv(&self) -> u8 { depth_variant(self.height) }

    fn shift_slots_for_push(&mut self) {
        self.slots[3] = self.slots[2];
        self.slots[2] = self.slots[1];
        self.slots[1] = self.slots[0];
        self.slots[0] = SlotLoc::canonical();
    }

    fn shift_slots_for_pop(&mut self) {
        self.slots[0] = self.slots[1];
        self.slots[1] = self.slots[2];
        self.slots[2] = self.slots[3];
        self.slots[3] = SlotLoc::canonical();
    }

    fn pop2_push_materialized(&mut self) {
        self.slots[0] = SlotLoc::canonical();
        self.slots[1] = self.slots[2];
        self.slots[2] = self.slots[3];
        self.slots[3] = SlotLoc::canonical();
        self.height -= 1;
    }

    fn pop2_push_fp(&mut self, fp: FpLoc) {
        self.slots[0] = SlotLoc::fp_only(fp);
        self.slots[1] = self.slots[2];
        self.slots[2] = self.slots[3];
        self.slots[3] = SlotLoc::canonical();
        self.height -= 1;
    }

    fn push_materialized_slot(&mut self) {
        self.height += 1;
        self.shift_slots_for_push();
    }

    fn push_gpr_alias_slot(&mut self, reg: Reg, fp: Option<FpLoc>) {
        self.height += 1;
        self.shift_slots_for_push();
        self.slots[0] = SlotLoc::gpr_alias(reg).with_fp(fp);
    }

    pub fn resolve_tos_reg(&self, pos: u8) -> Reg {
        match self.slots[(pos - 1) as usize].gpr {
            Some(GprLoc::Canonical) => tos_reg(self.dv(), pos),
            Some(GprLoc::Alias(reg)) => reg,
            None => tos_reg(self.dv(), pos),
        }
    }

    pub fn materialize_aliases(&mut self) {
        self.materialize_aliases_from(1);
    }

    fn materialize_aliases_from(&mut self, start_pos: usize) {
        let live = self.height.min(self.slots.len());
        for pos in start_pos..=live {
            self.materialize_slot_to_canonical(pos as u8);
            self.slots[pos - 1].fp = None;
        }
        self.local_fp_aliases = [None; 3];
    }

    fn materialize_slot_to_canonical(&mut self, pos: u8) {
        let idx = (pos - 1) as usize;
        let dst = tos_reg(self.dv(), pos);
        match self.slots[idx].gpr {
            Some(GprLoc::Canonical) => {}
            Some(GprLoc::Alias(src)) => {
                if src != dst {
                    self.buf.emit(arm64_enc::mov_reg_64(dst, src));
                }
                self.slots[idx].gpr = Some(GprLoc::Canonical);
            }
            None => {
                let fp = self.slots[idx].fp.expect("slot must have a source location");
                self.emit_fp_to_gpr(dst, fp);
                self.slots[idx].gpr = Some(GprLoc::Canonical);
            }
        }
    }

    fn slot_gpr_source(&mut self, pos: u8) -> Reg {
        let idx = (pos - 1) as usize;
        if self.slots[idx].gpr.is_none() {
            self.materialize_slot_to_canonical(pos);
        }
        self.resolve_tos_reg(pos)
    }

    fn copy_slot_to_gpr(&mut self, pos: u8, dst: Reg) {
        let idx = (pos - 1) as usize;
        match self.slots[idx].gpr {
            Some(GprLoc::Canonical) => {
                let src = tos_reg(self.dv(), pos);
                if src != dst {
                    self.buf.emit(arm64_enc::mov_reg_64(dst, src));
                }
            }
            Some(GprLoc::Alias(src)) => {
                if src != dst {
                    self.buf.emit(arm64_enc::mov_reg_64(dst, src));
                }
            }
            None => {
                let fp = self.slots[idx].fp.expect("slot must have a source location");
                self.emit_fp_to_gpr(dst, fp);
            }
        }
    }

    fn emit_fp_to_gpr(&mut self, dst: Reg, fp: FpLoc) {
        match fp.width {
            FpWidth::F32 => self.buf.emit(arm64_enc::fmov_fp32_to_gpr(dst, fp.reg as u32)),
            FpWidth::F64 => self.buf.emit(arm64_enc::fmov_fp64_to_gpr(dst, fp.reg as u32)),
        };
    }

    fn fp_reg_in_use(&self, reg: u8) -> bool {
        let live = self.height.min(self.slots.len());
        self.slots[..live]
            .iter()
            .filter_map(|slot| slot.fp)
            .any(|fp| fp.reg == reg)
            || self
                .local_fp_aliases
                .iter()
                .flatten()
                .any(|fp| fp.reg == reg)
    }

    fn alloc_fp_reg(&self) -> u8 {
        FP_ALIAS_REGS
            .into_iter()
            .find(|&reg| !self.fp_reg_in_use(reg))
            .expect("ran out of FP alias registers")
    }

    fn ensure_tos_fp(&mut self, pos: u8, width: FpWidth) -> u8 {
        let idx = (pos - 1) as usize;
        if let Some(fp) = self.slots[idx].fp {
            debug_assert_eq!(fp.width, width);
            return fp.reg;
        }

        let reg = self.alloc_fp_reg();
        let src = self.slot_gpr_source(pos);
        match width {
            FpWidth::F32 => self.buf.emit(arm64_enc::fmov_gpr_to_fp32(reg as u32, src)),
            FpWidth::F64 => self.buf.emit(arm64_enc::fmov_gpr_to_fp64(reg as u32, src)),
        };
        self.slots[idx].fp = Some(FpLoc { reg, width });
        reg
    }

    pub fn ensure_tos_f32(&mut self, pos: u8) -> u32 {
        self.ensure_tos_fp(pos, FpWidth::F32) as u32
    }

    pub fn ensure_tos_f64(&mut self, pos: u8) -> u32 {
        self.ensure_tos_fp(pos, FpWidth::F64) as u32
    }

    fn alloc_fp_result(&self) -> u8 {
        self.alloc_fp_reg()
    }

    /// Finish the group: emit dispatch stub (+ trap stub if needed). Returns start offset.
    pub fn finish(mut self) -> usize {
        self.materialize_aliases();
        emit::emit_dispatch_linear(self.buf);
        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish with an unconditional branch dispatch.
    ///
    /// Branches to the target in `pc->imm0` via nonlinear dispatch.
    pub fn finish_br(mut self) -> usize {
        self.materialize_aliases();
        emit::emit_dispatch_nonlinear(self.buf);
        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish with a br_if_simple conditional dispatch.
    ///
    /// Pops the condition from TOS top (i32). If nonzero, branches to target
    /// (loaded from pc->imm0 via nonlinear dispatch). If zero, falls through
    /// (linear dispatch to pc+1).
    ///
    /// The group's Instruction must have imm0 set to the branch target pointer.
    ///
    /// Emitted sequence (10 instructions = 40 bytes):
    /// ```text
    /// cbz  w_cond, +6           ; if cond == 0, skip to fallthrough
    /// <nonlinear dispatch: 5>   ; branch taken path
    /// <linear dispatch: 4>      ; fallthrough path
    /// ```
    pub fn finish_br_if_simple(mut self) -> usize {
        let cond = self.resolve_tos_reg(1);
        self.materialize_aliases_from(2);

        // CBZ w_cond, +6: if cond == 0, skip 5 taken-path instructions
        self.buf.emit(arm64_enc::cbz_32(cond, 6));

        // Branch taken: nonlinear dispatch (5 instructions)
        emit::emit_dispatch_nonlinear(self.buf);

        // Fallthrough: linear dispatch (4 instructions)
        emit::emit_dispatch_linear(self.buf);

        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish with an `if` conditional dispatch.
    ///
    /// Pops the condition from TOS top (i32). If zero, branches to the target
    /// in `pc->imm0` via nonlinear dispatch. If nonzero, falls through linearly
    /// to `pc + 1` (the then-path).
    pub fn finish_if(mut self) -> usize {
        let cond = self.resolve_tos_reg(1);
        self.materialize_aliases_from(2);

        // CBNZ w_cond, +6: cond != 0 skips the 5 taken-path instructions
        // and lands at the linear fallthrough dispatch.
        self.buf.emit(arm64_enc::cbnz_32(cond, 6));

        // Condition == 0: jump to else/end target.
        emit::emit_dispatch_nonlinear(self.buf);

        // Condition != 0: enter the then-path at pc+1.
        emit::emit_dispatch_linear(self.buf);

        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish with a conditional dispatch using an existing integer condition register.
    ///
    /// If `branch_on_zero` is true, zero branches to the nonlinear target in `pc->imm0`.
    /// Otherwise, nonzero branches to the nonlinear target.
    pub fn finish_br_if_simple_from_reg(mut self, cond: Reg, branch_on_zero: bool) -> usize {
        self.materialize_aliases_from(2);
        let branch = if branch_on_zero {
            arm64_enc::cbz_32(cond, 5)
        } else {
            arm64_enc::cbnz_32(cond, 5)
        };
        self.buf.emit(branch);
        emit::emit_dispatch_linear(self.buf);
        emit::emit_dispatch_nonlinear(self.buf);
        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish an `if` using an existing integer condition register.
    ///
    /// If `linear_on_zero` is true, zero enters the linear then-path. Otherwise
    /// nonzero enters the linear then-path.
    pub fn finish_if_from_reg(mut self, cond: Reg, linear_on_zero: bool) -> usize {
        self.materialize_aliases_from(2);
        let branch = if linear_on_zero {
            arm64_enc::cbz_32(cond, 6)
        } else {
            arm64_enc::cbnz_32(cond, 6)
        };
        self.buf.emit(branch);
        emit::emit_dispatch_nonlinear(self.buf);
        emit::emit_dispatch_linear(self.buf);
        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish `br_if_simple` when flags are already set.
    ///
    /// `taken_cond` is the condition under which control jumps to the nonlinear
    /// branch target in `pc->imm0`.
    pub fn finish_br_if_simple_from_flags(mut self, taken_cond: Cond) -> usize {
        self.materialize_aliases_from(3);
        self.buf.emit(arm64_enc::b_cond(taken_cond, 5));
        emit::emit_dispatch_linear(self.buf);
        emit::emit_dispatch_nonlinear(self.buf);
        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Finish `if` when flags are already set.
    ///
    /// `linear_cond` is the condition under which control enters the linear
    /// then-path at `pc + 1`.
    pub fn finish_if_from_flags(mut self, linear_cond: Cond) -> usize {
        self.materialize_aliases_from(3);
        self.buf.emit(arm64_enc::b_cond(linear_cond, 6));
        emit::emit_dispatch_nonlinear(self.buf);
        emit::emit_dispatch_linear(self.buf);
        self.emit_trap_stub_if_needed();
        self.start_offset
    }

    /// Emit the OOB trap stub and patch all B.HI branches to it.
    fn emit_trap_stub_if_needed(&mut self) {
        if self.trap_patches.is_empty() {
            return;
        }
        let trap_offset = self.buf.len();

        // Materialize OOB message pointer into TMP1
        let msg_addr = OOB_MSG.as_ptr() as u64;
        materialize_u64(self.buf, Reg::TMP1, msg_addr);

        // Store to ctx.trap_message
        self.buf.emit(arm64_enc::str_64(Reg::TMP1, Reg::CTX,
            emit::ctx_offset::TRAP_MESSAGE / 8));

        // Dispatch to term_inst
        self.buf.emit(arm64_enc::ldr_64(Reg::PC, Reg::CTX,
            emit::ctx_offset::TERM_INST / 8));
        self.buf.emit(arm64_enc::ldr_64(Reg::TMP0, Reg::PC, 0));
        self.buf.emit(arm64_enc::ldr_64(Reg::NH, Reg::PC, emit::INST_SIZE / 8));
        self.buf.emit(arm64_enc::br(Reg::TMP0));

        // Patch all B.HI branches to point to the trap stub
        for &patch_off in &self.trap_patches {
            let delta = (trap_offset - patch_off) as i32 / 4;
            self.buf.patch_u32(patch_off, arm64_enc::b_cond(Cond::HI, delta));
        }
    }

    /// Emit a raw u32 instruction (for test helpers).
    pub fn emit_raw(&mut self, inst: u32) {
        self.buf.emit(inst);
    }

    // ==================== Binary i32 ops (pop2_push1) ====================

    fn emit_binop_32(&mut self, f: fn(Reg, Reg, Reg) -> u32) {
        let d = self.dv();
        let lhs = tos_reg(d, 2);
        let rhs = tos_reg(d, 1);
        self.buf.emit(f(lhs, lhs, rhs));
        self.height -= 1;
    }

    pub fn i32_add(&mut self)   { self.emit_binop_32(arm64_enc::add_reg_32); }
    pub fn i32_sub(&mut self)   { self.emit_binop_32(arm64_enc::sub_reg_32); }
    pub fn i32_mul(&mut self)   { self.emit_binop_32(arm64_enc::mul_32); }
    pub fn i32_and(&mut self)   { self.emit_binop_32(arm64_enc::and_reg_32); }
    pub fn i32_or(&mut self)    { self.emit_binop_32(arm64_enc::orr_reg_32); }
    pub fn i32_xor(&mut self)   { self.emit_binop_32(arm64_enc::eor_reg_32); }
    pub fn i32_shl(&mut self)   { self.emit_binop_32(arm64_enc::lslv_32); }
    pub fn i32_shr_u(&mut self) { self.emit_binop_32(arm64_enc::lsrv_32); }
    pub fn i32_shr_s(&mut self) { self.emit_binop_32(arm64_enc::asrv_32); }
    pub fn i32_rotr(&mut self)  { self.emit_binop_32(arm64_enc::rorv_32); }

    pub fn i32_rotl(&mut self) {
        let d = self.dv();
        let lhs = tos_reg(d, 2);
        let rhs = tos_reg(d, 1);
        // rotl(a, b) = rotr(a, -b)
        self.buf.emit(arm64_enc::neg_32(Reg::TMP0, rhs));
        self.buf.emit(arm64_enc::rorv_32(lhs, lhs, Reg::TMP0));
        self.height -= 1;
    }

    // ==================== Binary i64 ops (pop2_push1) ====================

    fn emit_binop_64(&mut self, f: fn(Reg, Reg, Reg) -> u32) {
        let d = self.dv();
        let lhs = tos_reg(d, 2);
        let rhs = tos_reg(d, 1);
        self.buf.emit(f(lhs, lhs, rhs));
        self.height -= 1;
    }

    pub fn i64_add(&mut self)   { self.emit_binop_64(arm64_enc::add_reg_64); }
    pub fn i64_sub(&mut self)   { self.emit_binop_64(arm64_enc::sub_reg_64); }
    pub fn i64_mul(&mut self)   { self.emit_binop_64(arm64_enc::mul_64); }
    pub fn i64_and(&mut self)   { self.emit_binop_64(arm64_enc::and_reg_64); }
    pub fn i64_or(&mut self)    { self.emit_binop_64(arm64_enc::orr_reg_64); }
    pub fn i64_xor(&mut self)   { self.emit_binop_64(arm64_enc::eor_reg_64); }
    pub fn i64_shl(&mut self)   { self.emit_binop_64(arm64_enc::lslv_64); }
    pub fn i64_shr_u(&mut self) { self.emit_binop_64(arm64_enc::lsrv_64); }
    pub fn i64_shr_s(&mut self) { self.emit_binop_64(arm64_enc::asrv_64); }
    pub fn i64_rotr(&mut self)  { self.emit_binop_64(arm64_enc::rorv_64); }

    pub fn i64_rotl(&mut self) {
        let d = self.dv();
        let lhs = tos_reg(d, 2);
        let rhs = tos_reg(d, 1);
        // rotl(a, b) = rotr(a, -b)
        self.buf.emit(arm64_enc::neg_64(Reg::TMP0, rhs));
        self.buf.emit(arm64_enc::rorv_64(lhs, lhs, Reg::TMP0));
        self.height -= 1;
    }

    // ==================== i32 Comparisons (pop2_push1) ====================

    fn emit_cmp_32(&mut self, cond: Cond) {
        let d = self.dv();
        let lhs = tos_reg(d, 2);
        let rhs = tos_reg(d, 1);
        self.buf.emit(arm64_enc::subs_reg_32(Reg::XZR, lhs, rhs));
        self.buf.emit(arm64_enc::cset_32(lhs, cond));
        self.height -= 1;
    }

    pub fn i32_eq(&mut self)   { self.emit_cmp_32(Cond::EQ); }
    pub fn i32_ne(&mut self)   { self.emit_cmp_32(Cond::NE); }
    pub fn i32_lt_s(&mut self) { self.emit_cmp_32(Cond::LT); }
    pub fn i32_lt_u(&mut self) { self.emit_cmp_32(Cond::LO); }
    pub fn i32_gt_s(&mut self) { self.emit_cmp_32(Cond::GT); }
    pub fn i32_gt_u(&mut self) { self.emit_cmp_32(Cond::HI); }
    pub fn i32_le_s(&mut self) { self.emit_cmp_32(Cond::LE); }
    pub fn i32_le_u(&mut self) { self.emit_cmp_32(Cond::LS); }
    pub fn i32_ge_s(&mut self) { self.emit_cmp_32(Cond::GE); }
    pub fn i32_ge_u(&mut self) { self.emit_cmp_32(Cond::HS); }

    // ==================== i64 Comparisons (pop2_push1) ====================

    fn emit_cmp_64(&mut self, cond: Cond) {
        let d = self.dv();
        let lhs = tos_reg(d, 2);
        let rhs = tos_reg(d, 1);
        self.buf.emit(arm64_enc::subs_reg_64(Reg::XZR, lhs, rhs));
        self.buf.emit(arm64_enc::cset_64(lhs, cond));
        self.height -= 1;
    }

    pub fn i64_eq(&mut self)   { self.emit_cmp_64(Cond::EQ); }
    pub fn i64_ne(&mut self)   { self.emit_cmp_64(Cond::NE); }
    pub fn i64_lt_s(&mut self) { self.emit_cmp_64(Cond::LT); }
    pub fn i64_lt_u(&mut self) { self.emit_cmp_64(Cond::LO); }
    pub fn i64_gt_s(&mut self) { self.emit_cmp_64(Cond::GT); }
    pub fn i64_gt_u(&mut self) { self.emit_cmp_64(Cond::HI); }
    pub fn i64_le_s(&mut self) { self.emit_cmp_64(Cond::LE); }
    pub fn i64_le_u(&mut self) { self.emit_cmp_64(Cond::LS); }
    pub fn i64_ge_s(&mut self) { self.emit_cmp_64(Cond::GE); }
    pub fn i64_ge_u(&mut self) { self.emit_cmp_64(Cond::HS); }

    // ==================== Unary i32 ops (pop1_push1) ====================

    pub fn i32_eqz(&mut self) {
        let src = self.slot_gpr_source(1);
        let dst = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::cmp_imm_32(src, 0));
        self.buf.emit(arm64_enc::cset_32(dst, Cond::EQ));
        self.slots[0] = SlotLoc::canonical();
        // height unchanged
    }

    pub fn i32_clz(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::clz_32(src, src));
    }

    pub fn i32_ctz(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::rbit_32(src, src));
        self.buf.emit(arm64_enc::clz_32(src, src));
    }

    // ==================== Unary i64 ops (pop1_push1) ====================

    pub fn i64_eqz(&mut self) {
        let src = self.slot_gpr_source(1);
        let dst = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::cmp_imm_64(src, 0));
        self.buf.emit(arm64_enc::cset_64(dst, Cond::EQ));
        self.slots[0] = SlotLoc::canonical();
    }

    pub fn i64_clz(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::clz_64(src, src));
    }

    pub fn i64_ctz(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::rbit_64(src, src));
        self.buf.emit(arm64_enc::clz_64(src, src));
    }

    // ==================== Constants (push1) ====================

    pub fn i32_const(&mut self, value: u32) {
        self.push_materialized_slot();
        let dst = tos_reg(self.dv(), 1);
        materialize_u32(self.buf, dst, value);
    }

    pub fn i64_const(&mut self, value: u64) {
        self.push_materialized_slot();
        let dst = tos_reg(self.dv(), 1);
        materialize_u64(self.buf, dst, value);
    }

    // ==================== Hot locals (push1 / pop1) ====================

    pub fn local_get_ln(&mut self, n: u8) {
        self.push_gpr_alias_slot(LOCAL_REGS[n as usize], self.local_fp_aliases[n as usize]);
    }

    pub fn local_set_ln(&mut self, n: u8) {
        let dst = LOCAL_REGS[n as usize];
        self.copy_slot_to_gpr(1, dst);
        self.local_fp_aliases[n as usize] = self.slots[0].fp;
        self.height -= 1;
        self.shift_slots_for_pop();
    }

    pub fn local_tee_ln(&mut self, n: u8) {
        let dst = LOCAL_REGS[n as usize];
        self.copy_slot_to_gpr(1, dst);
        self.local_fp_aliases[n as usize] = self.slots[0].fp;
        self.slots[0].gpr = Some(GprLoc::Alias(dst));
        // height unchanged (tee keeps value on stack)
    }

    // ==================== Non-hot locals (push1 / pop1) ====================

    pub fn local_get(&mut self, idx: u16) {
        self.push_materialized_slot();
        let dst = tos_reg(self.dv(), 1);
        let scaled = idx as u32; // ldr_64 imm12 is pre-scaled by 8
        self.buf.emit(arm64_enc::ldr_64(dst, Reg::FP, scaled));
    }

    pub fn local_set(&mut self, idx: u16) {
        let src = match self.slots[0].gpr {
            Some(_) => self.resolve_tos_reg(1),
            None => {
                self.copy_slot_to_gpr(1, Reg::TMP0);
                Reg::TMP0
            }
        };
        let scaled = idx as u32;
        self.buf.emit(arm64_enc::str_64(src, Reg::FP, scaled));
        self.height -= 1;
        self.shift_slots_for_pop();
    }

    pub fn local_tee(&mut self, idx: u16) {
        let src = match self.slots[0].gpr {
            Some(_) => self.resolve_tos_reg(1),
            None => {
                self.copy_slot_to_gpr(1, Reg::TMP0);
                Reg::TMP0
            }
        };
        let scaled = idx as u32;
        self.buf.emit(arm64_enc::str_64(src, Reg::FP, scaled));
        // height unchanged (tee keeps value on stack)
    }

    // ==================== Hot local init prologue ====================

    /// Perform the InitLocals hot-local swaps and fill L0-L2.
    ///
    /// This mirrors `impl_init_locals` in `handlers_c/const_local.c`:
    /// each `Kn` swaps `fp[n]` with `fp[Kn]`, then fills `Ln` from `fp[n]`.
    pub fn init_locals(&mut self, k0: u16, k1: u16, k2: u16) {
        self.init_one_hot_local(0, k0, Reg::L0);
        self.init_one_hot_local(1, k1, Reg::L1);
        self.init_one_hot_local(2, k2, Reg::L2);
    }

    fn init_one_hot_local(&mut self, slot: u16, target: u16, reg: Reg) {
        if target == slot {
            self.buf.emit(arm64_enc::ldr_64(reg, Reg::FP, target as u32));
            return;
        }

        self.buf.emit(arm64_enc::ldr_64(Reg::TMP0, Reg::FP, slot as u32));
        self.buf.emit(arm64_enc::ldr_64(reg, Reg::FP, target as u32));
        self.buf.emit(arm64_enc::str_64(Reg::TMP0, Reg::FP, target as u32));
        self.buf.emit(arm64_enc::str_64(reg, Reg::FP, slot as u32));
    }

    // ==================== Drop (pop1, no code) ====================

    pub fn drop_val(&mut self) {
        self.height -= 1;
        self.shift_slots_for_pop();
        // No code emitted — value is simply forgotten
    }

    // ==================== TOS Spill/Fill ====================

    /// Spill `count` TOS registers to memory at fp[slot..slot-count+1].
    ///
    /// The `variant` (from the IrOp) determines which physical registers to store.
    /// Height is unchanged (spill/fill have (0,0) stack effect in IR).
    pub fn spill(&mut self, slot: u16, count: u8, variant: u8) {
        for i in 0..count as u32 {
            let reg = tos_reg(variant, (i + 1) as u8);
            self.buf.emit(arm64_enc::str_64(reg, Reg::FP, slot as u32 - i));
        }
    }

    /// Fill `count` TOS registers from memory at fp[slot..slot-count+1].
    ///
    /// The `variant` (from the IrOp) determines which physical registers to load.
    /// Height is unchanged (spill/fill have (0,0) stack effect in IR).
    pub fn fill(&mut self, slot: u16, count: u8, variant: u8) {
        for i in 0..count as u32 {
            let reg = tos_reg(variant, (i + 1) as u8);
            self.buf.emit(arm64_enc::ldr_64(reg, Reg::FP, slot as u32 - i));
        }
    }

    // ==================== Memory bounds check (shared) ====================

    /// Emit bounds check prologue for a memory access.
    ///
    /// Computes effective address `ea = u32(addr_reg) + offset`, checks
    /// `ea + byte_count <= mem0_size`, and branches to the trap stub on OOB.
    ///
    /// After this: TMP0 = ea, NH (x1) = mem0_base.
    fn emit_mem_bounds_check(&mut self, addr_reg: Reg, offset: u32, byte_count: u32) {
        // Truncate addr to u32 (zero-extends to 64-bit)
        self.buf.emit(arm64_enc::mov_reg_32(Reg::TMP0, addr_reg));

        // Add wasm memarg offset
        if offset > 0 {
            if offset <= 4095 {
                self.buf.emit(arm64_enc::add_imm_64(Reg::TMP0, Reg::TMP0, offset));
            } else {
                materialize_u32(self.buf, Reg::TMP1, offset);
                self.buf.emit(arm64_enc::add_reg_64(Reg::TMP0, Reg::TMP0, Reg::TMP1));
            }
        }

        // ea_end = ea + byte_count
        self.buf.emit(arm64_enc::add_imm_64(Reg::TMP1, Reg::TMP0, byte_count));

        // Load mem0_size
        self.buf.emit(arm64_enc::ldr_64(Reg::NH, Reg::CTX,
            emit::ctx_offset::MEM0_SIZE / 8));

        // CMP ea_end, mem0_size (sets flags for unsigned comparison)
        self.buf.emit(arm64_enc::subs_reg_64(Reg::XZR, Reg::TMP1, Reg::NH));

        // B.HI to trap stub (placeholder offset, patched by emit_trap_stub_if_needed)
        let patch_off = self.buf.emit(arm64_enc::b_cond(Cond::HI, 0));
        self.trap_patches.push(patch_off);

        // Load mem0_base
        self.buf.emit(arm64_enc::ldr_64(Reg::NH, Reg::CTX,
            emit::ctx_offset::MEM0_BASE / 8));
    }

    // ==================== Memory loads (pop1_push1) ====================

    pub fn i32_load(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 4);
        self.buf.emit(arm64_enc::ldr_32_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i32_load8_u(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 1);
        self.buf.emit(arm64_enc::ldrb_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i32_load8_s(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 1);
        self.buf.emit(arm64_enc::ldrb_reg(reg, Reg::NH, Reg::TMP0));
        self.buf.emit(arm64_enc::sxtb_32(reg, reg));
    }

    pub fn i32_load16_u(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 2);
        self.buf.emit(arm64_enc::ldrh_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i32_load16_s(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 2);
        self.buf.emit(arm64_enc::ldrh_reg(reg, Reg::NH, Reg::TMP0));
        self.buf.emit(arm64_enc::sxth_32(reg, reg));
    }

    pub fn i64_load(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 8);
        self.buf.emit(arm64_enc::ldr_64_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i64_load8_u(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 1);
        self.buf.emit(arm64_enc::ldrb_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i64_load8_s(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 1);
        self.buf.emit(arm64_enc::ldrb_reg(reg, Reg::NH, Reg::TMP0));
        self.buf.emit(arm64_enc::sxtb_64(reg, reg));
    }

    pub fn i64_load16_u(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 2);
        self.buf.emit(arm64_enc::ldrh_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i64_load16_s(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 2);
        self.buf.emit(arm64_enc::ldrh_reg(reg, Reg::NH, Reg::TMP0));
        self.buf.emit(arm64_enc::sxth_64(reg, reg));
    }

    pub fn i64_load32_u(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 4);
        self.buf.emit(arm64_enc::ldr_32_reg(reg, Reg::NH, Reg::TMP0));
    }

    pub fn i64_load32_s(&mut self, offset: u32) {
        let reg = tos_reg(self.dv(), 1);
        self.emit_mem_bounds_check(reg, offset, 4);
        self.buf.emit(arm64_enc::ldr_32_reg(reg, Reg::NH, Reg::TMP0));
        self.buf.emit(arm64_enc::sxtw(reg, reg));
    }

    // ==================== Memory stores (pop2_push0) ====================

    pub fn i32_store(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 4);
        self.buf.emit(arm64_enc::str_32_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    pub fn i32_store8(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 1);
        self.buf.emit(arm64_enc::strb_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    pub fn i32_store16(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 2);
        self.buf.emit(arm64_enc::strh_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    pub fn i64_store(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 8);
        self.buf.emit(arm64_enc::str_64_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    pub fn i64_store8(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 1);
        self.buf.emit(arm64_enc::strb_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    pub fn i64_store16(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 2);
        self.buf.emit(arm64_enc::strh_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    pub fn i64_store32(&mut self, offset: u32) {
        let d = self.dv();
        let val_reg = tos_reg(d, 1);
        let addr_reg = tos_reg(d, 2);
        self.emit_mem_bounds_check(addr_reg, offset, 4);
        self.buf.emit(arm64_enc::str_32_reg(val_reg, Reg::NH, Reg::TMP0));
        self.height -= 2;
    }

    // ==================== Float constants (push1) ====================

    pub fn f32_const(&mut self, bits: u32) {
        self.push_materialized_slot();
        let dst = tos_reg(self.dv(), 1);
        materialize_u32(self.buf, dst, bits);
    }

    pub fn f64_const(&mut self, bits: u64) {
        self.push_materialized_slot();
        let dst = tos_reg(self.dv(), 1);
        materialize_u64(self.buf, dst, bits);
    }

    // ==================== Type conversions (pop1_push1, no height change) ====================

    pub fn i32_wrap_i64(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::mov_reg_32(src, src));
    }

    pub fn i64_extend_i32_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::sxtw(src, src));
    }

    pub fn i64_extend_i32_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::mov_reg_32(src, src));
    }

    // ==================== Sign extensions (pop1_push1, no height change) ====================

    pub fn i32_extend8_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::sxtb_32(src, src));
    }

    pub fn i32_extend16_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::sxth_32(src, src));
    }

    pub fn i64_extend8_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::sxtb_64(src, src));
    }

    pub fn i64_extend16_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::sxth_64(src, src));
    }

    pub fn i64_extend32_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::sxtw(src, src));
    }

    // ==================== Select (pop3_push1) ====================

    pub fn select(&mut self) {
        let d = self.dv();
        let cond = tos_reg(d, 1);
        let val2 = tos_reg(d, 2);
        let val1 = tos_reg(d, 3);
        self.buf.emit(arm64_enc::cmp_imm_32(cond, 0));
        self.buf.emit(arm64_enc::csel_64(val1, val1, val2, Cond::NE));
        self.height -= 2;
    }

    // ==================== f64 binary ops (pop2_push1) ====================

    /// Scratch FP register indices for unary ops and conversions.
    const FP0: u32 = 0;
    const FP1: u32 = 1;

    fn emit_f64_binop(&mut self, f: fn(u32, u32, u32) -> u32) {
        let lhs = self.ensure_tos_f64(2);
        let rhs = self.ensure_tos_f64(1);
        let dst = self.alloc_fp_result() as u32;
        self.buf.emit(f(dst, lhs, rhs));
        self.pop2_push_fp(FpLoc {
            reg: dst as u8,
            width: FpWidth::F64,
        });
    }

    pub fn f64_add(&mut self) { self.emit_f64_binop(arm64_enc::fadd_64); }
    pub fn f64_sub(&mut self) { self.emit_f64_binop(arm64_enc::fsub_64); }
    pub fn f64_mul(&mut self) { self.emit_f64_binop(arm64_enc::fmul_64); }
    pub fn f64_div(&mut self) { self.emit_f64_binop(arm64_enc::fdiv_64); }
    pub fn f64_min(&mut self) { self.emit_f64_binop(arm64_enc::fmin_64); }
    pub fn f64_max(&mut self) { self.emit_f64_binop(arm64_enc::fmax_64); }

    // ==================== f32 binary ops (pop2_push1) ====================

    fn emit_f32_binop(&mut self, f: fn(u32, u32, u32) -> u32) {
        let lhs = self.ensure_tos_f32(2);
        let rhs = self.ensure_tos_f32(1);
        let dst = self.alloc_fp_result() as u32;
        self.buf.emit(f(dst, lhs, rhs));
        self.pop2_push_fp(FpLoc {
            reg: dst as u8,
            width: FpWidth::F32,
        });
    }

    pub fn f32_add(&mut self) { self.emit_f32_binop(arm64_enc::fadd_32); }
    pub fn f32_sub(&mut self) { self.emit_f32_binop(arm64_enc::fsub_32); }
    pub fn f32_mul(&mut self) { self.emit_f32_binop(arm64_enc::fmul_32); }
    pub fn f32_div(&mut self) { self.emit_f32_binop(arm64_enc::fdiv_32); }
    pub fn f32_min(&mut self) { self.emit_f32_binop(arm64_enc::fmin_32); }
    pub fn f32_max(&mut self) { self.emit_f32_binop(arm64_enc::fmax_32); }

    // ==================== f64 comparisons (pop2_push1) ====================
    // NaN-aware: use MI/LS instead of LT/LE to get false for unordered

    fn emit_f64_cmp(&mut self, cond: Cond) {
        let lhs = self.ensure_tos_f64(2);
        let rhs = self.ensure_tos_f64(1);
        let dst = tos_reg(depth_variant(self.height - 1), 1);
        self.buf.emit(arm64_enc::fcmp_64(lhs, rhs));
        self.buf.emit(arm64_enc::cset_32(dst, cond));
        self.pop2_push_materialized();
    }

    pub fn f64_eq(&mut self) { self.emit_f64_cmp(Cond::EQ); }
    pub fn f64_ne(&mut self) { self.emit_f64_cmp(Cond::NE); }
    pub fn f64_lt(&mut self) { self.emit_f64_cmp(Cond::MI); }
    pub fn f64_gt(&mut self) { self.emit_f64_cmp(Cond::GT); }
    pub fn f64_le(&mut self) { self.emit_f64_cmp(Cond::LS); }
    pub fn f64_ge(&mut self) { self.emit_f64_cmp(Cond::GE); }

    // ==================== f32 comparisons (pop2_push1) ====================

    fn emit_f32_cmp(&mut self, cond: Cond) {
        let lhs = self.ensure_tos_f32(2);
        let rhs = self.ensure_tos_f32(1);
        let dst = tos_reg(depth_variant(self.height - 1), 1);
        self.buf.emit(arm64_enc::fcmp_32(lhs, rhs));
        self.buf.emit(arm64_enc::cset_32(dst, cond));
        self.pop2_push_materialized();
    }

    pub fn f32_eq(&mut self) { self.emit_f32_cmp(Cond::EQ); }
    pub fn f32_ne(&mut self) { self.emit_f32_cmp(Cond::NE); }
    pub fn f32_lt(&mut self) { self.emit_f32_cmp(Cond::MI); }
    pub fn f32_gt(&mut self) { self.emit_f32_cmp(Cond::GT); }
    pub fn f32_le(&mut self) { self.emit_f32_cmp(Cond::LS); }
    pub fn f32_ge(&mut self) { self.emit_f32_cmp(Cond::GE); }

    // ==================== f64 unary ops (pop1_push1) ====================

    fn emit_f64_unop(&mut self, f: fn(u32, u32) -> u32) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp64(Self::FP0, src));
        self.buf.emit(f(Self::FP0, Self::FP0));
        self.buf.emit(arm64_enc::fmov_fp64_to_gpr(src, Self::FP0));
    }

    pub fn f64_abs(&mut self) { self.emit_f64_unop(arm64_enc::fabs_64); }
    pub fn f64_neg(&mut self) { self.emit_f64_unop(arm64_enc::fneg_64); }
    pub fn f64_sqrt(&mut self) { self.emit_f64_unop(arm64_enc::fsqrt_64); }
    pub fn f64_ceil(&mut self) { self.emit_f64_unop(arm64_enc::frintp_64); }
    pub fn f64_floor(&mut self) { self.emit_f64_unop(arm64_enc::frintm_64); }
    pub fn f64_trunc(&mut self) { self.emit_f64_unop(arm64_enc::frintz_64); }
    pub fn f64_nearest(&mut self) { self.emit_f64_unop(arm64_enc::frintn_64); }

    // ==================== f32 unary ops (pop1_push1) ====================

    fn emit_f32_unop(&mut self, f: fn(u32, u32) -> u32) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp32(Self::FP0, src));
        self.buf.emit(f(Self::FP0, Self::FP0));
        self.buf.emit(arm64_enc::fmov_fp32_to_gpr(src, Self::FP0));
    }

    pub fn f32_abs(&mut self) { self.emit_f32_unop(arm64_enc::fabs_32); }
    pub fn f32_neg(&mut self) { self.emit_f32_unop(arm64_enc::fneg_32); }
    pub fn f32_sqrt(&mut self) { self.emit_f32_unop(arm64_enc::fsqrt_32); }
    pub fn f32_ceil(&mut self) { self.emit_f32_unop(arm64_enc::frintp_32); }
    pub fn f32_floor(&mut self) { self.emit_f32_unop(arm64_enc::frintm_32); }
    pub fn f32_trunc(&mut self) { self.emit_f32_unop(arm64_enc::frintz_32); }
    pub fn f32_nearest(&mut self) { self.emit_f32_unop(arm64_enc::frintn_32); }

    // ==================== Float precision conversions (pop1_push1) ====================

    pub fn f64_promote_f32(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp32(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvt_f32_to_f64(Self::FP0, Self::FP0));
        self.buf.emit(arm64_enc::fmov_fp64_to_gpr(src, Self::FP0));
    }

    pub fn f32_demote_f64(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp64(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvt_f64_to_f32(Self::FP0, Self::FP0));
        self.buf.emit(arm64_enc::fmov_fp32_to_gpr(src, Self::FP0));
    }

    // ==================== Int-to-float conversions (pop1_push1) ====================

    pub fn f32_convert_i32_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::scvtf_s_w(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp32_to_gpr(src, Self::FP0));
    }

    pub fn f32_convert_i32_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::ucvtf_s_w(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp32_to_gpr(src, Self::FP0));
    }

    pub fn f32_convert_i64_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::scvtf_s_x(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp32_to_gpr(src, Self::FP0));
    }

    pub fn f32_convert_i64_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::ucvtf_s_x(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp32_to_gpr(src, Self::FP0));
    }

    pub fn f64_convert_i32_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::scvtf_d_w(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp64_to_gpr(src, Self::FP0));
    }

    pub fn f64_convert_i32_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::ucvtf_d_w(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp64_to_gpr(src, Self::FP0));
    }

    pub fn f64_convert_i64_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::scvtf_d_x(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp64_to_gpr(src, Self::FP0));
    }

    pub fn f64_convert_i64_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::ucvtf_d_x(Self::FP0, src));
        self.buf.emit(arm64_enc::fmov_fp64_to_gpr(src, Self::FP0));
    }

    // ==================== Saturating float-to-int truncation (pop1_push1) ====================
    // ARM64 FCVTZS/FCVTZU naturally saturates and returns 0 for NaN.

    pub fn i32_trunc_sat_f32_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp32(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzs_w_s(src, Self::FP0));
    }

    pub fn i32_trunc_sat_f32_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp32(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzu_w_s(src, Self::FP0));
    }

    pub fn i32_trunc_sat_f64_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp64(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzs_w_d(src, Self::FP0));
    }

    pub fn i32_trunc_sat_f64_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp64(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzu_w_d(src, Self::FP0));
    }

    pub fn i64_trunc_sat_f32_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp32(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzs_x_s(src, Self::FP0));
    }

    pub fn i64_trunc_sat_f32_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp32(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzu_x_s(src, Self::FP0));
    }

    pub fn i64_trunc_sat_f64_s(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp64(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzs_x_d(src, Self::FP0));
    }

    pub fn i64_trunc_sat_f64_u(&mut self) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::fmov_gpr_to_fp64(Self::FP0, src));
        self.buf.emit(arm64_enc::fcvtzu_x_d(src, Self::FP0));
    }

    // ==================== Reinterpret ops (pop1_push1, NOP) ====================
    // These are NOPs — same bit pattern, just reinterpreted.
    // No code emitted, no height change.
}

// ==================== Constant materialization ====================

fn materialize_u32(buf: &mut CodeBuffer, dst: Reg, value: u32) {
    let lo = (value & 0xFFFF) as u16;
    let hi = ((value >> 16) & 0xFFFF) as u16;
    if hi == 0 {
        buf.emit(arm64_enc::movz_32(dst, lo, 0));
    } else if lo == 0 {
        buf.emit(arm64_enc::movz_32(dst, hi, 16));
    } else {
        buf.emit(arm64_enc::movz_32(dst, lo, 0));
        buf.emit(arm64_enc::movk_32(dst, hi, 16));
    }
}

fn materialize_u64(buf: &mut CodeBuffer, dst: Reg, value: u64) {
    if value == 0 {
        buf.emit(arm64_enc::movz_64(dst, 0, 0));
        return;
    }
    let chunks: [(u16, u32); 4] = [
        ((value & 0xFFFF) as u16, 0),
        (((value >> 16) & 0xFFFF) as u16, 16),
        (((value >> 32) & 0xFFFF) as u16, 32),
        (((value >> 48) & 0xFFFF) as u16, 48),
    ];
    let mut first = true;
    for &(chunk, shift) in &chunks {
        if chunk != 0 || first {
            if first {
                buf.emit(arm64_enc::movz_64(dst, chunk, shift));
                first = false;
            } else {
                buf.emit(arm64_enc::movk_64(dst, chunk, shift));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::vm::interp::fast::instruction::Instruction;
    use crate::vm::interp::fast::handlers::{self, OpHandler, NextHandler, run_trampoline};
    use crate::vm::interp::fast::context::Context;
    use super::super::code_buf::CodeBuffer;

    /// Map Reg to TOS index (for test setup).
    fn tos_idx(r: Reg) -> usize {
        match r {
            Reg::T0 => 0, Reg::T1 => 1, Reg::T2 => 2, Reg::T3 => 3,
            _ => panic!("not a TOS register"),
        }
    }

    /// Run a JIT group and return the top-of-stack value after execution.
    ///
    /// `setup_fn` emits opcodes via the JitEmitter.
    /// After execution, the TOS top is stored to fp[0] and returned.
    fn run_jit_test(
        setup_fn: impl FnOnce(&mut JitEmitter),
        initial_height: usize,
        t0: u64, t1: u64, t2: u64, t3: u64,
        l0: u64, l1: u64, l2: u64,
    ) -> u64 {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut emitter = JitEmitter::new(&mut buf, initial_height);
        setup_fn(&mut emitter);

        // Store TOS top to fp[0] for verification
        emitter.materialize_aliases();
        let result_reg = tos_reg(emitter.dv(), 1);
        emitter.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));

        // Finish: dispatch stub + trap stub if needed
        let start = emitter.finish();

        let total_len = buf.len();
        buf.finish_write(0, total_len);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                l0, l1, l2,
                t0, t1, t2, t3,
                nh,
            );
        }

        frame[0]
    }

    /// Run a JIT group and return a hot local register value after execution.
    /// Stores the specified local register to fp[0] instead of TOS top.
    fn run_jit_test_local(
        setup_fn: impl FnOnce(&mut JitEmitter),
        initial_height: usize,
        local_reg: Reg,
        t0: u64, t1: u64, t2: u64, t3: u64,
        l0: u64, l1: u64, l2: u64,
    ) -> u64 {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut emitter = JitEmitter::new(&mut buf, initial_height);
        setup_fn(&mut emitter);

        // Store the local register to fp[0] for verification
        emitter.materialize_aliases();
        emitter.emit_raw(arm64_enc::str_64(local_reg, Reg::FP, 0));

        let start = emitter.finish();

        let total_len = buf.len();
        buf.finish_write(0, total_len);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                l0, l1, l2,
                t0, t1, t2, t3,
                nh,
            );
        }

        frame[0]
    }

    /// Run a JIT group with linear memory and return TOS top.
    fn run_jit_mem_test(
        setup_fn: impl FnOnce(&mut JitEmitter),
        initial_height: usize,
        t0: u64, t1: u64, t2: u64, t3: u64,
        l0: u64, l1: u64, l2: u64,
        mem: &mut [u8],
    ) -> u64 {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut emitter = JitEmitter::new(&mut buf, initial_height);
        setup_fn(&mut emitter);

        let result_reg = tos_reg(emitter.dv(), 1);
        emitter.emit_raw(arm64_enc::str_64(result_reg, Reg::FP, 0));
        let start = emitter.finish();

        let total_len = buf.len();
        buf.finish_write(0, total_len);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            mem.as_mut_ptr(), mem.len() as u64,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                l0, l1, l2,
                t0, t1, t2, t3,
                nh,
            );
        }

        frame[0]
    }

    /// Run a JIT group with linear memory. Returns (frame[0], trap_message_ptr).
    fn run_jit_mem_store_test(
        setup_fn: impl FnOnce(&mut JitEmitter),
        initial_height: usize,
        t0: u64, t1: u64, t2: u64, t3: u64,
        mem: &mut [u8],
    ) -> *const core::ffi::c_char {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let mut emitter = JitEmitter::new(&mut buf, initial_height);
        setup_fn(&mut emitter);
        let start = emitter.finish();

        let total_len = buf.len();
        buf.finish_write(0, total_len);

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            mem.as_mut_ptr(), mem.len() as u64,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                t0, t1, t2, t3,
                nh,
            );
        }

        ctx.hot.trap_message
    }

    // ==================== Unit tests: depth_variant / tos_reg ====================

    #[test]
    fn test_depth_variant() {
        assert_eq!(depth_variant(0), 1);
        assert_eq!(depth_variant(1), 1);
        assert_eq!(depth_variant(2), 2);
        assert_eq!(depth_variant(3), 3);
        assert_eq!(depth_variant(4), 4);
        assert_eq!(depth_variant(5), 1);
        assert_eq!(depth_variant(6), 2);
    }

    #[test]
    fn test_tos_reg_selection() {
        // D1: pos1=T0, pos2=T3
        assert_eq!(tos_reg(1, 1), Reg::T0);
        assert_eq!(tos_reg(1, 2), Reg::T3);
        // D2: pos1=T1, pos2=T0
        assert_eq!(tos_reg(2, 1), Reg::T1);
        assert_eq!(tos_reg(2, 2), Reg::T0);
        // D3: pos1=T2, pos2=T1
        assert_eq!(tos_reg(3, 1), Reg::T2);
        assert_eq!(tos_reg(3, 2), Reg::T1);
        // D4: pos1=T3, pos2=T2
        assert_eq!(tos_reg(4, 1), Reg::T3);
        assert_eq!(tos_reg(4, 2), Reg::T2);
    }

    // ==================== Step 4: Single-op i32 arithmetic tests ====================

    #[test]
    fn test_i32_add_all_depths() {
        // Binary ops need height >= 2. Heights 2,3,4,5 cover D2,D3,D4,D1.
        for height in 2usize..=5 {
            let d = depth_variant(height);
            let lhs_reg = tos_reg(d, 2);
            let rhs_reg = tos_reg(d, 1);

            let mut t = [0u64; 4];
            t[tos_idx(lhs_reg)] = 5;
            t[tos_idx(rhs_reg)] = 3;

            let result = run_jit_test(
                |e| e.i32_add(),
                height,
                t[0], t[1], t[2], t[3],
                0, 0, 0,
            );
            assert_eq!(result, 8, "i32_add failed at D{} (height={})", d, height);
        }
    }

    #[test]
    fn test_i32_sub_d2() {
        // D2: lhs=T0(pos2), rhs=T1(pos1). 10 - 3 = 7
        let result = run_jit_test(|e| e.i32_sub(), 2, 10, 3, 0, 0, 0, 0, 0);
        assert_eq!(result, 7);
    }

    #[test]
    fn test_i32_mul_d2() {
        let result = run_jit_test(|e| e.i32_mul(), 2, 6, 7, 0, 0, 0, 0, 0);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_i32_and_d2() {
        let result = run_jit_test(|e| e.i32_and(), 2, 0xFF, 0x0F, 0, 0, 0, 0, 0);
        assert_eq!(result, 0x0F);
    }

    #[test]
    fn test_i32_or_d2() {
        let result = run_jit_test(|e| e.i32_or(), 2, 0xF0, 0x0F, 0, 0, 0, 0, 0);
        assert_eq!(result, 0xFF);
    }

    #[test]
    fn test_i32_xor_d2() {
        let result = run_jit_test(|e| e.i32_xor(), 2, 0xFF, 0x0F, 0, 0, 0, 0, 0);
        assert_eq!(result, 0xF0);
    }

    #[test]
    fn test_i32_shl_d2() {
        let result = run_jit_test(|e| e.i32_shl(), 2, 1, 4, 0, 0, 0, 0, 0);
        assert_eq!(result, 16);
    }

    #[test]
    fn test_i32_shr_u_d2() {
        let result = run_jit_test(|e| e.i32_shr_u(), 2, 16, 2, 0, 0, 0, 0, 0);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_i32_shr_s_d2() {
        // -16 >> 2 signed = -4 (0xFFFFFFFC as i32)
        // But we're using 32-bit ops, so input must be in lower 32 bits
        let result = run_jit_test(|e| e.i32_shr_s(), 2, (-16i32 as u32) as u64, 2, 0, 0, 0, 0, 0);
        assert_eq!(result as u32, (-4i32 as u32));
    }

    #[test]
    fn test_i32_rotr_d2() {
        // rotr(0x80000001, 1) = 0xC0000000
        let result = run_jit_test(|e| e.i32_rotr(), 2, 0x80000001, 1, 0, 0, 0, 0, 0);
        assert_eq!(result as u32, 0xC0000000);
    }

    #[test]
    fn test_i32_rotl_d2() {
        // rotl(0x80000001, 1) = 0x00000003
        let result = run_jit_test(|e| e.i32_rotl(), 2, 0x80000001, 1, 0, 0, 0, 0, 0);
        assert_eq!(result as u32, 0x00000003);
    }

    // ==================== Step 4: i64 arithmetic tests ====================

    #[test]
    fn test_i64_add_d2() {
        let result = run_jit_test(|e| e.i64_add(), 2, 0x100000000, 0x200000000, 0, 0, 0, 0, 0);
        assert_eq!(result, 0x300000000);
    }

    #[test]
    fn test_i64_sub_d2() {
        let result = run_jit_test(|e| e.i64_sub(), 2, 0x300000000, 0x100000000, 0, 0, 0, 0, 0);
        assert_eq!(result, 0x200000000);
    }

    #[test]
    fn test_i64_mul_d2() {
        let result = run_jit_test(|e| e.i64_mul(), 2, 0x10000, 0x10000, 0, 0, 0, 0, 0);
        assert_eq!(result, 0x100000000);
    }

    #[test]
    fn test_i64_and_d2() {
        let result = run_jit_test(|e| e.i64_and(), 2, 0xFF00FF00FF00FF00, 0x0F0F0F0F0F0F0F0F, 0, 0, 0, 0, 0);
        assert_eq!(result, 0x0F000F000F000F00);
    }

    #[test]
    fn test_i64_rotl_d2() {
        // rotl(1, 63) = 0x8000000000000000
        let result = run_jit_test(|e| e.i64_rotl(), 2, 1, 63, 0, 0, 0, 0, 0);
        assert_eq!(result, 0x8000000000000000);
    }

    // ==================== Step 4: i32 comparison tests ====================

    #[test]
    fn test_i32_eq_d2() {
        let result = run_jit_test(|e| e.i32_eq(), 2, 5, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_eq(), 2, 5, 3, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_ne_d2() {
        let result = run_jit_test(|e| e.i32_ne(), 2, 5, 3, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_ne(), 2, 5, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_lt_s_d2() {
        let result = run_jit_test(|e| e.i32_lt_s(), 2, 3, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_lt_s(), 2, 5, 3, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_lt_u_d2() {
        // Unsigned: 0xFFFFFFFF > 1
        let result = run_jit_test(|e| e.i32_lt_u(), 2, 1, 0xFFFFFFFF, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_i32_gt_s_d2() {
        let result = run_jit_test(|e| e.i32_gt_s(), 2, 5, 3, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_gt_s(), 2, 3, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_le_s_d2() {
        let result = run_jit_test(|e| e.i32_le_s(), 2, 3, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_le_s(), 2, 5, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_le_s(), 2, 6, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_ge_s_d2() {
        let result = run_jit_test(|e| e.i32_ge_s(), 2, 5, 3, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_ge_s(), 2, 5, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_ge_s(), 2, 3, 5, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    // ==================== Step 4: Unary ops ====================

    #[test]
    fn test_i32_eqz_d1() {
        let result = run_jit_test(|e| e.i32_eqz(), 1, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i32_eqz(), 1, 5, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_clz_d1() {
        let result = run_jit_test(|e| e.i32_clz(), 1, 1, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 31);
        let result = run_jit_test(|e| e.i32_clz(), 1, 0x80000000, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i32_ctz_d1() {
        let result = run_jit_test(|e| e.i32_ctz(), 1, 0x80, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 7);
        let result = run_jit_test(|e| e.i32_ctz(), 1, 1, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i64_eqz_d1() {
        let result = run_jit_test(|e| e.i64_eqz(), 1, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 1);
        let result = run_jit_test(|e| e.i64_eqz(), 1, 0x100000000, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i64_clz_d1() {
        let result = run_jit_test(|e| e.i64_clz(), 1, 1, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 63);
    }

    #[test]
    fn test_i64_ctz_d1() {
        let result = run_jit_test(|e| e.i64_ctz(), 1, 0x8000000000000000, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 63);
    }

    // ==================== Step 4: Constants ====================

    #[test]
    fn test_i32_const_d0() {
        let result = run_jit_test(|e| e.i32_const(42), 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_i32_const_large() {
        let result = run_jit_test(|e| e.i32_const(0xDEAD_BEEF), 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0xDEAD_BEEF);
    }

    #[test]
    fn test_i32_const_hi_only() {
        let result = run_jit_test(|e| e.i32_const(0xFFFF0000), 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0xFFFF0000);
    }

    #[test]
    fn test_i64_const_zero() {
        let result = run_jit_test(|e| e.i64_const(0), 0, 0xDEAD, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_i64_const_large() {
        let result = run_jit_test(|e| e.i64_const(0xDEAD_BEEF_CAFE_BABE), 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(result, 0xDEAD_BEEF_CAFE_BABE);
    }

    // ==================== Step 4: Hot locals ====================

    #[test]
    fn test_local_get_l0() {
        let result = run_jit_test(|e| e.local_get_ln(0), 0, 0, 0, 0, 0, 42, 0, 0);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_local_get_l1() {
        let result = run_jit_test(|e| e.local_get_ln(1), 0, 0, 0, 0, 0, 0, 99, 0);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_local_get_l2() {
        let result = run_jit_test(|e| e.local_get_ln(2), 0, 0, 0, 0, 0, 0, 0, 77);
        assert_eq!(result, 77);
    }

    #[test]
    fn test_local_set_l0() {
        // Push 55, set to l0, verify l0 = 55
        let result = run_jit_test_local(
            |e| { e.i32_const(55); e.local_set_ln(0); },
            0, Reg::L0,
            0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 55);
    }

    #[test]
    fn test_local_tee_l0() {
        // Push 77, tee to l0 → l0=77 AND TOS still has 77
        let result = run_jit_test(
            |e| { e.i32_const(77); e.local_tee_ln(0); },
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 77); // TOS still 77
    }

    #[test]
    fn test_init_locals_swaps_frame_and_fills_hot_registers() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start;
        {
            let mut e = JitEmitter::new(&mut buf, 0);
            e.init_locals(5, 3, 2);
            e.emit_raw(arm64_enc::str_64(Reg::L0, Reg::FP, 8));
            e.emit_raw(arm64_enc::str_64(Reg::L1, Reg::FP, 9));
            e.emit_raw(arm64_enc::str_64(Reg::L2, Reg::FP, 10));
            start = e.finish();
        }

        buf.finish_write(0, buf.len());

        let handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        let mut frame = [0u64; 32];
        frame[0] = 10;
        frame[1] = 11;
        frame[2] = 12;
        frame[3] = 30;
        frame[5] = 50;

        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx,
                pc,
                frame.as_mut_ptr(),
                777,
                888,
                999,
                0,
                0,
                0,
                0,
                nh,
            );
        }

        assert_eq!(frame[0], 50, "fp[0] should hold the swapped hot local");
        assert_eq!(frame[5], 10, "fp[5] should receive the original fp[0]");
        assert_eq!(frame[1], 30, "fp[1] should hold the swapped hot local");
        assert_eq!(frame[3], 11, "fp[3] should receive the original fp[1]");
        assert_eq!(frame[2], 12, "fp[2] should remain unchanged");
        assert_eq!(frame[8], 50, "l0 should be filled from fp[0]");
        assert_eq!(frame[9], 30, "l1 should be filled from fp[1]");
        assert_eq!(frame[10], 12, "l2 should be filled from fp[2]");
    }

    // ==================== Step 4: Drop ====================

    #[test]
    fn test_drop_val() {
        // Push 99, push 42, drop → TOS = 99
        let result = run_jit_test(
            |e| { e.i32_const(99); e.i32_const(42); e.drop_val(); },
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 99);
    }

    // ==================== Step 5: Multi-instruction group tests ====================

    #[test]
    fn test_group_const_const_add() {
        let result = run_jit_test(
            |e| { e.i32_const(5); e.i32_const(3); e.i32_add(); },
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 8);
    }

    #[test]
    fn test_group_local_get_add() {
        let result = run_jit_test(
            |e| { e.local_get_ln(0); e.local_get_ln(1); e.i32_add(); },
            0, 0, 0, 0, 0, 10, 20, 0,
        );
        assert_eq!(result, 30);
    }

    #[test]
    fn test_group_const_mul_const_add() {
        // 2 * 3 + 4 = 10
        let result = run_jit_test(
            |e| {
                e.i32_const(2);
                e.i32_const(3);
                e.i32_mul();
                e.i32_const(4);
                e.i32_add();
            },
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 10);
    }

    #[test]
    fn test_group_local_get_const_add_local_set() {
        // l0 = l0 + 1 (increment l0)
        // Start: l0=10. After: l0=11.
        let result = run_jit_test(
            |e| {
                e.local_get_ln(0);  // push l0 (=10)
                e.i32_const(1);     // push 1
                e.i32_add();        // pop 2, push 10+1=11
                e.local_set_ln(0);  // pop 11, store to l0
                e.local_get_ln(0);  // push l0 (=11) for result capture
            },
            0, 0, 0, 0, 0, 10, 0, 0,
        );
        assert_eq!(result, 11);
    }

    #[test]
    fn test_group_comparison_chain() {
        // l0 < l1 (10 < 20 = 1)
        let result = run_jit_test(
            |e| { e.local_get_ln(0); e.local_get_ln(1); e.i32_lt_s(); },
            0, 0, 0, 0, 0, 10, 20, 0,
        );
        assert_eq!(result, 1);
    }

    #[test]
    fn test_group_complex_expression() {
        // (l0 + l1) * (l0 - l1)  with l0=10, l1=3
        // = 13 * 7 = 91
        let result = run_jit_test(
            |e| {
                e.local_get_ln(0);  // push 10
                e.local_get_ln(1);  // push 3
                e.i32_add();        // push 13
                e.local_get_ln(0);  // push 10
                e.local_get_ln(1);  // push 3
                e.i32_sub();        // push 7
                e.i32_mul();        // push 91
            },
            0, 0, 0, 0, 0, 10, 3, 0,
        );
        assert_eq!(result, 91);
    }

    #[test]
    fn test_group_depth_cycling() {
        // Push 4 constants (height goes 0→1→2→3→4), depth cycles D1→D2→D3→D4
        // Then 3 adds reduce: height 4→3→2→1
        // Result should be 10+20+30+40 = 100
        let result = run_jit_test(
            |e| {
                e.i32_const(10);
                e.i32_const(20);
                e.i32_const(30);
                e.i32_const(40);
                e.i32_add();  // 30+40=70
                e.i32_add();  // 20+70=90
                e.i32_add();  // 10+90=100
            },
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 100);
    }

    #[test]
    fn test_group_i64_expression() {
        // l0 + l1 with 64-bit values
        let result = run_jit_test(
            |e| { e.local_get_ln(0); e.local_get_ln(1); e.i64_add(); },
            0, 0, 0, 0, 0, 0x100000000, 0x200000000, 0,
        );
        assert_eq!(result, 0x300000000);
    }

    #[test]
    fn test_group_mixed_const_drop() {
        // const(1), const(2), const(3), drop, add → 1 + 2 = 3
        let result = run_jit_test(
            |e| {
                e.i32_const(1);
                e.i32_const(2);
                e.i32_const(3);
                e.drop_val();
                e.i32_add();
            },
            0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(result, 3);
    }

    #[test]
    fn test_group_local_tee_chain() {
        // Push 42, tee to l0, push l1(=10), add → 42 + 10 = 52
        let result = run_jit_test(
            |e| {
                e.i32_const(42);
                e.local_tee_ln(0);
                e.local_get_ln(1);
                e.i32_add();
            },
            0, 0, 0, 0, 0, 0, 10, 0,
        );
        assert_eq!(result, 52);
    }

    #[test]
    fn test_group_hot_f64_mul() {
        let result = run_jit_test(
            |e| {
                e.local_get_ln(0);
                e.local_get_ln(0);
                e.f64_mul();
            },
            0,
            0,
            0,
            0,
            0,
            3.0f64.to_bits(),
            0,
            0,
        );
        assert_eq!(result, 9.0f64.to_bits());
    }

    #[test]
    fn test_group_hot_f64_tee_chain() {
        let result = run_jit_test(
            |e| {
                e.local_get_ln(0);
                e.local_get_ln(0);
                e.f64_mul();
                e.local_tee_ln(1);
                e.local_get_ln(1);
                e.f64_add();
            },
            0,
            0,
            0,
            0,
            0,
            3.0f64.to_bits(),
            0,
            0,
        );
        assert_eq!(result, 18.0f64.to_bits());
    }

    #[test]
    fn test_group_hot_f64_tee_store_chain() {
        let result = run_jit_test(
            |e| {
                e.local_get_ln(0);
                e.local_get_ln(0);
                e.f64_mul();
                e.local_tee_ln(1);
                e.local_get_ln(1);
                e.f64_add();
                e.local_set(1);
                e.local_get(1);
            },
            0,
            0,
            0,
            0,
            0,
            3.0f64.to_bits(),
            0,
            0,
        );
        assert_eq!(result, 18.0f64.to_bits());
    }

    #[test]
    fn test_group_eqz_after_sub() {
        // l0 - l1, eqz: 5 - 5 = 0, eqz(0) = 1
        let result = run_jit_test(
            |e| {
                e.local_get_ln(0);
                e.local_get_ln(1);
                e.i32_sub();
                e.i32_eqz();
            },
            0, 0, 0, 0, 0, 5, 5, 0,
        );
        assert_eq!(result, 1);
    }

    // ==================== Step 9: br_if_simple tests ====================

    #[test]
    fn test_br_if_simple_fallthrough() {
        // Condition = 0 → fallthrough (linear dispatch to inst[1]).
        // height=2, D=2: pos1(top)=T1(cond=0), pos2=T0(value=42)
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start;
        {
            let mut e = JitEmitter::new(&mut buf, 2);
            let val_reg = tos_reg(e.dv(), 2); // T0 = value
            e.emit_raw(arm64_enc::str_64(val_reg, Reg::FP, 0));
            start = e.finish_br_if_simple();
        }
        buf.finish_write(0, buf.len());

        let jit_handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit_handler), // [0] JIT
            Instruction::new_handler_only(term),         // [1] fallthrough
            Instruction::new_handler_only(term),         // [2] for nh
            Instruction::new_handler_only(term),         // [3] target (not reached)
            Instruction::new_handler_only(term),         // [4] for target nh
        ];
        insts[0].imm0 = &insts[3] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                42, 0, 0, 0, // t0=42(value), t1=0(cond=0 → fallthrough)
                nh,
            );
        }
        assert_eq!(frame[0], 42, "value should be stored to fp[0]");
    }

    #[test]
    fn test_br_if_simple_taken() {
        // Condition = 1 → branch taken (nonlinear dispatch to imm0 target).
        // height=2, D=2: pos1(top)=T1(cond=1), pos2=T0(value=42)
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start;
        {
            let mut e = JitEmitter::new(&mut buf, 2);
            let val_reg = tos_reg(e.dv(), 2);
            e.emit_raw(arm64_enc::str_64(val_reg, Reg::FP, 0));
            start = e.finish_br_if_simple();
        }
        buf.finish_write(0, buf.len());

        let jit_handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit_handler), // [0] JIT
            Instruction::new_handler_only(term),         // [1] fallthrough (not reached)
            Instruction::new_handler_only(term),         // [2]
            Instruction::new_handler_only(term),         // [3] target
            Instruction::new_handler_only(term),         // [4] for target nh
        ];
        insts[0].imm0 = &insts[3] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                42, 1, 0, 0, // t0=42(value), t1=1(cond=1 → branch taken)
                nh,
            );
        }
        assert_eq!(frame[0], 42);
        // Returned without crashing: nonlinear dispatch to target worked.
    }

    #[test]
    fn test_br_if_simple_taken_multi_group() {
        // Group 1: const 42, const 1(cond), store value, br_if → branches to inst[3]
        // Group 2 (at inst[3]): const 99, store to fp[1], linear dispatch → term
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start1;
        {
            let mut e = JitEmitter::new(&mut buf, 0);
            e.i32_const(42);  // height=1: value
            e.i32_const(1);   // height=2: condition=1 → branch taken
            let val_reg = tos_reg(e.dv(), 2); // value at pos2
            e.emit_raw(arm64_enc::str_64(val_reg, Reg::FP, 0));
            start1 = e.finish_br_if_simple();
        }

        let start2 = buf.len();
        {
            let mut e = JitEmitter::new(&mut buf, 1); // height=1 after br_if pop
            e.i32_const(99);
            let top = tos_reg(e.dv(), 1);
            e.emit_raw(arm64_enc::str_64(top, Reg::FP, 1)); // fp[1] = 99
            e.finish();
        }

        buf.finish_write(0, buf.len());

        let jit1: OpHandler = unsafe { buf.fn_ptr(start1) };
        let jit2: OpHandler = unsafe { buf.fn_ptr(start2) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit1),  // [0] group1 (br_if)
            Instruction::new_handler_only(term),  // [1] fallthrough (not reached)
            Instruction::new_handler_only(term),  // [2]
            Instruction::new_handler_only(jit2),  // [3] target → group2
            Instruction::new_handler_only(term),  // [4] group2 fallthrough
            Instruction::new_handler_only(term),  // [5] for nh
        ];
        insts[0].imm0 = &insts[3] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0, 0, 0, 0, 0, nh,
            );
        }

        assert_eq!(frame[0], 42, "group1 stored value");
        assert_eq!(frame[1], 99, "group2 at target stored value");
    }

    #[test]
    fn test_br_if_simple_fallthrough_multi_group() {
        // Group 1: const 42, const 0(cond=0), store value, br_if → falls through
        // Group 2 (at inst[1]): const 88, store to fp[1], linear dispatch → term
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start1;
        {
            let mut e = JitEmitter::new(&mut buf, 0);
            e.i32_const(42);
            e.i32_const(0);  // condition=0 → fallthrough
            let val_reg = tos_reg(e.dv(), 2);
            e.emit_raw(arm64_enc::str_64(val_reg, Reg::FP, 0));
            start1 = e.finish_br_if_simple();
        }

        let start2 = buf.len();
        {
            let mut e = JitEmitter::new(&mut buf, 1);
            e.i32_const(88);
            let top = tos_reg(e.dv(), 1);
            e.emit_raw(arm64_enc::str_64(top, Reg::FP, 1));
            e.finish();
        }

        buf.finish_write(0, buf.len());

        let jit1: OpHandler = unsafe { buf.fn_ptr(start1) };
        let jit2: OpHandler = unsafe { buf.fn_ptr(start2) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit1),  // [0] group1 (br_if)
            Instruction::new_handler_only(jit2),  // [1] fallthrough → group2
            Instruction::new_handler_only(term),  // [2] group2 fallthrough
            Instruction::new_handler_only(term),  // [3] for nh
            Instruction::new_handler_only(term),  // [4] target (not reached)
            Instruction::new_handler_only(term),  // [5]
        ];
        insts[0].imm0 = &insts[4] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0, 0, 0, 0, 0, nh,
            );
        }

        assert_eq!(frame[0], 42, "group1 stored value");
        assert_eq!(frame[1], 88, "group2 at fallthrough stored value");
    }

    #[test]
    fn test_br_taken_multi_group() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start1;
        {
            let mut e = JitEmitter::new(&mut buf, 0);
            e.i32_const(42);
            let top = tos_reg(e.dv(), 1);
            e.emit_raw(arm64_enc::str_64(top, Reg::FP, 0));
            start1 = e.finish_br();
        }

        let start2 = buf.len();
        {
            let mut e = JitEmitter::new(&mut buf, 0);
            e.i32_const(99);
            let top = tos_reg(e.dv(), 1);
            e.emit_raw(arm64_enc::str_64(top, Reg::FP, 1));
            e.finish();
        }

        buf.finish_write(0, buf.len());

        let jit1: OpHandler = unsafe { buf.fn_ptr(start1) };
        let jit2: OpHandler = unsafe { buf.fn_ptr(start2) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit1),  // [0] group1 (br)
            Instruction::new_handler_only(term),  // [1] fallthrough (not reached)
            Instruction::new_handler_only(term),  // [2]
            Instruction::new_handler_only(jit2),  // [3] branch target
            Instruction::new_handler_only(term),  // [4] group2 fallthrough
            Instruction::new_handler_only(term),  // [5] for nh
        ];
        insts[0].imm0 = &insts[3] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0, 0, 0, 0, 0, nh,
            );
        }

        assert_eq!(frame[0], 42, "group1 stored value");
        assert_eq!(frame[1], 99, "group2 at branch target stored value");
    }

    #[test]
    fn test_if_taken_jumps_to_target_on_zero() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start;
        {
            let mut e = JitEmitter::new(&mut buf, 2);
            let val_reg = tos_reg(e.dv(), 2);
            e.emit_raw(arm64_enc::str_64(val_reg, Reg::FP, 0));
            start = e.finish_if();
        }
        buf.finish_write(0, buf.len());

        let jit_handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit_handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        insts[0].imm0 = &insts[3] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                42, 0, 0, 0, // t0=value, t1=cond=0 -> branch to alt target
                nh,
            );
        }

        assert_eq!(frame[0], 42);
    }

    #[test]
    fn test_if_fallthrough_on_nonzero() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();

        let start;
        {
            let mut e = JitEmitter::new(&mut buf, 2);
            let val_reg = tos_reg(e.dv(), 2);
            e.emit_raw(arm64_enc::str_64(val_reg, Reg::FP, 0));
            start = e.finish_if();
        }
        buf.finish_write(0, buf.len());

        let jit_handler: OpHandler = unsafe { buf.fn_ptr(start) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit_handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];
        insts[0].imm0 = &insts[3] as *const Instruction as u64;

        let mut frame = [0u64; 32];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            frame.as_mut_ptr().wrapping_add(32),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, frame.as_mut_ptr(),
                0, 0, 0,
                42, 1, 0, 0, // t0=value, t1=cond=1 -> fallthrough
                nh,
            );
        }

        assert_eq!(frame[0], 42);
    }

    // ==================== Step 10: Memory load/store tests ====================

    #[test]
    fn test_i32_load_basic() {
        let mut mem = [0u8; 1024];
        mem[0..4].copy_from_slice(&42u32.to_le_bytes());

        // addr=0 in TOS, load 4 bytes → should get 42
        let result = run_jit_mem_test(
            |e| e.i32_load(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, 42);
    }

    #[test]
    fn test_i32_load_with_offset() {
        let mut mem = [0u8; 1024];
        mem[8..12].copy_from_slice(&99u32.to_le_bytes());

        // addr=0, offset=8 → loads from byte 8
        let result = run_jit_mem_test(
            |e| e.i32_load(8),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, 99);
    }

    #[test]
    fn test_i32_load_large_offset() {
        let mut mem = vec![0u8; 8192];
        let off: u32 = 5000;
        mem[off as usize..off as usize + 4].copy_from_slice(&77u32.to_le_bytes());

        // addr=0, offset=5000 (> 4095, uses materialize path)
        let result = run_jit_mem_test(
            |e| e.i32_load(off),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, 77);
    }

    #[test]
    fn test_i64_load_basic() {
        let mut mem = [0u8; 1024];
        let val: u64 = 0xDEAD_BEEF_CAFE_BABE;
        mem[0..8].copy_from_slice(&val.to_le_bytes());

        let result = run_jit_mem_test(
            |e| e.i64_load(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, val);
    }

    #[test]
    fn test_i32_load8_u() {
        let mut mem = [0u8; 1024];
        mem[0] = 0xFF;

        let result = run_jit_mem_test(
            |e| e.i32_load8_u(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, 0xFF);
    }

    #[test]
    fn test_i32_load8_s() {
        let mut mem = [0u8; 1024];
        mem[0] = 0x80; // -128 as signed byte

        let result = run_jit_mem_test(
            |e| e.i32_load8_s(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        // Sign-extended: 0x80 → 0xFFFFFF80 as u32 = 4294967168
        assert_eq!(result as u32, (-128i32) as u32);
    }

    #[test]
    fn test_i32_load16_s() {
        let mut mem = [0u8; 1024];
        mem[0..2].copy_from_slice(&0x8000u16.to_le_bytes()); // -32768

        let result = run_jit_mem_test(
            |e| e.i32_load16_s(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result as u32, (-32768i32) as u32);
    }

    #[test]
    fn test_i64_load8_s() {
        let mut mem = [0u8; 1024];
        mem[0] = 0xFE; // -2 as signed byte

        let result = run_jit_mem_test(
            |e| e.i64_load8_s(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, (-2i64) as u64);
    }

    #[test]
    fn test_i64_load32_s() {
        let mut mem = [0u8; 1024];
        mem[0..4].copy_from_slice(&0x80000000u32.to_le_bytes()); // -2147483648

        let result = run_jit_mem_test(
            |e| e.i64_load32_s(0),
            1, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, (-2147483648i64) as u64);
    }

    #[test]
    fn test_i32_store_basic() {
        let mut mem = [0u8; 1024];

        // D2: pos2=T0(addr=0), pos1=T1(val=42)
        let trap = run_jit_mem_store_test(
            |e| e.i32_store(0),
            2, 0, 42, 0, 0, &mut mem,
        );
        assert!(trap.is_null(), "should not trap");
        let stored = u32::from_le_bytes([mem[0], mem[1], mem[2], mem[3]]);
        assert_eq!(stored, 42);
    }

    #[test]
    fn test_i64_store_basic() {
        let mut mem = [0u8; 1024];
        let val: u64 = 0xCAFE_BABE_DEAD_BEEF;

        // D2: pos2=T0(addr=0), pos1=T1(val)
        let trap = run_jit_mem_store_test(
            |e| e.i64_store(0),
            2, 0, val, 0, 0, &mut mem,
        );
        assert!(trap.is_null(), "should not trap");
        let stored = u64::from_le_bytes(mem[0..8].try_into().unwrap());
        assert_eq!(stored, val);
    }

    #[test]
    fn test_i32_store8() {
        let mut mem = [0u8; 1024];
        // Store 0xABCD to byte — should truncate to 0xCD
        let trap = run_jit_mem_store_test(
            |e| e.i32_store8(0),
            2, 0, 0xABCD, 0, 0, &mut mem,
        );
        assert!(trap.is_null());
        assert_eq!(mem[0], 0xCD);
    }

    #[test]
    fn test_mem_oob_trap() {
        let mut mem = [0u8; 16];

        // Try to load from addr=13 with i32_load (4 bytes) → ea_end=17 > 16
        let trap = run_jit_mem_store_test(
            |e| e.i32_load(0),
            1, 13, 0, 0, 0, &mut mem,
        );
        assert!(!trap.is_null(), "should trap on OOB access");
    }

    #[test]
    fn test_mem_const_store_load() {
        // Group: const 0 (addr), const 999, i32_store, const 0 (addr), i32_load → 999
        let mut mem = [0u8; 1024];

        let result = run_jit_mem_test(
            |e| {
                e.i32_const(0);     // addr
                e.i32_const(999);   // val
                e.i32_store(0);     // store 999 at mem[0]
                e.i32_const(0);     // addr
                e.i32_load(0);      // load from mem[0]
            },
            0, 0, 0, 0, 0, 0, 0, 0, &mut mem,
        );
        assert_eq!(result, 999);
    }
}
