//! JIT codegen: per-opcode emission and group assembly.
//!
//! `JitEmitter` tracks TOS height and emits ARM64 instructions for each
//! supported Wasm opcode. Depth-variant register selection matches the
//! interpreter's TOS register window convention.

use super::code_buf::CodeBuffer;
use super::reg::Reg;
use super::arm64_enc::{self, Cond};
use super::emit;

/// TOS registers in index order: T0=x26, T1=x27, T2=x28, T3=x0.
const TOS_REGS: [Reg; 4] = [Reg::T0, Reg::T1, Reg::T2, Reg::T3];

/// Hot local registers: L0=x23, L1=x24, L2=x25.
const LOCAL_REGS: [Reg; 3] = [Reg::L0, Reg::L1, Reg::L2];

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
    /// Byte offset where this group's code starts.
    pub start_offset: usize,
}

impl<'a> JitEmitter<'a> {
    pub fn new(buf: &'a mut CodeBuffer, initial_height: usize) -> Self {
        let start_offset = buf.len();
        Self { buf, height: initial_height, start_offset }
    }

    pub fn height(&self) -> usize { self.height }

    fn dv(&self) -> u8 { depth_variant(self.height) }

    /// Finish the group: emit dispatch stub. Returns start offset.
    pub fn finish(self) -> usize {
        emit::emit_dispatch_linear(self.buf);
        self.start_offset
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
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::cmp_imm_32(src, 0));
        self.buf.emit(arm64_enc::cset_32(src, Cond::EQ));
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
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::cmp_imm_64(src, 0));
        self.buf.emit(arm64_enc::cset_64(src, Cond::EQ));
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
        self.height += 1;
        let dst = tos_reg(self.dv(), 1);
        materialize_u32(self.buf, dst, value);
    }

    pub fn i64_const(&mut self, value: u64) {
        self.height += 1;
        let dst = tos_reg(self.dv(), 1);
        materialize_u64(self.buf, dst, value);
    }

    // ==================== Hot locals (push1 / pop1) ====================

    pub fn local_get_ln(&mut self, n: u8) {
        self.height += 1;
        let dst = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::mov_reg_64(dst, LOCAL_REGS[n as usize]));
    }

    pub fn local_set_ln(&mut self, n: u8) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::mov_reg_64(LOCAL_REGS[n as usize], src));
        self.height -= 1;
    }

    pub fn local_tee_ln(&mut self, n: u8) {
        let src = tos_reg(self.dv(), 1);
        self.buf.emit(arm64_enc::mov_reg_64(LOCAL_REGS[n as usize], src));
        // height unchanged (tee keeps value on stack)
    }

    // ==================== Non-hot locals (push1 / pop1) ====================

    pub fn local_get(&mut self, idx: u16) {
        self.height += 1;
        let dst = tos_reg(self.dv(), 1);
        let scaled = idx as u32; // ldr_64 imm12 is pre-scaled by 8
        self.buf.emit(arm64_enc::ldr_64(dst, Reg::FP, scaled));
    }

    pub fn local_set(&mut self, idx: u16) {
        let src = tos_reg(self.dv(), 1);
        let scaled = idx as u32;
        self.buf.emit(arm64_enc::str_64(src, Reg::FP, scaled));
        self.height -= 1;
    }

    // ==================== Drop (pop1, no code) ====================

    pub fn drop_val(&mut self) {
        self.height -= 1;
        // No code emitted — value is simply forgotten
    }
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
        let result_reg = tos_reg(emitter.dv(), 1);
        let start = emitter.start_offset;
        buf.emit(arm64_enc::str_64(result_reg, Reg::FP, 0));

        // Dispatch stub
        emit::emit_dispatch_linear(&mut buf);

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
        ctx.term_inst = handlers::term() as *mut u8;

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

        let start = emitter.start_offset;
        // Store the local register to fp[0] for verification
        buf.emit(arm64_enc::str_64(local_reg, Reg::FP, 0));

        emit::emit_dispatch_linear(&mut buf);

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
        ctx.term_inst = handlers::term() as *mut u8;

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
}
