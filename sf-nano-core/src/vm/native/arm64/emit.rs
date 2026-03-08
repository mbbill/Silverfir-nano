//! JIT emission helpers.
//!
//! Provides the dispatch stub (universal JIT group exit sequence)
//! and hot-context field offset constants.

use alloc::vec::Vec;

use super::super::CodeBuffer;
use super::arm64_enc;
use super::reg::Reg;
use crate::vm::native::bridge::{self, HelperMetadata};
use crate::vm::native::instruction::{NativeEntry, NativeInst};
pub use crate::vm::native::context::ctx_offset;

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn native_helper_reload_tos_from_x1();
}

/// Emit a direct branch to an entry pointer loaded from an inline literal.
///
/// Layout:
/// ```text
/// ldr  target_reg, .+8
/// br   target_reg
/// .quad 0      ; patched later with NativeEntry
/// ```
///
/// Returns the byte offset of the 8-byte literal payload that should be
/// patched with the entry pointer.
pub fn emit_direct_branch_literal(buf: &mut CodeBuffer, target_reg: Reg) -> usize {
    buf.emit(arm64_enc::ldr_lit_64(target_reg, 2));
    buf.emit(arm64_enc::br(target_reg));
    buf.emit_u64(0)
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

/// Emit a cold wrapper thunk that installs its metadata pointer in `x20`
/// and tail-branches to a shared helper entry.
pub fn emit_helper_wrapper(
    buf: &mut CodeBuffer,
    helper_entry: NativeEntry,
    meta: *const HelperMetadata,
) -> (usize, usize) {
    let start = buf.len();
    // Keep wrapper thunks tiny: load the metadata pointer and shared helper
    // entry from inline literals instead of materializing full 64-bit values.
    buf.emit(arm64_enc::ldr_lit_64(Reg::PC, 4));
    buf.emit(arm64_enc::ldr_lit_64(Reg::TMP1, 5));
    buf.emit(arm64_enc::br(Reg::TMP1));
    buf.emit(arm64_enc::nop());
    buf.emit_u64(meta as usize as u64);
    buf.emit_u64(helper_entry as usize as u64);
    (start, buf.len() - start)
}

#[derive(Clone, Copy, Debug)]
pub struct DirectBrTableRecord {
    pub packed_branch: u64,
}

#[derive(Debug)]
pub struct DirectBrTableEmit {
    pub start: usize,
    pub len: usize,
    pub target_entry_literals: Vec<usize>,
}

#[derive(Debug)]
pub struct DirectCallInternalEmit {
    pub start: usize,
    pub len: usize,
    pub callee_entry_literal: usize,
    pub next_entry_literal: usize,
}

/// Emit a direct br_table entry specialized for one site.
///
/// This restores the old hot-path shape: the entry executes the whole
/// br_table dispatch directly instead of going through a wrapper thunk and a
/// shared helper entry.
pub fn emit_direct_br_table_entry(
    buf: &mut CodeBuffer,
    src_reg: Reg,
    records: &[DirectBrTableRecord],
    operand_base_offset: u32,
    height: u16,
) -> DirectBrTableEmit {
    let start = buf.len();
    let table_ptr_load = buf.emit(arm64_enc::ldr_lit_64(Reg::TMP2, 0));
    materialize_u64(buf, Reg::TMP1, records.len().saturating_sub(1) as u64);

    // selected = min(idx, entry_count - 1)
    buf.emit(arm64_enc::mov_reg_64(Reg::TMP0, src_reg));
    buf.emit(arm64_enc::cmp_reg_64(Reg::TMP0, Reg::TMP1));
    buf.emit(arm64_enc::csel_64(Reg::TMP0, Reg::TMP0, Reg::TMP1, arm64_enc::Cond::LO));

    // record_ptr = table_base + selected * 16
    buf.emit(arm64_enc::lsl_imm_64(Reg::TMP3, Reg::TMP0, 4));
    buf.emit(arm64_enc::add_reg_64(Reg::TMP3, Reg::TMP2, Reg::TMP3));
    buf.emit(arm64_enc::mov_reg_64(Reg::TMP4, Reg::TMP3));

    buf.emit(arm64_enc::ldr_64(Reg::TMP5, Reg::TMP3, 1));

    // stack_drop in TMP0, arity in TMP5.
    buf.emit(arm64_enc::movz_64(Reg::A1, 0xffff, 0));
    buf.emit(arm64_enc::mov_reg_64(Reg::A0, Reg::TMP5));
    buf.emit(arm64_enc::lsr_imm_64(Reg::TMP0, Reg::TMP5, 16));
    buf.emit(arm64_enc::and_reg_64(Reg::TMP0, Reg::TMP0, Reg::A1));
    buf.emit(arm64_enc::and_reg_64(Reg::TMP5, Reg::A0, Reg::A1));

    // Packed operand_base_offset / height-1 in A1 / A0.
    materialize_u64(
        buf,
        Reg::TMP2,
        ((operand_base_offset as u64) << 32) | (height.saturating_sub(1) as u64),
    );
    buf.emit(arm64_enc::lsr_imm_64(Reg::TMP1, Reg::TMP2, 32));
    buf.emit(arm64_enc::mov_reg_32(Reg::A0, Reg::TMP2));

    let skip_fixup_stack_drop = buf.emit(arm64_enc::cbz_64(Reg::TMP0, 0));
    let skip_fixup_arity = buf.emit(arm64_enc::cbz_64(Reg::TMP5, 0));

    // operand_base = fp + operand_base_offset
    buf.emit(arm64_enc::add_reg_64(Reg::TMP1, Reg::FP, Reg::TMP1));
    // src_index = (height - 1) - arity
    buf.emit(arm64_enc::sub_reg_64(Reg::TMP3, Reg::A0, Reg::TMP5));
    // dst_index = src_index - stack_drop
    buf.emit(arm64_enc::sub_reg_64(Reg::A0, Reg::TMP3, Reg::TMP0));
    buf.emit(arm64_enc::lsl_imm_64(Reg::TMP3, Reg::TMP3, 3));
    buf.emit(arm64_enc::lsl_imm_64(Reg::A0, Reg::A0, 3));
    buf.emit(arm64_enc::add_reg_64(Reg::TMP3, Reg::TMP1, Reg::TMP3));
    buf.emit(arm64_enc::add_reg_64(Reg::A0, Reg::TMP1, Reg::A0));
    let fixup_loop = buf.len();
    buf.emit(arm64_enc::ldr_64(Reg::A1, Reg::TMP3, 0));
    buf.emit(arm64_enc::str_64(Reg::A1, Reg::A0, 0));
    buf.emit(arm64_enc::add_imm_64(Reg::TMP3, Reg::TMP3, 8));
    buf.emit(arm64_enc::add_imm_64(Reg::A0, Reg::A0, 8));
    buf.emit(arm64_enc::subs_imm_64(Reg::TMP5, Reg::TMP5, 1));
    buf.emit(arm64_enc::b_cond(
        arm64_enc::Cond::NE,
        ((fixup_loop as i32) - (buf.len() as i32)) / 4,
    ));
    let after_fixup = buf.len();
    buf.patch_u32(
        skip_fixup_stack_drop,
        arm64_enc::cbz_64(Reg::TMP0, ((after_fixup - skip_fixup_stack_drop) as i32) / 4),
    );
    buf.patch_u32(
        skip_fixup_arity,
        arm64_enc::cbz_64(Reg::TMP5, ((after_fixup - skip_fixup_arity) as i32) / 4),
    );

    buf.emit(arm64_enc::ldr_64(Reg::A0, Reg::TMP4, 0));
    buf.emit(arm64_enc::br(Reg::A0));

    let table_ptr_literal = buf.len();
    buf.emit_u64(0);
    let table_start = buf.len();
    let mut target_entry_literals = Vec::with_capacity(records.len());
    for record in records {
        target_entry_literals.push(buf.len());
        buf.emit_u64(0);
        buf.emit_u64(record.packed_branch);
    }

    let table_ptr = unsafe { buf.base_ptr().add(table_start) as usize as u64 };
    buf.patch_u64(table_ptr_literal, table_ptr);
    let table_ptr_delta = ((table_ptr_literal - table_ptr_load) as i32) / 4;
    buf.patch_u32(table_ptr_load, arm64_enc::ldr_lit_64(Reg::TMP2, table_ptr_delta));

    DirectBrTableEmit {
        start,
        len: buf.len() - start,
        target_entry_literals,
    }
}

pub fn emit_direct_call_internal_entry(
    buf: &mut CodeBuffer,
    params_count: u16,
    locals_count: u16,
    delta_slot: u16,
) -> DirectCallInternalEmit {
    const STACK_OVERFLOW_MSG: &[u8] = b"stack overflow\0";
    const CALL_STACK_EXHAUSTED_MSG: &[u8] = b"call stack exhausted\0";

    let start = buf.len();

    buf.emit(arm64_enc::str_64(Reg::L0, Reg::FP, 0));
    buf.emit(arm64_enc::str_64(Reg::L1, Reg::FP, 1));
    buf.emit(arm64_enc::str_64(Reg::L2, Reg::FP, 2));

    materialize_u64(buf, Reg::TMP0, delta_slot as u64);
    buf.emit(arm64_enc::lsl_imm_64(Reg::TMP0, Reg::TMP0, 3));
    buf.emit(arm64_enc::add_reg_64(Reg::TMP3, Reg::FP, Reg::TMP0));

    let frame_size = params_count as u64 + locals_count as u64;
    materialize_u64(buf, Reg::TMP4, frame_size);
    buf.emit(arm64_enc::lsl_imm_64(Reg::TMP5, Reg::TMP4, 3));
    buf.emit(arm64_enc::add_reg_64(Reg::TMP5, Reg::TMP3, Reg::TMP5));
    buf.emit(arm64_enc::add_imm_64(Reg::TMP5, Reg::TMP5, 24));
    buf.emit(arm64_enc::ldr_64(Reg::TMP1, Reg::CTX, ctx_offset::STACK_END / 8));
    buf.emit(arm64_enc::cmp_reg_64(Reg::TMP5, Reg::TMP1));
    let stack_overflow_patch = buf.emit(arm64_enc::b_cond(arm64_enc::Cond::HI, 0));

    buf.emit(arm64_enc::ldr_64(Reg::TMP5, Reg::CTX, ctx_offset::CALL_DEPTH / 8));
    materialize_u64(buf, Reg::TMP1, 300);
    buf.emit(arm64_enc::cmp_reg_64(Reg::TMP5, Reg::TMP1));
    let call_depth_patch = buf.emit(arm64_enc::b_cond(arm64_enc::Cond::HS, 0));
    buf.emit(arm64_enc::add_imm_64(Reg::TMP5, Reg::TMP5, 1));
    buf.emit(arm64_enc::str_64(Reg::TMP5, Reg::CTX, ctx_offset::CALL_DEPTH / 8));

    if locals_count != 0 {
        materialize_u64(buf, Reg::TMP0, params_count as u64);
        buf.emit(arm64_enc::lsl_imm_64(Reg::TMP0, Reg::TMP0, 3));
        buf.emit(arm64_enc::add_reg_64(Reg::TMP5, Reg::TMP3, Reg::TMP0));
        let mut remaining = locals_count;
        while remaining >= 2 {
            buf.emit(arm64_enc::str_64(Reg::XZR, Reg::TMP5, 0));
            buf.emit(arm64_enc::str_64(Reg::XZR, Reg::TMP5, 1));
            buf.emit(arm64_enc::add_imm_64(Reg::TMP5, Reg::TMP5, 16));
            remaining -= 2;
        }
        if remaining != 0 {
            buf.emit(arm64_enc::str_64(Reg::XZR, Reg::TMP5, 0));
        }
    }

    buf.emit(arm64_enc::lsl_imm_64(Reg::TMP4, Reg::TMP4, 3));
    buf.emit(arm64_enc::add_reg_64(Reg::TMP5, Reg::TMP3, Reg::TMP4));
    let next_entry_load = buf.emit(arm64_enc::ldr_lit_64(Reg::TMP1, 0));
    buf.emit(arm64_enc::str_64(Reg::TMP1, Reg::TMP5, 0));
    buf.emit(arm64_enc::str_64(Reg::FP, Reg::TMP5, 1));
    buf.emit(arm64_enc::str_64(Reg::XZR, Reg::TMP5, 2));
    buf.emit(arm64_enc::mov_reg_64(Reg::FP, Reg::TMP3));
    buf.emit(arm64_enc::mov_reg_64(Reg::T0, Reg::XZR));
    buf.emit(arm64_enc::mov_reg_64(Reg::T1, Reg::XZR));
    buf.emit(arm64_enc::mov_reg_64(Reg::T2, Reg::XZR));
    buf.emit(arm64_enc::mov_reg_64(Reg::T3, Reg::XZR));
    let callee_entry_load = buf.emit(arm64_enc::ldr_lit_64(Reg::TMP1, 0));
    buf.emit(arm64_enc::br(Reg::TMP1));

    let stack_overflow_label = buf.len();
    materialize_u64(buf, Reg::TMP0, STACK_OVERFLOW_MSG.as_ptr() as u64);
    buf.emit(arm64_enc::str_64(Reg::TMP0, Reg::CTX, ctx_offset::TRAP_MESSAGE / 8));
    buf.emit(arm64_enc::ldr_64(Reg::TMP0, Reg::CTX, ctx_offset::TERM_ENTRY / 8));
    buf.emit(arm64_enc::br(Reg::TMP0));

    let call_stack_label = buf.len();
    materialize_u64(buf, Reg::TMP0, CALL_STACK_EXHAUSTED_MSG.as_ptr() as u64);
    buf.emit(arm64_enc::str_64(Reg::TMP0, Reg::CTX, ctx_offset::TRAP_MESSAGE / 8));
    buf.emit(arm64_enc::ldr_64(Reg::TMP0, Reg::CTX, ctx_offset::TERM_ENTRY / 8));
    buf.emit(arm64_enc::br(Reg::TMP0));

    let next_entry_literal = buf.len();
    buf.emit_u64(0);
    let callee_entry_literal = buf.len();
    buf.emit_u64(0);

    let stack_overflow_delta = ((stack_overflow_label - stack_overflow_patch) as i32) / 4;
    buf.patch_u32(
        stack_overflow_patch,
        arm64_enc::b_cond(arm64_enc::Cond::HI, stack_overflow_delta),
    );
    let call_depth_delta = ((call_stack_label - call_depth_patch) as i32) / 4;
    buf.patch_u32(
        call_depth_patch,
        arm64_enc::b_cond(arm64_enc::Cond::HS, call_depth_delta),
    );

    let next_entry_delta = ((next_entry_literal - next_entry_load) as i32) / 4;
    buf.patch_u32(next_entry_load, arm64_enc::ldr_lit_64(Reg::TMP1, next_entry_delta));
    let callee_entry_delta = ((callee_entry_literal - callee_entry_load) as i32) / 4;
    buf.patch_u32(
        callee_entry_load,
        arm64_enc::ldr_lit_64(Reg::TMP1, callee_entry_delta),
    );

    DirectCallInternalEmit {
        start,
        len: buf.len() - start,
        callee_entry_literal,
        next_entry_literal,
    }
}

pub fn emit_resume_entry(
    buf: &mut CodeBuffer,
    target_entry: NativeEntry,
    target_tos_slots: u64,
) -> (usize, usize) {
    let start = buf.len();
    materialize_u64(buf, Reg::A0, target_entry as usize as u64);
    if target_tos_slots == crate::vm::native::instruction::EMPTY_TOS_SLOTS {
        buf.emit(arm64_enc::mov_reg_64(Reg::T0, Reg::XZR));
        buf.emit(arm64_enc::mov_reg_64(Reg::T1, Reg::XZR));
        buf.emit(arm64_enc::mov_reg_64(Reg::T2, Reg::XZR));
        buf.emit(arm64_enc::mov_reg_64(Reg::T3, Reg::XZR));
    } else {
        materialize_u64(buf, Reg::A1, target_tos_slots);
        materialize_u64(
            buf,
            Reg::TMP1,
            native_helper_reload_tos_from_x1 as usize as u64,
        );
        buf.emit(arm64_enc::blr(Reg::TMP1));
    }
    buf.emit(arm64_enc::br(Reg::A0));
    (start, buf.len() - start)
}

/// Emit a direct branch wrapper that skips the generic cold-helper entry path.
///
/// The wrapper materializes its metadata pointer in `x20`, loads the patched
/// branch target, optionally reloads TOS slots from `branch_tos_slots`, and
/// branches directly to the native target entry.
pub fn emit_meta_branch_wrapper(
    buf: &mut CodeBuffer,
    meta: *const HelperMetadata,
    reload_tos: bool,
) -> (usize, usize) {
    let start = buf.len();
    materialize_u64(buf, Reg::PC, meta as usize as u64);
    buf.emit(arm64_enc::ldr_64(
        Reg::A0,
        Reg::PC,
        bridge::meta_offset::BRANCH_ENTRY / 8,
    ));
    if reload_tos {
        buf.emit(arm64_enc::ldr_64(
            Reg::A1,
            Reg::PC,
            bridge::meta_offset::BRANCH_TOS_SLOTS / 8,
        ));
        materialize_u64(
            buf,
            Reg::TMP1,
            native_helper_reload_tos_from_x1 as usize as u64,
        );
        buf.emit(arm64_enc::blr(Reg::TMP1));
    } else {
        buf.emit(arm64_enc::mov_reg_64(Reg::T0, Reg::XZR));
        buf.emit(arm64_enc::mov_reg_64(Reg::T1, Reg::XZR));
        buf.emit(arm64_enc::mov_reg_64(Reg::T2, Reg::XZR));
        buf.emit(arm64_enc::mov_reg_64(Reg::T3, Reg::XZR));
    }
    buf.emit(arm64_enc::br(Reg::A0));
    (start, buf.len() - start)
}

/// Packed TOS slot state offset within legacy native instruction records.
///
/// This remains temporarily while the cold bridge and return paths are being
/// migrated off `NativeInst` descriptors.
pub const INST_TOS_SLOTS_OFFSET: u32 = 32;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "legacy-fast-native-tests"))]
mod offset_checks {
    use super::ctx_offset;
    use crate::vm::interp::fast::context::ContextHot;
    use crate::vm::interp::fast::instruction::Instruction;
    use super::{INST_SIZE, INST_HANDLER_OFFSET, INST_IMM0_OFFSET};

    #[test]
    fn verify_context_offsets() {
        assert_eq!(core::mem::offset_of!(ContextHot, stack_end), ctx_offset::STACK_END as usize);
        assert_eq!(core::mem::offset_of!(ContextHot, call_depth), ctx_offset::CALL_DEPTH as usize);
        assert_eq!(core::mem::offset_of!(ContextHot, mem0_base), ctx_offset::MEM0_BASE as usize);
        assert_eq!(core::mem::offset_of!(ContextHot, mem0_size), ctx_offset::MEM0_SIZE as usize);
        assert_eq!(core::mem::offset_of!(ContextHot, trap_message), ctx_offset::TRAP_MESSAGE as usize);
        assert_eq!(core::mem::offset_of!(ContextHot, term_inst), ctx_offset::TERM_INST as usize);
    }

    #[test]
    fn verify_instruction_layout() {
        assert_eq!(core::mem::size_of::<Instruction>(), INST_SIZE as usize);
        assert_eq!(core::mem::offset_of!(Instruction, handler), INST_HANDLER_OFFSET as usize);
        assert_eq!(core::mem::offset_of!(Instruction, imm0), INST_IMM0_OFFSET as usize);
    }
}

#[cfg(all(test, feature = "legacy-fast-native-tests"))]
mod tests {
    use super::*;
    use crate::vm::interp::fast::instruction::Instruction;
    use crate::vm::interp::fast::handlers::{self, OpHandler, NextHandler, run_trampoline};
    use crate::vm::interp::fast::context::Context;

    #[test]
    fn test_dispatch_stub_encoding() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let off = emit_dispatch_linear(&mut buf);
        buf.finish_write(off, 16);

        let base = buf.base_ptr() as *const u32;
        let inst0 = unsafe { *base.add(0) };
        let inst1 = unsafe { *base.add(1) };
        let inst2 = unsafe { *base.add(2) };
        let inst3 = unsafe { *base.add(3) };

        // Verify against hand-computed expected values
        assert_eq!(inst0, 0x910082B5, "add x21, x21, #0x20");
        assert_eq!(inst1, 0xF94002A2, "ldr x2, [x21]");
        assert_eq!(inst2, 0xF94012A1, "ldr x1, [x21, #0x20]");
        assert_eq!(inst3, 0xD61F0040, "br x2");
    }

    #[test]
    fn test_dispatch_stub_integration() {
        // Proves JIT code can participate in the interpreter's dispatch chain.
        //
        // inst[0].handler = JIT dispatch stub (advances to inst[1])
        // inst[1].handler = op_term (returns from dispatch chain)
        // inst[2].handler = op_term (for nh preload)

        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let stub_off = emit_dispatch_linear(&mut buf);
        buf.finish_write(stub_off, 16);

        let stub_handler: OpHandler = unsafe { buf.fn_ptr(stub_off) };
        let term_handler = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(stub_handler),
            Instruction::new_handler_only(term_handler),
            Instruction::new_handler_only(term_handler),
        ];

        let mut dummy_stack = [0u64; 16];
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            dummy_stack.as_mut_ptr().wrapping_add(16),
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        // If this returns without crashing, the ABI contract is proven.
        unsafe {
            run_trampoline(
                &mut ctx, pc, dummy_stack.as_mut_ptr(),
                0, 0, 0,
                0, 0, 0, 0,
                nh,
            );
        }
    }

    #[test]
    fn test_dispatch_stub_multi_hop() {
        // Two JIT dispatch stubs in sequence, then op_term.
        // Proves multi-hop JIT dispatch works.

        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let stub1_off = emit_dispatch_linear(&mut buf);
        let stub2_off = emit_dispatch_linear(&mut buf);
        buf.finish_write(stub1_off, buf.len());

        let stub1_handler: OpHandler = unsafe { buf.fn_ptr(stub1_off) };
        let stub2_handler: OpHandler = unsafe { buf.fn_ptr(stub2_off) };
        let term_handler = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(stub1_handler),
            Instruction::new_handler_only(stub2_handler),
            Instruction::new_handler_only(term_handler),
            Instruction::new_handler_only(term_handler),
        ];

        let mut dummy_stack = [0u64; 16];
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            dummy_stack.as_mut_ptr().wrapping_add(16),
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        unsafe {
            run_trampoline(
                &mut ctx, pc, dummy_stack.as_mut_ptr(),
                0x1111, 0x2222, 0x3333,
                0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD,
                nh,
            );
        }
        // Reached here: dispatch chain worked through TWO JIT stubs → op_term.
    }

    #[test]
    fn test_dispatch_nonlinear_encoding() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let off = emit_dispatch_nonlinear(&mut buf);
        buf.finish_write(off, 20);

        let base = buf.base_ptr() as *const u32;
        unsafe {
            // ldr x3, [x21, #8]
            assert_eq!(*base.add(0), arm64_enc::ldr_64(Reg::TMP1, Reg::PC, 1));
            // ldr x2, [x3, #0]
            assert_eq!(*base.add(1), arm64_enc::ldr_64(Reg::TMP0, Reg::TMP1, 0));
            // ldr x1, [x3, #0x20]
            assert_eq!(*base.add(2), arm64_enc::ldr_64(Reg::NH, Reg::TMP1, INST_SIZE / 8));
            // mov x21, x3
            assert_eq!(*base.add(3), arm64_enc::mov_reg_64(Reg::PC, Reg::TMP1));
            // br x2
            assert_eq!(*base.add(4), arm64_enc::br(Reg::TMP0));
        }
    }

    #[test]
    fn test_dispatch_nonlinear_integration() {
        // Proves nonlinear dispatch jumps to the target specified in imm0.
        //
        // inst[0].handler = JIT nonlinear dispatch (reads imm0 → jumps to target)
        // inst[0].imm0 = &inst[2] (target)
        // inst[1] = op_term (fallthrough — should NOT be reached)
        // inst[2] = op_term (target — reached via nonlinear dispatch)
        // inst[3] = op_term (for nh preload at target)

        let mut buf = CodeBuffer::new().expect("mmap failed");
        buf.begin_write();
        let off = emit_dispatch_nonlinear(&mut buf);
        buf.finish_write(off, 20);

        let jit_handler: OpHandler = unsafe { buf.fn_ptr(off) };
        let term = handlers::full_set::op_term;

        let mut insts = [
            Instruction::new_handler_only(jit_handler),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
            Instruction::new_handler_only(term),
        ];

        // Patch imm0 to point to inst[2]
        insts[0].imm0 = &insts[2] as *const Instruction as u64;

        let mut stack = [0u64; 16];
        let mut ctx = Context::new(
            core::ptr::null_mut(), core::ptr::null(),
            stack.as_mut_ptr().wrapping_add(16),
            core::ptr::null_mut(), 0,
        );
        ctx.hot.term_inst = handlers::term();

        let pc = &mut insts[0] as *mut Instruction;
        let nh: NextHandler = unsafe { core::mem::transmute(insts[1].handler) };

        // If this returns, nonlinear dispatch to inst[2] worked correctly.
        unsafe {
            run_trampoline(&mut ctx, pc, stack.as_mut_ptr(),
                0, 0, 0, 0, 0, 0, 0, nh);
        }
    }
}
