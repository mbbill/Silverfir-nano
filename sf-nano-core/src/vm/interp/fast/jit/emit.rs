//! JIT emission helpers.
//!
//! Provides the dispatch stub (universal JIT group exit sequence)
//! and context field offset constants.

use super::code_buf::CodeBuffer;
use super::reg::Reg;
use super::arm64_enc;

// ---------------------------------------------------------------------------
// Context field offsets (must match context.rs layout)
// ---------------------------------------------------------------------------

/// Byte offsets into the Context struct, used by JIT-emitted code.
pub mod ctx_offset {
    pub const STACK_END: u32 = 0;
    pub const CALL_DEPTH: u32 = 8;
    pub const MEM0_BASE: u32 = 16;
    pub const MEM0_SIZE: u32 = 24;
    pub const TRAP_MESSAGE: u32 = 32;
    pub const TERM_INST: u32 = 40;
}

// ---------------------------------------------------------------------------
// Instruction layout constants
// ---------------------------------------------------------------------------

/// Size of one Instruction struct in bytes (handler + 3 immediates = 32 bytes).
pub const INST_SIZE: u32 = 32;

/// Byte offset of handler field within Instruction (always 0).
pub const INST_HANDLER_OFFSET: u32 = 0;

// ---------------------------------------------------------------------------
// Dispatch stub
// ---------------------------------------------------------------------------

/// Emit the universal dispatch stub (JIT group exit sequence).
///
/// This advances PC by one Instruction slot (32 bytes), loads the handler
/// at the new PC, preloads the next-handler (nh), and tail-jumps.
///
/// Emitted instructions:
/// ```text
/// add  x21, x21, #0x20     ; pc += 1 instruction (32 bytes)
/// ldr  x2,  [x21]          ; handler = pc->handler
/// ldr  x1,  [x21, #0x20]   ; nh = (pc+1)->handler
/// br   x2                   ; tail-jump to handler
/// ```
///
/// Returns the byte offset where the stub starts in the buffer.
pub fn emit_dispatch_linear(buf: &mut CodeBuffer) -> usize {
    let start = buf.len();
    buf.emit(arm64_enc::add_imm_64(Reg::PC, Reg::PC, INST_SIZE));
    buf.emit(arm64_enc::ldr_64(Reg::TMP0, Reg::PC, INST_HANDLER_OFFSET / 8));
    buf.emit(arm64_enc::ldr_64(Reg::NH, Reg::PC, INST_SIZE / 8));
    buf.emit(arm64_enc::br(Reg::TMP0));
    start
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod offset_checks {
    use super::ctx_offset;
    use crate::vm::interp::fast::context::Context;
    use crate::vm::interp::fast::instruction::Instruction;
    use super::{INST_SIZE, INST_HANDLER_OFFSET};

    #[test]
    fn verify_context_offsets() {
        assert_eq!(core::mem::offset_of!(Context, stack_end), ctx_offset::STACK_END as usize);
        assert_eq!(core::mem::offset_of!(Context, call_depth), ctx_offset::CALL_DEPTH as usize);
        assert_eq!(core::mem::offset_of!(Context, mem0_base), ctx_offset::MEM0_BASE as usize);
        assert_eq!(core::mem::offset_of!(Context, mem0_size), ctx_offset::MEM0_SIZE as usize);
        assert_eq!(core::mem::offset_of!(Context, trap_message), ctx_offset::TRAP_MESSAGE as usize);
        assert_eq!(core::mem::offset_of!(Context, term_inst), ctx_offset::TERM_INST as usize);
    }

    #[test]
    fn verify_instruction_layout() {
        assert_eq!(core::mem::size_of::<Instruction>(), INST_SIZE as usize);
        assert_eq!(core::mem::offset_of!(Instruction, handler), INST_HANDLER_OFFSET as usize);
    }
}

#[cfg(test)]
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
        ctx.term_inst = handlers::term() as *mut u8;

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
        ctx.term_inst = handlers::term() as *mut u8;

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
}
