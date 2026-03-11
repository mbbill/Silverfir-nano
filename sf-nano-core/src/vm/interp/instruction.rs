//! Interpreter instruction format.
//!
//! This is interpreter-specific and must not leak into `native/`.

use alloc::{boxed::Box, vec::Vec};

use crate::vm::interp::handlers::OpHandler;

/// Interpreter instruction record.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    pub handler: OpHandler,
    pub imm0: u64,
    pub imm1: u64,
    pub imm2: u64,
}

const _: [(); 32] = [(); core::mem::size_of::<Instruction>()];

impl Instruction {
    #[inline]
    pub fn new(handler: OpHandler, imm0: u64, imm1: u64, imm2: u64) -> Self {
        Self {
            handler,
            imm0,
            imm1,
            imm2,
        }
    }

    #[inline]
    pub fn new_handler_only(handler: OpHandler) -> Self {
        Self::new(handler, 0, 0, 0)
    }

    #[inline]
    pub fn make_terminal(&mut self, op_term: OpHandler) {
        self.handler = op_term;
        self.imm0 = 0;
        self.imm1 = 0;
        self.imm2 = 0;
    }
}

#[derive(Default)]
pub struct InstructionArena {
    insts: Vec<Instruction>,
}

impl InstructionArena {
    pub fn new() -> Self {
        Self { insts: Vec::new() }
    }

    pub fn push(&mut self, inst: Instruction) -> *mut Instruction {
        self.insts.push(inst);
        unsafe { self.insts.as_mut_ptr().add(self.insts.len() - 1) }
    }

    pub fn slice_mut(&mut self) -> &mut [Instruction] {
        &mut self.insts
    }

    pub fn len(&self) -> usize {
        self.insts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    pub fn into_boxed_slice(self) -> Box<[Instruction]> {
        self.insts.into_boxed_slice()
    }
}
