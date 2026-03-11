//! Interpreter resolved instruction stream.

use crate::vm::interp::instruction::Instruction;

/// Interpreter resolved instruction.
#[derive(Clone, Debug)]
pub struct ResolvedInterpInst {
    pub inst: Instruction,
}
