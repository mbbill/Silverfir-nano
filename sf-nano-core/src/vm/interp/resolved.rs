//! Fast-interpreter resolved instruction stream.

use crate::vm::interp::instruction::Instruction;

/// Fast-interpreter resolved instruction.
#[derive(Clone, Debug)]
pub struct ResolvedFastInst {
    pub inst: Instruction,
}
