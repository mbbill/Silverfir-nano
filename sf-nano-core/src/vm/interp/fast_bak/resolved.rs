//! Fast-interpreter resolved instruction stream.

use crate::vm::interp::fast::instruction::Instruction;

/// Fast-interpreter resolved instruction.
#[derive(Clone, Debug)]
pub struct ResolvedFastInst {
    pub inst: Instruction,
}
