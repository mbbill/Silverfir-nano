//! Target-neutral native frame/layout helpers.
//!
//! This module is shared by runtime and backend consumers. It must stay above
//! `arch/` so generic native code does not depend on a specific consumer.

use crate::vm::native::ir::NativeProgram;

/// Returns the total frame slots required by the finalized native program.
#[inline]
pub fn frame_slots_used(program: &NativeProgram) -> usize {
    program.frame.operands.end().0 as usize + program.spill_slots as usize
}
