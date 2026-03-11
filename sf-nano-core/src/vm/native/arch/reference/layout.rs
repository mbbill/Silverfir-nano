//! Native frame/layout helpers shared by runtime and the reference backend.

use crate::vm::native::ir::NativeProgram;

/// Returns the total frame slots required by the finalized native program.
#[inline]
pub fn frame_slots_used(program: &NativeProgram) -> usize {
    program.frame.operands.end().0 as usize
}
