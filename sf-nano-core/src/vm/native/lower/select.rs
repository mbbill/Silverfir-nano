//! Native selection from shared LIR.
//!
//! This phase is where native-owned policy will decide:
//! - inline leaf op vs cold helper
//! - fast `call_local` vs generic call path
//! - direct return shapes
//! - block tail fusion opportunities before placement
//!
//! The live native backend has not been migrated here yet; this is the stable
//! file boundary for that work.

use crate::vm::lir::ir::LirProgram;

#[derive(Clone, Copy, Debug)]
pub(super) struct SelectedProgram<'a> {
    pub lir: &'a LirProgram,
}

#[inline]
pub(super) fn select_program(lir: &LirProgram) -> SelectedProgram<'_> {
    SelectedProgram { lir }
}
