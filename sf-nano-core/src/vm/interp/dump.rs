//! Interpreter-specific debug appendix to the shared SSA-IR dump.

use crate::vm::interp::resolved::ResolvedInterpInst;

pub fn dump_interpreter_resolution(_resolved: &[ResolvedInterpInst]) {
    // TODO: Append interpreter-specific resolved information next to the
    // shared SSA-IR dump without owning the shared dump pipeline.
}
