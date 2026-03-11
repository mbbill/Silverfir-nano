//! Fast-specific debug appendix to the shared LIR dump.

use crate::vm::interp::resolved::ResolvedFastInst;

pub fn dump_fast_resolution(_resolved: &[ResolvedFastInst]) {
    // TODO: Append fast/base/fusion-specific resolved information next to the
    // shared LIR dump without owning the shared dump pipeline.
}
