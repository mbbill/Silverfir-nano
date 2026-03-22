//! Trace comparison helpers.

use crate::vm::debug::function_trace::FunctionTraceEvent;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TraceMismatch {
    pub left_index: usize,
    pub right_index: usize,
    pub left: Option<FunctionTraceEvent>,
    pub right: Option<FunctionTraceEvent>,
}

pub(crate) fn compare_sparse_traces(
    left: &[FunctionTraceEvent],
    right: &[FunctionTraceEvent],
) -> Result<(), TraceMismatch> {
    let len = core::cmp::max(left.len(), right.len());
    for idx in 0..len {
        let l = left.get(idx);
        let r = right.get(idx);
        if l != r {
            return Err(TraceMismatch {
                left_index: idx,
                right_index: idx,
                left: l.cloned(),
                right: r.cloned(),
            });
        }
    }
    Ok(())
}
