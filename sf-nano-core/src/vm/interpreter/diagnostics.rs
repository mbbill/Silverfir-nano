use alloc::{format, string::String, vec::Vec};

use super::InterpInstance;

/// An owned interpreter diagnostic snapshot, collected only when requested.
/// Operation labels describe internal handlers and may change between releases;
/// they are display text, not stable instruction identifiers.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InterpreterStats {
    /// Executed handler count, or None when `interp-count` is disabled.
    pub dispatches: Option<u64>,
    pub engine_code_bytes: usize,
    /// Static fallthrough pairs, ordered by descending count.
    pub bigrams: Vec<((String, String), u64)>,
    /// Executed slow-path handlers, ordered by descending count.
    pub slow_exits: Vec<(String, u64)>,
}

impl InterpreterStats {
    pub(super) fn collect(instance: &InterpInstance) -> Self {
        Self {
            dispatches: instance
                .dispatch_counting_enabled()
                .then(|| instance.dispatch_count()),
            engine_code_bytes: instance.engine_code_len(),
            bigrams: instance
                .bigram_stats()
                .into_iter()
                .map(|((first, second), count)| {
                    ((format!("{first:?}"), format!("{second:?}")), count)
                })
                .collect(),
            slow_exits: instance
                .slow_exit_stats()
                .into_iter()
                .map(|(op, count)| (format!("{op:?}"), count))
                .collect(),
        }
    }
}

impl InterpInstance {
    pub(crate) fn statistics(&self) -> InterpreterStats {
        InterpreterStats::collect(self)
    }
}
