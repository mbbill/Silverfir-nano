//! Backend-supplied preparation policy.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupingMode {
    Ignore,
    /// Reserved for future fusion-specific grouping. Today this falls back to
    /// the same maximal grouping used by native.
    PatternDriven,
    /// Group every prepared LIR op that is not separated by an explicit call,
    /// runtime boundary, or control boundary.
    Maximal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanPolicy {
    pub grouping: GroupingMode,
}

impl PlanPolicy {
    #[inline]
    pub const fn new(grouping: GroupingMode) -> Self {
        Self { grouping }
    }
}
