//! Shared compile-time IR metadata used by both semantic and lowered stages.

/// Index into the pre-compaction op stream used across compile lowering,
/// backend resolution, and finalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpIndex(usize);

impl OpIndex {
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    #[inline]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl From<usize> for OpIndex {
    #[inline]
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl From<OpIndex> for usize {
    #[inline]
    fn from(index: OpIndex) -> Self {
        index.as_usize()
    }
}

/// Entry for `br_table`: target info for each label.
#[derive(Debug, Clone)]
pub struct BrTableEntry {
    pub target_idx: Option<OpIndex>,
    pub stack_offset: usize,
    pub arity: usize,
}
