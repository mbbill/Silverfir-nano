//! Shared compile-time IR metadata used by both semantic and lowered stages.

/// Logical reference to a frame slot used by encoded handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRef {
    Absolute(u16),
    OperandRelative(u16),
}

impl SlotRef {
    #[inline]
    pub const fn absolute(slot: u16) -> Self {
        Self::Absolute(slot)
    }

    #[inline]
    pub const fn operand_relative(offset: u16) -> Self {
        Self::OperandRelative(offset)
    }

    #[inline]
    pub fn resolve(self, operand_base: usize) -> u16 {
        match self {
            Self::Absolute(slot) => slot,
            Self::OperandRelative(offset) => (operand_base + offset as usize) as u16,
        }
    }

    #[inline]
    pub const fn operand_relative_offset(self) -> Option<u16> {
        match self {
            Self::Absolute(_) => None,
            Self::OperandRelative(offset) => Some(offset),
        }
    }

    #[inline]
    pub fn expect_operand_relative(self, what: &str) -> u16 {
        match self {
            Self::Absolute(slot) => panic!("{what} must be operand-relative, got absolute slot {slot}"),
            Self::OperandRelative(offset) => offset,
        }
    }
}

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
