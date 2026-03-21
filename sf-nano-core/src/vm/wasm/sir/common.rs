//! Shared semantic/frontend ids and small types.
//!
//! This module must stay semantic-only. It must not expose backend frame-slot
//! or rotating-window placement details.

/// Semantic instruction index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SemanticIndex(u32);

impl SemanticIndex {
    #[inline]
    pub(crate) const fn new(raw: usize) -> Self {
        Self(raw as u32)
    }

    #[inline]
    pub(crate) const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Structured control-flow target in semantic space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SemanticTarget(SemanticIndex);

impl SemanticTarget {
    #[inline]
    pub(crate) const fn new(raw: usize) -> Self {
        Self(SemanticIndex::new(raw))
    }

    #[inline]
    pub(crate) const fn index(self) -> SemanticIndex {
        self.0
    }

    #[inline]
    pub(crate) const fn pending() -> Self {
        Self(SemanticIndex(u32::MAX))
    }

    #[inline]
    pub(crate) const fn is_pending(self) -> bool {
        self.0 .0 == u32::MAX
    }
}

/// Semantic branch-table entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BrTableEntry {
    pub target: SemanticTarget,
    pub stack_drop: u32,
    pub arity: u16,
}
