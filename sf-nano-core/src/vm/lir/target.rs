//! Backend-facing targets.
//!
//! Targets should refer to explicit lowered entries, not interpreter
//! instruction descriptors.

/// Lowered target id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LirTarget(pub u32);

impl LirTarget {
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}
