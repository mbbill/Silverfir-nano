//! Backend-facing targets.

/// Lowered target id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SsaTarget(pub u32);

impl SsaTarget {
    #[inline]
    pub(crate) const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub(crate) const fn as_usize(self) -> usize {
        self.0 as usize
    }
}
