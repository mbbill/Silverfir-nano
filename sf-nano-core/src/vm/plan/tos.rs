//! Rotating T-window planning model.
//!
//! Important:
//! - the rotating cache is always valid
//! - planning decides where the logical top sits in the ring
//! - this module must not leak "live-count" or stack-height deduction into
//!   backend-facing IR

/// Rotation of the logical top within the 4-register ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosRotation(pub u8);

impl TosRotation {
    #[inline]
    pub const fn new(raw: u8) -> Self {
        Self(raw & 0b11)
    }

    /// Physical register index of the current logical top inside the rotating
    /// 4-register window.
    #[inline]
    pub const fn top_index(self) -> u8 {
        self.0 & 0b11
    }

    /// Physical register index for the `depth_from_top`-th value in the
    /// rotating window.
    #[inline]
    pub const fn nth_from_top(self, depth_from_top: u8) -> u8 {
        self.0.wrapping_sub(depth_from_top) & 0b11
    }

    /// Rotation after popping `count` values.
    #[inline]
    pub const fn after_pop(self, count: u8) -> Self {
        Self::new(self.0.wrapping_sub(count))
    }

    /// Rotation after pushing `count` values.
    #[inline]
    pub const fn after_push(self, count: u8) -> Self {
        Self::new(self.0.wrapping_add(count))
    }
}
