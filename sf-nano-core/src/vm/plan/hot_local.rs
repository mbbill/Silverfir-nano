//! Hot-local planning.
//!
//! This module chooses which locals live in the dedicated hot-local register
//! class. It must remain entirely on the planning side.

/// Number of dedicated hot-local registers in the current design.
pub const HOT_LOCAL_COUNT: usize = 3;

/// Planned hot-local mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HotLocalPlan {
    raw: [Option<u32>; HOT_LOCAL_COUNT],
    effective: [Option<u32>; HOT_LOCAL_COUNT],
}

impl HotLocalPlan {
    #[inline]
    pub const fn new(
        raw: [Option<u32>; HOT_LOCAL_COUNT],
        effective: [Option<u32>; HOT_LOCAL_COUNT],
    ) -> Self {
        Self { raw, effective }
    }

    #[inline]
    pub const fn raw(self) -> [Option<u32>; HOT_LOCAL_COUNT] {
        self.raw
    }

    #[inline]
    pub const fn effective(self) -> [Option<u32>; HOT_LOCAL_COUNT] {
        self.effective
    }

    #[inline]
    pub fn resolve(self, local_idx: u32) -> Option<u8> {
        let mut i = 0;
        while i < HOT_LOCAL_COUNT {
            if self.effective[i] == Some(local_idx) {
                return Some(i as u8);
            }
            i += 1;
        }
        None
    }
}
