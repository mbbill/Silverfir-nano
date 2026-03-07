/// Backend-managed TOS cache state for compile-time planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TosState {
    spill_depth: usize,
    tos_register_count: usize,
}

impl TosState {
    #[inline]
    pub const fn new(tos_register_count: usize) -> Self {
        Self {
            spill_depth: 0,
            tos_register_count,
        }
    }

    #[inline]
    pub fn spill_depth(&self) -> usize {
        self.spill_depth
    }

    #[inline]
    pub fn depth_variant(&self, height: usize) -> u8 {
        if height == 0 {
            1
        } else {
            (((height - 1) % self.tos_register_count) + 1) as u8
        }
    }

    #[inline]
    pub fn min_spill_depth(&self, height: usize) -> usize {
        height.saturating_sub(self.tos_register_count)
    }

    #[inline]
    pub fn tos_count(&self, height: usize) -> usize {
        height.saturating_sub(self.spill_depth)
    }

    #[inline]
    pub fn needs_spill_before_push(&self, height: usize) -> bool {
        self.tos_count(height) >= self.tos_register_count
    }

    #[inline]
    pub fn needs_fill_before_control_flow(&self, height: usize) -> bool {
        self.spill_depth > self.min_spill_depth(height)
    }

    #[inline]
    pub fn record_spill(&mut self, height: usize, count: usize) {
        self.spill_depth += count;
        self.assert_invariants(height);
    }

    #[inline]
    pub fn record_fill(&mut self, height: usize, count: usize) {
        self.spill_depth = self.spill_depth.saturating_sub(count);
        self.assert_invariants(height);
    }

    #[inline]
    pub fn clamp_to_height(&mut self, height: usize) {
        if self.spill_depth > height {
            self.spill_depth = height;
        }
        self.assert_invariants(height);
    }

    #[inline]
    pub fn reset(&mut self, height: usize) {
        self.spill_depth = height;
        self.assert_invariants(height);
    }

    #[inline]
    pub fn normalize_for_control_flow(&mut self, height: usize) -> usize {
        let min = self.min_spill_depth(height);
        let fill_count = self.spill_depth.saturating_sub(min);
        self.spill_depth = min;
        self.assert_invariants(height);
        fill_count
    }

    #[inline]
    pub fn restore_preserved_depth(&mut self, height: usize, preserved_tos_depth: usize) {
        self.spill_depth = height.saturating_sub(preserved_tos_depth);
        self.assert_invariants(height);
    }

    #[inline]
    pub fn assert_invariants(&self, height: usize) {
        debug_assert!(
            self.spill_depth <= height,
            "spill_depth ({}) must be <= height ({})",
            self.spill_depth,
            height
        );
        debug_assert!(
            height - self.spill_depth <= self.tos_register_count,
            "TOS invariant violated: tos_count ({}) > TOS_REGISTER_COUNT ({})",
            height - self.spill_depth,
            self.tos_register_count
        );
    }
}
