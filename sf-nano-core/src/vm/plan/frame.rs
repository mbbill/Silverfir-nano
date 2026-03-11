//! Frame layout planning.
//!
//! This is where logical locals / operands become `fp[...]`-relative memory
//! concepts. Backends should consume the result, not recompute it.

/// Explicit `fp[...]` frame-relative slot selected during planning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameSlot(pub u16);

impl FrameSlot {
    #[inline]
    pub const fn advance(self, slots: u16) -> Self {
        Self(self.0 + slots)
    }
}

/// Explicit contiguous `fp[...]` frame-relative range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameSpan {
    pub start: FrameSlot,
    pub count: u16,
}

impl FrameSpan {
    #[inline]
    pub const fn new(start: FrameSlot, count: u16) -> Self {
        Self { start, count }
    }

    #[inline]
    pub const fn single(slot: FrameSlot) -> Self {
        Self {
            start: slot,
            count: 1,
        }
    }

    #[inline]
    pub const fn end(self) -> FrameSlot {
        self.start.advance(self.count)
    }
}

/// Planned frame layout for one function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameLayoutPlan {
    pub local_base: u16,
    pub operand_base: u16,
    pub locals: FrameSpan,
    pub operands: FrameSpan,
    pub call_scratch: Option<FrameSpan>,
    pub backend_reserved: Option<FrameSpan>,
    /// Size of the callee-local frame prefix before native call metadata.
    ///
    /// For the native backend this includes Wasm locals plus any backend-owned
    /// frame homes, but excludes call-link metadata slots that sit between the
    /// frame prefix and the operand area.
    pub frame_size: u16,
}

impl FrameLayoutPlan {
    #[inline]
    pub const fn local_slot(self, idx: u16) -> FrameSlot {
        FrameSlot(self.local_base + idx)
    }

    #[inline]
    pub const fn operand_slot(self, idx: u16) -> FrameSlot {
        FrameSlot(self.operand_base + idx)
    }
}

/// Incremental frame planner used before backend-facing IR exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramePlanner {
    local_count: u16,
    backend_reserved: u16,
    call_scratch: u16,
    next_operand: u16,
}

impl FramePlanner {
    #[inline]
    pub const fn new(local_count: u16) -> Self {
        Self {
            local_count,
            backend_reserved: 0,
            call_scratch: 0,
            next_operand: 0,
        }
    }

    #[inline]
    pub const fn reserve_backend_reserved(mut self, count: u16) -> Self {
        self.backend_reserved += count;
        self
    }

    #[inline]
    pub const fn reserve_call_scratch(mut self, count: u16) -> Self {
        self.call_scratch += count;
        self
    }

    #[inline]
    pub const fn reserve_operands(mut self, count: u16) -> (Self, FrameSpan) {
        let span = FrameSpan::new(
            FrameSlot(
                self.local_count + self.backend_reserved + self.call_scratch + self.next_operand,
            ),
            count,
        );
        self.next_operand += count;
        (self, span)
    }

    #[inline]
    pub const fn finish(self) -> FrameLayoutPlan {
        let frame_size = self.local_count + self.backend_reserved;
        let operand_base = frame_size + self.call_scratch;
        FrameLayoutPlan {
            local_base: 0,
            operand_base,
            locals: FrameSpan::new(FrameSlot(0), self.local_count),
            operands: FrameSpan::new(FrameSlot(operand_base), self.next_operand),
            call_scratch: if self.call_scratch == 0 {
                None
            } else {
                Some(FrameSpan::new(FrameSlot(frame_size), self.call_scratch))
            },
            backend_reserved: if self.backend_reserved == 0 {
                None
            } else {
                Some(FrameSpan::new(
                    FrameSlot(self.local_count),
                    self.backend_reserved,
                ))
            },
            frame_size,
        }
    }
}

#[inline]
pub fn plan_frame_layout(
    local_count: u16,
    max_stack_height: u16,
    hot_local_count: u8,
    call_scratch_slots: u16,
) -> FrameLayoutPlan {
    let backend_reserved = (hot_local_count as u16).saturating_sub(local_count);
    let planner = FramePlanner::new(local_count)
        .reserve_backend_reserved(backend_reserved)
        .reserve_call_scratch(call_scratch_slots);
    let (planner, _) = planner.reserve_operands(max_stack_height);
    planner.finish()
}
