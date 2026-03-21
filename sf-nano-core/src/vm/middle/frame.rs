//! Canonical frame layout planning.
//!
//! This is the shared Wasm/native contract:
//! - locals keep their Wasm-relative frame homes
//! - call scratch sits after locals
//! - deep stack / call payloads live in operand slots after call scratch

/// Explicit `fp[...]` frame-relative slot selected during preparation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct FrameSlot(pub u16);

impl FrameSlot {
    #[inline]
    pub(crate) const fn advance(self, slots: u16) -> Self {
        Self(self.0 + slots)
    }
}

/// Explicit contiguous `fp[...]` frame-relative range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameSpan {
    pub start: FrameSlot,
    pub count: u16,
}

impl FrameSpan {
    #[inline]
    pub(crate) const fn new(start: FrameSlot, count: u16) -> Self {
        Self { start, count }
    }

    #[inline]
    pub(crate) const fn single(slot: FrameSlot) -> Self {
        Self {
            start: slot,
            count: 1,
        }
    }

    #[inline]
    pub(crate) const fn end(self) -> FrameSlot {
        self.start.advance(self.count)
    }
}

/// Canonical frame layout for one function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameLayoutPlan {
    pub locals: FrameSpan,
    pub call_scratch: Option<FrameSpan>,
    pub operands: FrameSpan,
    /// Size of the callee-local frame prefix before native call scratch.
    pub frame_prefix_size: u16,
}

impl FrameLayoutPlan {
    #[inline]
    pub(crate) const fn local_slot(self, idx: u16) -> FrameSlot {
        self.locals.start.advance(idx)
    }

    #[inline]
    pub(crate) const fn operand_slot(self, idx: u16) -> FrameSlot {
        self.operands.start.advance(idx)
    }

    #[inline]
    pub(crate) const fn call_scratch_slots(self) -> u16 {
        match self.call_scratch {
            Some(span) => span.count,
            None => 0,
        }
    }

    #[inline]
    pub(crate) const fn total_slots(self) -> u16 {
        self.operands.end().0
    }

    #[inline]
    pub(crate) const fn return_results(self, count: u16) -> FrameSpan {
        FrameSpan::new(self.operand_slot(0), count)
    }
}

#[inline]
pub(crate) const fn plan_frame_layout(
    local_count: u16,
    max_stack_height: u16,
    call_scratch_slots: u16,
) -> FrameLayoutPlan {
    let locals = FrameSpan::new(FrameSlot(0), local_count);
    let frame_prefix_size = local_count;
    let call_scratch = if call_scratch_slots == 0 {
        None
    } else {
        Some(FrameSpan::new(
            FrameSlot(frame_prefix_size),
            call_scratch_slots,
        ))
    };
    let operand_base = frame_prefix_size + call_scratch_slots;
    FrameLayoutPlan {
        locals,
        call_scratch,
        operands: FrameSpan::new(FrameSlot(operand_base), max_stack_height),
        frame_prefix_size,
    }
}
