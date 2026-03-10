//! Spill/fill planning.
//!
//! This module owns explicit insertion of spill/fill artifacts needed to keep
//! the rotating T-window valid.

use super::frame::FrameSpan;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheTransferDirection {
    Spill,
    Fill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheTransfer {
    pub direction: CacheTransferDirection,
    pub frame: FrameSpan,
}

/// Planning-stage spill/fill artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillArtifact {
    CacheTransfer(CacheTransfer),
}

impl SpillArtifact {
    #[inline]
    pub const fn spill(frame: FrameSpan) -> Self {
        Self::CacheTransfer(CacheTransfer {
            direction: CacheTransferDirection::Spill,
            frame,
        })
    }

    #[inline]
    pub const fn fill(frame: FrameSpan) -> Self {
        Self::CacheTransfer(CacheTransfer {
            direction: CacheTransferDirection::Fill,
            frame,
        })
    }
}
