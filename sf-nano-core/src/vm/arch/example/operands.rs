//! Prepared operand wrappers.
//!
//! A `PreparedGp` / `PreparedFp` is a physical register that may or may not
//! own a scratch pool slot. Use `*prepared` to read the physical register for
//! encoding. The pool slot (if any) is freed on drop.

use std::ops::Deref;

use crate::vm::arch::common::scratch_pool::{DetachedScratch, ScratchGuard};

use super::reg::{FpReg, GpReg};

pub(super) enum PreparedGp<'a> {
    Mapped(GpReg),
    Scratch(ScratchGuard<'a, GpReg, 2>),
}

pub(super) enum OwnedPreparedGp {
    Mapped(GpReg),
    Scratch(DetachedScratch<GpReg, 2>),
}

impl PreparedGp<'_> {
    #[inline]
    pub(super) fn detach(self) -> OwnedPreparedGp {
        match self {
            Self::Mapped(r) => OwnedPreparedGp::Mapped(r),
            Self::Scratch(g) => OwnedPreparedGp::Scratch(g.detach()),
        }
    }
}

impl Deref for PreparedGp<'_> {
    type Target = GpReg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}

impl OwnedPreparedGp {}

impl Deref for OwnedPreparedGp {
    type Target = GpReg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}

pub(super) enum PreparedFp<'a> {
    Mapped(FpReg),
    Scratch(ScratchGuard<'a, FpReg, 3>),
}

pub(super) enum OwnedPreparedFp {
    Mapped(FpReg),
    Scratch(DetachedScratch<FpReg, 3>),
}

impl PreparedFp<'_> {
    #[inline]
    pub(super) fn detach(self) -> OwnedPreparedFp {
        match self {
            Self::Mapped(r) => OwnedPreparedFp::Mapped(r),
            Self::Scratch(g) => OwnedPreparedFp::Scratch(g.detach()),
        }
    }
}

impl Deref for PreparedFp<'_> {
    type Target = FpReg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}

impl OwnedPreparedFp {}

impl Deref for OwnedPreparedFp {
    type Target = FpReg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}
