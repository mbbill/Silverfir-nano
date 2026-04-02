//! Prepared operand wrappers.
//!
//! A `PreparedGp` / `PreparedFp` is a physical register that may or may not
//! own a scratch pool slot. Use `*prepared` to read the physical register for
//! encoding. The pool slot (if any) is freed on drop.

use std::ops::Deref;

use crate::vm::arch::common::scratch_pool::{DetachedScratch, ScratchGuard};

use super::reg::Arm32Reg;

pub(super) enum PreparedGp<'a> {
    Mapped(Arm32Reg),
    Scratch(ScratchGuard<'a, Arm32Reg, 2>),
}

pub(super) enum OwnedPreparedGp {
    Mapped(Arm32Reg),
    Scratch(DetachedScratch<Arm32Reg, 2>),
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
    type Target = Arm32Reg;

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
    type Target = Arm32Reg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}

pub(super) enum PreparedFp<'a> {
    Mapped(u32),
    Scratch(ScratchGuard<'a, u32, 3>),
}

pub(super) enum OwnedPreparedFp {
    Mapped(u32),
    Scratch(DetachedScratch<u32, 3>),
}

impl PreparedFp<'_> {
    #[inline]
    pub(super) fn detach(self) -> OwnedPreparedFp {
        match self {
            Self::Mapped(d) => OwnedPreparedFp::Mapped(d),
            Self::Scratch(g) => OwnedPreparedFp::Scratch(g.detach()),
        }
    }
}

impl Deref for PreparedFp<'_> {
    type Target = u32;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(d) => d,
            Self::Scratch(g) => g,
        }
    }
}

impl OwnedPreparedFp {}

impl Deref for OwnedPreparedFp {
    type Target = u32;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(d) => d,
            Self::Scratch(g) => g,
        }
    }
}
