//! Prepared operand wrappers.
//!
//! A `PreparedGp` / `PreparedFp` is a physical register that may or may not
//! own a scratch pool slot. Use `*prepared` to read the physical register for
//! encoding. The pool slot (if any) is freed on drop.

use core::ops::Deref;

use crate::vm::arch::common::scratch_pool::{DetachedScratch, ScratchGuard};

use super::reg::{Arm64FpReg, Arm64Reg};

pub(super) enum PreparedGp<'a> {
    Mapped(Arm64Reg),
    Scratch(ScratchGuard<'a, Arm64Reg, 2>),
}

pub(super) enum OwnedPreparedGp {
    Mapped(Arm64Reg),
    Scratch(DetachedScratch<Arm64Reg, 2>),
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
    type Target = Arm64Reg;

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
    type Target = Arm64Reg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}

pub(super) enum PreparedFp<'a> {
    Mapped(Arm64FpReg),
    Scratch(ScratchGuard<'a, Arm64FpReg, 2>),
}

pub(super) enum OwnedPreparedFp {
    Mapped(Arm64FpReg),
    Scratch(DetachedScratch<Arm64FpReg, 2>),
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
    type Target = Arm64FpReg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}

impl OwnedPreparedFp {
    /// True when this prepared value owns an FP scratch slot (i.e. the caller
    /// may clobber its physical register without destroying live JIT state).
    #[inline]
    pub(super) fn is_scratch(&self) -> bool {
        matches!(self, Self::Scratch(_))
    }
}

impl Deref for OwnedPreparedFp {
    type Target = Arm64FpReg;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g,
        }
    }
}
