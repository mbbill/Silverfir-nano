//! Prepared operand wrappers.
//!
//! A `PreparedGp` / `PreparedFp` is a physical register that may or may not
//! own a scratch pool slot. Call `.reg()` to read the register for encoding.
//! The pool slot (if any) is freed on drop, or explicitly via `.release()`.

use crate::vm::arch::common::scratch_pool::ScratchGuard;

use super::reg::{FpReg, GpReg};

pub(super) enum PreparedGp<'a> {
    Mapped(GpReg),
    Scratch(ScratchGuard<'a, GpReg, 2>),
}

impl PreparedGp<'_> {
    #[inline]
    pub(super) fn reg(&self) -> GpReg {
        match self {
            Self::Mapped(r) => *r,
            Self::Scratch(g) => **g,
        }
    }

    #[inline]
    pub(super) fn release(self) -> GpReg {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g.release(),
        }
    }
}

pub(super) enum PreparedFp<'a> {
    Mapped(FpReg),
    Scratch(ScratchGuard<'a, FpReg, 3>),
}

impl PreparedFp<'_> {
    #[inline]
    pub(super) fn reg(&self) -> FpReg {
        match self {
            Self::Mapped(r) => *r,
            Self::Scratch(g) => **g,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(super) fn release(self) -> FpReg {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g.release(),
        }
    }
}
