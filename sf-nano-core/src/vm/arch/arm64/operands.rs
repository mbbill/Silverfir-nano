//! Prepared operand wrappers.
//!
//! A `PreparedGp` / `PreparedFp` is a physical register that may or may not
//! own a scratch pool slot. Call `.reg()` to read the register for encoding.
//! The pool slot (if any) is freed on drop, or explicitly via `.release()`.

use crate::vm::arch::common::scratch_pool::ScratchGuard;

use super::reg::{Arm64FpReg, Arm64Reg};

pub(super) enum PreparedGp<'a> {
    Mapped(Arm64Reg),
    Scratch(ScratchGuard<'a, Arm64Reg, 2>),
}

impl PreparedGp<'_> {
    #[inline]
    pub(super) fn reg(&self) -> Arm64Reg {
        match self {
            Self::Mapped(r) => *r,
            Self::Scratch(g) => **g,
        }
    }

    #[inline]
    pub(super) fn release(self) -> Arm64Reg {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g.release(),
        }
    }
}

pub(super) enum PreparedFp<'a> {
    Mapped(Arm64FpReg),
    Scratch(ScratchGuard<'a, Arm64FpReg, 3>),
}

impl PreparedFp<'_> {
    #[inline]
    pub(super) fn reg(&self) -> Arm64FpReg {
        match self {
            Self::Mapped(r) => *r,
            Self::Scratch(g) => **g,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(super) fn release(self) -> Arm64FpReg {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g.release(),
        }
    }
}
