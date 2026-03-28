//! Prepared operand wrappers.
//!
//! A `PreparedGp` / `PreparedFp` is a physical register that may or may not
//! own a scratch pool slot. Call `.reg()` to read the register for encoding.
//! The pool slot (if any) is freed on drop, or explicitly via `.release()`.

use crate::vm::arch::common::scratch_pool::ScratchGuard;

use super::reg::Arm32Reg;

pub(super) enum PreparedGp<'a> {
    Mapped(Arm32Reg),
    Scratch(ScratchGuard<'a, Arm32Reg, 2>),
}

impl PreparedGp<'_> {
    #[inline]
    pub(super) fn reg(&self) -> Arm32Reg {
        match self {
            Self::Mapped(r) => *r,
            Self::Scratch(g) => **g,
        }
    }

    #[inline]
    pub(super) fn release(self) -> Arm32Reg {
        match self {
            Self::Mapped(r) => r,
            Self::Scratch(g) => g.release(),
        }
    }
}

pub(super) enum PreparedFp<'a> {
    Mapped(u32),
    Scratch(ScratchGuard<'a, u32, 3>),
}

impl PreparedFp<'_> {
    /// Physical D-register number.
    #[inline]
    pub(super) fn reg(&self) -> u32 {
        match self {
            Self::Mapped(d) => *d,
            Self::Scratch(g) => **g,
        }
    }

    #[inline]
    pub(super) fn release(self) -> u32 {
        match self {
            Self::Mapped(d) => d,
            Self::Scratch(g) => g.release(),
        }
    }
}
