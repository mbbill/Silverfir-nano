//! Physical register model for the example backend.
//!
//! This file owns hardware facts only. It does not know about MachineIR,
//! scratch policy, ABI mapping, or prologue logic.
//!
//! The `gp` and `fp` submodules are `pub(super)` — only `abi.rs` should
//! reference hardware register names directly. All other files go through
//! named ABI roles.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GpReg(u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FpReg(u8);

impl GpReg {
    /// Stack pointer (encoding 31, context-dependent).
    pub(crate) const SP: Self = Self(31);
    /// Zero register (encoding 31, context-dependent).
    pub(crate) const ZR: Self = Self(31);
    /// Link register.
    pub(crate) const LR: Self = Self(30);

    #[inline]
    pub(crate) const fn index(self) -> u8 {
        self.0
    }
}

impl FpReg {
    #[inline]
    pub(crate) const fn index(self) -> u8 {
        self.0
    }
}

/// Hardware GP register names. Only `abi.rs` should import this module.
pub(super) mod gp {
    use super::GpReg;

    pub(crate) const X0: GpReg = GpReg(0);
    pub(crate) const X1: GpReg = GpReg(1);
    pub(crate) const X2: GpReg = GpReg(2);
    pub(crate) const X9: GpReg = GpReg(9);
    pub(crate) const X10: GpReg = GpReg(10);
    pub(crate) const X16: GpReg = GpReg(16);
    pub(crate) const X17: GpReg = GpReg(17);
    pub(crate) const X19: GpReg = GpReg(19);
    pub(crate) const X20: GpReg = GpReg(20);
    pub(crate) const X21: GpReg = GpReg(21);
    pub(crate) const X22: GpReg = GpReg(22);
    pub(crate) const X23: GpReg = GpReg(23);
    pub(crate) const X24: GpReg = GpReg(24);
    pub(crate) const X25: GpReg = GpReg(25);
    pub(crate) const X26: GpReg = GpReg(26);
    pub(crate) const X27: GpReg = GpReg(27);
    pub(crate) const X28: GpReg = GpReg(28);
    pub(crate) const X29: GpReg = GpReg(29);
    pub(crate) const X30: GpReg = GpReg(30);
}

/// Hardware FP register names. Only `abi.rs` should import this module.
pub(super) mod fp {
    use super::FpReg;

    pub(crate) const D0: FpReg = FpReg(0);
    pub(crate) const D1: FpReg = FpReg(1);
    pub(crate) const D2: FpReg = FpReg(2);
    pub(crate) const D8: FpReg = FpReg(8);
    pub(crate) const D9: FpReg = FpReg(9);
    pub(crate) const D10: FpReg = FpReg(10);
    pub(crate) const D11: FpReg = FpReg(11);
    pub(crate) const D12: FpReg = FpReg(12);
    pub(crate) const D13: FpReg = FpReg(13);
    pub(crate) const D14: FpReg = FpReg(14);
    pub(crate) const D15: FpReg = FpReg(15);
    pub(crate) const D16: FpReg = FpReg(16);
    pub(crate) const D17: FpReg = FpReg(17);
    pub(crate) const D18: FpReg = FpReg(18);
    pub(crate) const D19: FpReg = FpReg(19);
}
