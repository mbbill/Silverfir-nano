//! ARM64 physical register names.
//!
//! The native ABI can be layered on top of these later. For bring-up the entry
//! trampoline only needs one scratch register, but the register file is kept
//! explicit so later lowering can stay mechanical.

/// One ARM64 general-purpose register number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Arm64Reg {
    X0 = 0,
    X1 = 1,
    X2 = 2,
    X3 = 3,
    X4 = 4,
    X5 = 5,
    X6 = 6,
    X7 = 7,
    X8 = 8,
    X9 = 9,
    X10 = 10,
    X11 = 11,
    X12 = 12,
    X13 = 13,
    X14 = 14,
    X15 = 15,
    X16 = 16,
    X17 = 17,
    X18 = 18,
    X19 = 19,
    X20 = 20,
    X21 = 21,
    X22 = 22,
    X23 = 23,
    X24 = 24,
    X25 = 25,
    X26 = 26,
    X27 = 27,
    X28 = 28,
    X29 = 29,
    X30 = 30,
    Xzr = 31,
}

impl Arm64Reg {
    #[inline]
    pub const fn idx(self) -> u32 {
        self as u32
    }
}
