/// x86_64 general-purpose registers.
///
/// Encoding uses the standard 4-bit register numbering (0-15) from the
/// Intel/AMD manuals. Registers R8-R15 require REX prefix extension bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum X86Reg {
    RAX = 0,
    RCX = 1,
    RDX = 2,
    RBX = 3,
    RSP = 4,
    RBP = 5,
    RSI = 6,
    RDI = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl X86Reg {
    /// The 3-bit register field for ModR/M and SIB encoding.
    #[inline]
    pub(crate) const fn idx3(self) -> u8 {
        (self as u8) & 0x7
    }

    /// Whether this register requires the REX.B / REX.R / REX.X extension bit.
    #[inline]
    pub(crate) const fn needs_rex_ext(self) -> bool {
        (self as u8) >= 8
    }

    /// Full 4-bit register number.
    #[inline]
    pub(crate) const fn idx(self) -> u8 {
        self as u8
    }
}
