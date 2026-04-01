#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Arm64FpReg(u8);

impl Arm64FpReg {
    #[inline]
    pub(super) const fn from_raw(index: u8) -> Self {
        Self(index)
    }

    #[inline]
    pub(crate) const fn index(self) -> u32 {
        self.0 as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Arm64Reg(u8);

impl Arm64Reg {
    #[inline]
    pub(super) const fn from_raw(index: u8) -> Self {
        Self(index)
    }

    #[inline]
    pub(crate) const fn index(self) -> u32 {
        self.0 as u32
    }
}
