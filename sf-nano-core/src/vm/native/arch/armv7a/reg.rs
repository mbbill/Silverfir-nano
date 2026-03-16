#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Arm32Reg {
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
    R4 = 4,
    R5 = 5,
    R6 = 6,
    R7 = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Arm32Reg {
    pub const SP: Self = Self::R13;
    pub const LR: Self = Self::R14;
    pub const PC: Self = Self::R15;
    pub const IP: Self = Self::R12;

    #[inline]
    pub const fn idx(self) -> u32 {
        self as u32
    }

    #[inline]
    pub fn from_idx(idx: u32) -> Self {
        match idx {
            0 => Self::R0,
            1 => Self::R1,
            2 => Self::R2,
            3 => Self::R3,
            4 => Self::R4,
            5 => Self::R5,
            6 => Self::R6,
            7 => Self::R7,
            8 => Self::R8,
            9 => Self::R9,
            10 => Self::R10,
            11 => Self::R11,
            12 => Self::R12,
            13 => Self::R13,
            14 => Self::R14,
            15 => Self::R15,
            _ => Self::R0,
        }
    }
}
