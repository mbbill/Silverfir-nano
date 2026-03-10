//! ARM64 register assignment for the native ABI.

/// ARM64 register in the native ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm64Reg {
    Ctx,
    Fp,
    Hot0,
    Hot1,
    Hot2,
    Tos0,
    Tos1,
    Tos2,
    Tos3,
    Tmp0,
    Tmp1,
    Tmp2,
    Tmp3,
}
