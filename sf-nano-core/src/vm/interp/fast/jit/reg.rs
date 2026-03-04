//! ARM64 register abstraction for the micro-JIT.
//!
//! Maps interpreter ABI registers to ARM64 physical registers.
//! Under preserve_none: ctx=x20, pc=x21, fp=x22, l0-l2=x23-x25,
//! t0-t2=x26-x28, t3=x0, nh=x1.

/// ARM64 general-purpose register.
///
/// The discriminant is the 5-bit register number used in ARM64 encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    // Interpreter ABI registers (preserve_none on ARM64)
    T3  = 0,   // x0  - TOS register 3
    NH  = 1,   // x1  - next handler preload
    TMP0 = 2,  // x2  - scratch (trap msg, address computation)
    TMP1 = 3,  // x3  - scratch
    CTX = 20,  // x20 - context pointer
    PC  = 21,  // x21 - program counter (Instruction*)
    FP  = 22,  // x22 - frame pointer (locals array)
    L0  = 23,  // x23 - hot local 0
    L1  = 24,  // x24 - hot local 1
    L2  = 25,  // x25 - hot local 2
    T0  = 26,  // x26 - TOS register 0
    T1  = 27,  // x27 - TOS register 1
    T2  = 28,  // x28 - TOS register 2
    LR  = 30,  // x30 - link register
    XZR = 31,  // xzr/wzr - zero register (or SP in some contexts)
}

impl Reg {
    /// Returns the 5-bit register number for ARM64 instruction encoding.
    #[inline]
    pub const fn idx(self) -> u32 {
        self as u32
    }
}
