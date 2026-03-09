//! ARM64 register abstraction for the native backend.
//!
//! Maps the native backend VM ABI to ARM64 physical registers.
//!
//! The native backend keeps long-lived VM state in callee-saved registers so
//! future cold helper bridges can call normal-ABI Rust helpers without relying
//! on `preserve_none`.

/// ARM64 general-purpose register.
///
/// The discriminant is the 5-bit register number used in ARM64 encodings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    A0   = 0,  // x0   - helper call arg / result register
    A1   = 1,  // x1   - helper call arg / result register
    A2   = 2,  // x2   - helper call arg register
    TMP0 = 9,  // x9   - scratch
    TMP1 = 10, // x10  - scratch
    TMP2 = 11, // x11  - native-private scratch / frame-alias register 0
    TMP3 = 12, // x12  - native-private scratch / frame-alias register 1
    TMP4 = 13, // x13  - native-private group-local cache register 0
    TMP5 = 14, // x14  - native-private group-local cache register 1
    CTX = 19,  // x19  - native context pointer
    PC  = 20,  // x20  - program counter (NativeInst*)
    FP  = 21,  // x21  - frame pointer (locals array)
    L0  = 22,  // x22  - hot local 0
    L1  = 23,  // x23  - hot local 1
    L2  = 24,  // x24  - hot local 2
    T0  = 25,  // x25  - TOS register 0
    T1  = 26,  // x26  - TOS register 1
    T2  = 27,  // x27  - TOS register 2
    T3  = 28,  // x28  - TOS register 3
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
