//! Arch-specific glue for the Pico 2 firmware.
//!
//! RP2350 boots either as 2× Cortex-M33 (ARM) or 2× Hazard3 (RV32IMAC).
//! Same chip, same memory map, same peripherals — but a different runtime
//! crate (cortex-m-rt vs riscv-rt) and different cpu-barrier intrinsics.
//! This module is the single seam that switches between them.
//!
//! `rp235x-hal`'s `entry` macro and `ImageDef::secure_exe()` are already
//! arch-aware (they dispatch on `target_arch`), so binaries write
//! `#[hal::entry]` and `ImageDef::secure_exe()` unchanged for both
//! targets — no shim needed for those.

#[cfg(target_arch = "arm")]
mod arm;
#[cfg(target_arch = "arm")]
pub use arm::*;

#[cfg(target_arch = "riscv32")]
mod rv;
#[cfg(target_arch = "riscv32")]
pub use rv::*;
