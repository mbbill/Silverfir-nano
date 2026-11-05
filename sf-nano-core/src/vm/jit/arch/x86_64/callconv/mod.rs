//! Selected x86_64 C calling convention.
//!
//! This module is the single place in `arch/x86_64/` where the Win64 vs
//! System V AMD64 difference is decided. The rest of the backend imports
//! `super::callconv::*` and uses the same names regardless of which ABI
//! is active.
//!
//! The two implementations export identical symbols:
//!
//! | symbol                          | kind        | description                           |
//! |---------------------------------|-------------|---------------------------------------|
//! | `C_ARG0` .. `C_ARG2`, `C_RET0`  | const X86Reg| C ABI argument / return registers     |
//! | `CALLEE_SAVED_GP`               | const slice | GP registers the ABI says we must save|
//! | `CALLEE_SAVED_XMM`              | const slice | XMM registers a C callee preserves    |
//! | `CALL_SHADOW_BYTES`             | const u32   | callee home area reserved at RSP      |
//! | `STACK_PADDING`                 | const u32   | extra bytes so RSP stays 16-aligned   |
//! | `emit_prologue_extra`           | fn          | extra spills after GP saves (XMM6-15) |
//! | `emit_epilogue_extra`           | fn          | extra restores before GP pops         |
//!
//! Adding another x86_64 ABI variant (e.g. the Microsoft "vectorcall"
//! convention, or a hypothetical future one) is a matter of adding another
//! submodule that exports the same names and extending the `cfg` below.

#[cfg(sf_os_windows)]
mod win64;
#[cfg(sf_os_windows)]
pub(super) use win64::*;

#[cfg(not(sf_os_windows))]
mod sysv;
#[cfg(not(sf_os_windows))]
pub(super) use sysv::*;
