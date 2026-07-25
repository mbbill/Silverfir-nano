//! Shared RISC-V building blocks used by RV64 and RV32 backends.
//!
//! This module deliberately contains only ISA-neutral pieces: register
//! identities, base encoders, and the common register allocation plan. XLEN
//! policy, stack layout, literal width, and pair-valued i64 lowering stay in
//! the concrete `riscv64` / `riscv32` backends.

pub(crate) mod abi;
pub(crate) mod enc;
pub(crate) mod reg;
